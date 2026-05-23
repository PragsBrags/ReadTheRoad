"""
Ingestion Service — Video source management and FastAPI API.

Orchestrates: source → FFmpegDecoder → FrameSampler → TaskDispatcher.
Supports video file uploads, RTSP streams, and RTMP streams.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, field_validator

from services.config import PipelineConfig
from services.decoder import FFmpegDecoder, FrameData
from services.monitoring import MetricsCollector
from services.sampler import FrameSampler
from services.task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


# REQUEST / RESPONSE MODELS

# Allowed video file extensions for upload validation
_ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp",
}
# Maximum upload file size (500 MB)
_MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024


class IngestStreamRequest(BaseModel):
    """Request to ingest a stream (RTSP/RTMP)."""
    url: str = Field(..., description="RTSP or RTMP stream URL")
    stream_id: Optional[str] = Field(None, description="Optional custom stream ID")
    continuous: bool = Field(False, description="Keep processing stream continuously")

    @field_validator("url")
    @classmethod
    def validate_stream_url(cls, v: str) -> str:
        """Validate that URL is a supported stream protocol (prevents SSRF)."""
        v = v.strip()
        if not v.startswith(("rtsp://", "rtmp://")):
            raise ValueError(
                "URL must be an RTSP or RTMP stream "
                "(e.g., rtsp://... or rtmp://...)"
            )
        return v


class IngestResponse(BaseModel):
    """Response from ingestion request."""
    job_id: str
    source: str
    status: str
    frames_extracted: int = 0
    frames_sampled: int = 0
    frames_processed: int = 0
    plates: list[dict[str, Any]] = []
    inference_mode: str = ""
    processing_mode: str = ""  # "local" | "distributed"
    message: str = ""


class StreamStatus(BaseModel):
    """Status of an active stream."""
    stream_id: str
    source: str
    status: str  # active | stopped | error
    frames_processed: int = 0
    started_at: float = 0.0
    error: Optional[str] = None


# STREAM TRACKER

@dataclass
class ActiveStream:
    """Tracks state of an active stream ingestion."""
    stream_id: str
    source: str
    status: str = "active"
    frames_processed: int = 0
    started_at: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None
    error: Optional[str] = None


# INGESTION SERVICE

class IngestionService:
    """
    Video ingestion service.

    Manages video file uploads and stream connections.
    Orchestrates the decode → sample → dispatch pipeline.
    Integrates aggregation for multi-frame dedup and cache for result storage.
    """

    def __init__(
        self,
        config: PipelineConfig,
        decoder: FFmpegDecoder,
        sampler: FrameSampler,
        dispatcher: TaskDispatcher,
        metrics: Optional[MetricsCollector] = None,
        cache=None,
        aggregation=None,
    ):
        self._config = config
        self._decoder = decoder
        self._sampler = sampler
        self._dispatcher = dispatcher
        self._metrics = metrics
        self._cache = cache
        self._aggregation = aggregation
        self._active_streams: dict[str, ActiveStream] = {}
        self._upload_dir = Path(config.video_ingestion.upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._frame_dir = Path(config.video_ingestion.frame_output_dir)
        self._frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_streams(self) -> dict[str, ActiveStream]:
        return self._active_streams

    def _save_sampled_frame(self, frame: FrameData, job_id: str, index: int) -> None:
        """Save a sampled frame to the configured output directory."""
        filename = f"sampled_{job_id}_{index:06d}.jpg"
        file_path = self._frame_dir / filename
        try:
            with open(file_path, "wb") as f:
                f.write(frame.frame_bytes)
            logger.debug(f"Saved sampled frame: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save sampled frame {file_path}: {e}")

    async def ingest_file(self, file: UploadFile) -> IngestResponse:
        """
        Ingest a video file upload.

        Pipeline: save file → validate → FFmpeg decode → sample → dispatch → aggregate
        """
        job_id = str(uuid.uuid4())
        start = time.time()

        # --- Validate file type ---
        filename = file.filename or "upload"
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{ext}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_VIDEO_EXTENSIONS))}"
                ),
            )

        # Save uploaded file
        file_path = self._upload_dir / f"{job_id}_{filename}"
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

        # --- Validate file size ---
        file_size = file_path.stat().st_size
        if file_size > _MAX_UPLOAD_SIZE_BYTES:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({file_size / 1024 / 1024:.1f} MB). "
                    f"Maximum: {_MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB"
                ),
            )

        logger.info(f"[{job_id}] Saved upload: {file_path} ({file_size} bytes)")

        try:
            # 1) FFmpeg decode (MANDATORY)
            frames = self._decoder.extract_frames_sync(
                source=str(file_path),
                output_dir=str(self._upload_dir / f"frames_{job_id}"),
            )

            # 2) Sample frames
            sampled = self._sampler.sample(frames)

            # 3) Save sampled frames to disk
            for i, frame in enumerate(sampled):
                self._save_sampled_frame(frame, job_id, i)

            # Store total frames in cache before dispatching
            if self._cache:
                self._cache.set_job_total_frames(job_id, len(sampled))

            # 4) Dispatch frames for inference
            processing_mode, inference_results = self._dispatcher.dispatch_frames(
                sampled, job_id=job_id
            )

            if processing_mode == "distributed":
                elapsed = (time.time() - start) * 1000
                logger.info(
                    f"[{job_id}] Queued for distributed processing: {len(frames)} extracted → "
                    f"{len(sampled)} sampled ({elapsed:.0f}ms)"
                )
                return IngestResponse(
                    job_id=job_id,
                    source=filename,
                    status="processing",
                    frames_extracted=len(frames),
                    frames_sampled=len(sampled),
                    frames_processed=0,
                    plates=[],
                    inference_mode=self._config.inference.mode.value,
                    processing_mode=processing_mode,
                    message=f"Job queued for distributed processing. {len(sampled)} frames dispatched.",
                )

            # 5) Aggregate results (multi-frame dedup + confidence voting)
            all_plates: list[dict[str, Any]] = []
            if self._aggregation and inference_results:
                for result in inference_results:
                    self._aggregation.add_result(job_id, result)
                all_plates = self._aggregation.flush(job_id)
            else:
                # Fallback: collect plates without aggregation
                for result in inference_results:
                    for plate in result.get("plates", []):
                        all_plates.append(plate)

            # 6) Store results in cache for /results/{job_id} retrieval
            if self._cache and all_plates:
                self._cache.store_job_results(job_id, all_plates)

            elapsed = (time.time() - start) * 1000
            if self._metrics:
                self._metrics.record_ingestion_latency(elapsed / 1000)
                self._metrics.increment_frames_processed(len(sampled))

            logger.info(
                f"[{job_id}] Complete: {len(frames)} extracted → "
                f"{len(sampled)} sampled → {len(all_plates)} plates "
                f"({processing_mode} mode, {elapsed:.0f}ms)"
            )

            return IngestResponse(
                job_id=job_id,
                source=filename,
                status="completed",
                frames_extracted=len(frames),
                frames_sampled=len(sampled),
                frames_processed=len(inference_results),
                plates=all_plates,
                inference_mode=self._config.inference.mode.value,
                processing_mode=processing_mode,
                message=(
                    f"{len(all_plates)} plates detected from "
                    f"{len(sampled)} frames in {elapsed:.0f}ms "
                    f"({processing_mode})"
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{job_id}] Ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def ingest_stream(self, request: IngestStreamRequest) -> IngestResponse:
        """Start ingesting an RTSP/RTMP stream."""
        stream_id = request.stream_id or str(uuid.uuid4())

        if len(self._active_streams) >= self._config.video_ingestion.max_streams:
            raise HTTPException(
                status_code=429,
                detail=f"Max streams ({self._config.video_ingestion.max_streams}) reached",
            )

        stream = ActiveStream(stream_id=stream_id, source=request.url)
        self._active_streams[stream_id] = stream

        if self._metrics:
            self._metrics.set_active_streams(len(self._active_streams))

        if request.continuous:
            task = asyncio.create_task(
                self._process_continuous_stream(stream)
            )
            stream.task = task
            return IngestResponse(
                job_id=stream_id,
                source=request.url,
                status="streaming",
                processing_mode=self._dispatcher.mode,
                message="Continuous stream started",
            )
        else:
            return await self._process_stream_once(stream)

    async def _process_stream_once(self, stream: ActiveStream) -> IngestResponse:
        """Process a stream once (non-continuous)."""
        start = time.time()

        try:
            frames: list[FrameData] = []
            async for frame in self._decoder.extract_frames(
                source=stream.source,
                output_dir=str(self._upload_dir / f"frames_{stream.stream_id}"),
            ):
                frames.append(frame)

            sampled = self._sampler.sample(frames)

            for i, frame in enumerate(sampled):
                self._save_sampled_frame(frame, stream.stream_id, i)

            # Store total frames in cache before dispatching
            if self._cache:
                self._cache.set_job_total_frames(stream.stream_id, len(sampled))

            processing_mode, inference_results = self._dispatcher.dispatch_frames(
                sampled, job_id=stream.stream_id
            )

            if processing_mode == "distributed":
                stream.status = "processing"
                stream.frames_processed = 0
                elapsed = (time.time() - start) * 1000
                return IngestResponse(
                    job_id=stream.stream_id,
                    source=stream.source,
                    status="processing",
                    frames_extracted=len(frames),
                    frames_sampled=len(sampled),
                    frames_processed=0,
                    plates=[],
                    inference_mode=self._config.inference.mode.value,
                    processing_mode=processing_mode,
                    message=f"Stream queued for distributed processing. {len(sampled)} frames dispatched.",
                )

            all_plates: list[dict[str, Any]] = []
            if self._aggregation and inference_results:
                for result in inference_results:
                    self._aggregation.add_result(stream.stream_id, result)
                all_plates = self._aggregation.flush(stream.stream_id)
            else:
                for result in inference_results:
                    for plate in result.get("plates", []):
                        all_plates.append(plate)

            # Store results in cache for /results/{job_id} retrieval
            if self._cache and all_plates:
                self._cache.store_job_results(stream.stream_id, all_plates)

            stream.status = "completed"
            stream.frames_processed = len(sampled)
            elapsed = (time.time() - start) * 1000

            if self._metrics:
                self._metrics.record_ingestion_latency(elapsed / 1000)
                self._metrics.increment_frames_processed(len(sampled))

            return IngestResponse(
                job_id=stream.stream_id,
                source=stream.source,
                status="completed",
                frames_extracted=len(frames),
                frames_sampled=len(sampled),
                frames_processed=len(inference_results),
                plates=all_plates,
                inference_mode=self._config.inference.mode.value,
                processing_mode=processing_mode,
                message=f"{len(all_plates)} plates in {elapsed:.0f}ms ({processing_mode})",
            )

        except Exception as e:
            stream.status = "error"
            stream.error = str(e)
            logger.error(f"[{stream.stream_id}] Stream processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            del self._active_streams[stream.stream_id]
            if self._metrics:
                self._metrics.set_active_streams(len(self._active_streams))

    async def _process_continuous_stream(self, stream: ActiveStream) -> None:
        """Process a stream continuously until stopped."""
        logger.info(f"[{stream.stream_id}] Starting continuous stream: {stream.source}")

        try:
            while stream.status == "active":
                try:
                    async for frame in self._decoder.extract_frames(
                        source=stream.source,
                    ):
                        if stream.status != "active":
                            break

                        if self._sampler.strategy.should_sample(
                            frame, stream.frames_processed
                        ):
                            self._save_sampled_frame(
                                frame, stream.stream_id, stream.frames_processed
                            )
                            self._dispatcher.dispatch_frame(
                                frame, job_id=stream.stream_id
                            )
                            stream.frames_processed += 1

                            if self._metrics:
                                self._metrics.increment_frames_processed(1)

                except Exception as e:
                    logger.error(
                        f"[{stream.stream_id}] Stream error (will retry): {e}"
                    )
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info(f"[{stream.stream_id}] Stream cancelled")
        finally:
            stream.status = "stopped"
            if stream.stream_id in self._active_streams:
                del self._active_streams[stream.stream_id]
            if self._metrics:
                self._metrics.set_active_streams(len(self._active_streams))

    async def stop_stream(self, stream_id: str) -> dict:
        """Stop an active stream."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

        stream.status = "stopping"
        if stream.task:
            stream.task.cancel()
            try:
                await stream.task
            except asyncio.CancelledError:
                pass

        return {
            "stream_id": stream_id,
            "status": "stopped",
            "frames_processed": stream.frames_processed,
        }

    def get_stream_status(self, stream_id: str) -> StreamStatus:
        """Get status of a specific stream."""
        stream = self._active_streams.get(stream_id)
        if not stream:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

        return StreamStatus(
            stream_id=stream.stream_id,
            source=stream.source,
            status=stream.status,
            frames_processed=stream.frames_processed,
            started_at=stream.started_at,
            error=stream.error,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — stop all active streams."""
        logger.info("Shutting down ingestion service...")
        for stream_id in list(self._active_streams.keys()):
            try:
                await self.stop_stream(stream_id)
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")


# FASTAPI ROUTER

def create_ingestion_router(service: IngestionService) -> APIRouter:
    """Create FastAPI router for ingestion endpoints."""

    router = APIRouter(prefix="/ingest", tags=["ingestion"])

    @router.post("/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)):
        """Upload and process a video file."""
        return await service.ingest_file(file)

    @router.post("/stream", response_model=IngestResponse)
    async def ingest_stream(request: IngestStreamRequest):
        """Start ingesting an RTSP/RTMP stream."""
        return await service.ingest_stream(request)

    @router.delete("/stream/{stream_id}")
    async def stop_stream(stream_id: str):
        """Stop an active stream."""
        return await service.stop_stream(stream_id)

    @router.get("/stream/{stream_id}", response_model=StreamStatus)
    async def stream_status(stream_id: str):
        """Get status of an active stream."""
        return service.get_stream_status(stream_id)

    @router.get("/streams")
    async def list_streams():
        """List all active streams."""
        return {
            sid: StreamStatus(
                stream_id=s.stream_id,
                source=s.source,
                status=s.status,
                frames_processed=s.frames_processed,
                started_at=s.started_at,
                error=s.error,
            )
            for sid, s in service.active_streams.items()
        }

    return router
