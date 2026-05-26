"""
Local Processor — Direct inference processing without queue/workers.

When RabbitMQ and Celery workers are not available (local development),
this module processes frames directly in the API process using the
same inference pipeline that workers use.

This is the FALLBACK path. In production with Docker, frames go through:
    Queue → Celery Workers → InferenceRouter

In local dev without Docker:
    IngestionService → LocalProcessor → InferenceRouter (direct)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from services.config import PipelineConfig
from services.decoder import FrameData
from services.database import ResultPersistenceService
from services.database.schema import FrameResultCreate, PlateDetectionCreate
from services.config import load_config

logger = logging.getLogger(__name__)

config = load_config()
persistence = ResultPersistenceService(config.database)

class LocalProcessor:
    """
    Processes frames directly through the inference pipeline
    without requiring RabbitMQ or Celery workers.

    Lazy-loads the ModelRegistry and InferenceRouter on first use
    to avoid loading heavy models (YOLO, OCR) unless actually needed.

    Supports optional cache (for dedup + result storage) and metrics
    to maintain feature parity with the distributed worker path.
    """

    def __init__(
        self,
        config: PipelineConfig,
        cache: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ):
        self._config = config
        self._cache = cache
        self._metrics = metrics
        self._inference_router = None
        self._model_registry = None
        self._initialized = False
        self._init_error: Optional[str] = None

    def _ensure_initialized(self) -> bool:
        """Lazy-initialize models and inference router."""
        if self._initialized:
            return True

        if self._init_error:
            # Don't retry if initialization already failed
            return False

        logger.info("LocalProcessor: Loading models for direct inference...")
        start = time.time()

        try:
            from services.models import ModelRegistry
            from services.inference import InferenceRouter

            self._model_registry = ModelRegistry(self._config)
            self._model_registry.load_all()

            self._inference_router = InferenceRouter(
                self._config, self._model_registry
            )

            elapsed = (time.time() - start) * 1000
            self._initialized = True

            logger.info(
                f"LocalProcessor: Models loaded in {elapsed:.0f}ms — "
                f"YOLO={self._model_registry.detector_loaded if self._model_registry else False}, "
                f"OCR={self._model_registry.ocr_loaded if self._model_registry else False}, "
                f"LLM={self._model_registry.llm_available if self._model_registry else False}"
            )
            return True

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"LocalProcessor: Failed to initialize — {e}")
            return False

    def process_frame(self, frame: FrameData, job_id: str = "") -> dict[str, Any]:
        """
        Process a single frame through the inference pipeline directly.

        This mirrors what the Celery worker's process_frame task does,
        but runs synchronously in the API process. Includes dedup check,
        result caching, job result storage, and metrics — matching the
        distributed worker path for feature parity.

        Args:
            frame: FrameData to process.
            job_id: Associated job ID.

        Returns:
            Inference result dictionary.
        """
        frame_id = frame.frame_id
        start = time.time()

        if not self._ensure_initialized():
            return {
                "frame_id": frame_id,
                "job_id": job_id,
                "status": "error",
                "error": f"LocalProcessor not initialized: {self._init_error}",
                "plates": [],
                "processing_time_ms": 0,
            }

        try:
            # --- Deduplication check (mirrors worker.py) ---
            content_hash = frame.content_hash or ""
            if content_hash and self._cache:
                cached = self._cache.get_result(content_hash)
                if cached:
                    logger.debug(f"[{frame_id}] Local cache hit for hash {content_hash[:8]}")
                    if self._metrics:
                        self._metrics.increment_cache_hit()
                        self._metrics.increment_frames_processed(1)
                    return cached
                self._cache.mark_processed(content_hash)

            # --- Run inference ---
            result = self._inference_router.process(frame)

            # Enrich result
            result["frame_id"] = frame_id
            result["job_id"] = job_id
            result["processed_at"] = time.time()
            result["processing_time_ms"] = (time.time() - start) * 1000

            plates = [
            PlateDetectionCreate(
                job_id=job_id,
                frame_id=frame_id,
                plate_text=plate.get("text"),
                vehicle_class=plate.get("vehicle_class"),
                confidence=plate.get("confidence", 0.0),
                raw_plate=plate,
            )
            for plate in result.get("plates", [])
            ]

            persistence.save_frame_result(
                job=FrameResultCreate(
                    job_id=job_id,
                    frame_id=frame_id,
                    source=frame.source,
                    timestamp_ms=frame.timestamp_ms,
                    inference_mode=result.get("inference_mode"),
                    plate_count=len(plates),
                    processing_time_ms=result.get("processing_time_ms", 0.0),
                    detection_time_ms=result.get("timings", {}).get("detection_ms", 0.0),
                    ocr_time_ms=result.get("timings", {}).get("ocr_ms", 0.0),
                    llm_time_ms=result.get("timings", {}).get("llm_ms", 0.0),
                ),
                plates=plates,
            )


            # --- Cache result (mirrors worker.py) ---
            if content_hash and self._cache:
                self._cache.store_result(content_hash, result)

            # --- Store by job_id so /results/{job_id} works (mirrors worker.py) ---
            if job_id and self._cache:
                plates = result.get("plates", [])
                if plates:
                    self._cache.append_job_plates(job_id, plates)
                self._cache.increment_job_frames_processed(job_id)

            # --- Metrics (mirrors worker.py) ---
            if self._metrics:
                self._metrics.record_inference_latency(
                    result.get("processing_time_ms", 0) / 1000,
                    mode=result.get("inference_mode", "unknown"),
                )
                self._metrics.increment_frames_processed(1)
                self._metrics.increment_cache_miss()

            logger.info(
                f"[{frame_id}] Local inference: {result['processing_time_ms']:.1f}ms, "
                f"mode={result.get('inference_mode', 'unknown')}, "
                f"plates={len(result.get('plates', []))}"
            )

            return result

        except Exception as e:
            logger.error(f"[{frame_id}] Local inference failed: {e}")
            return {
                "frame_id": frame_id,
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "plates": [],
                "processing_time_ms": (time.time() - start) * 1000,
            }

    def process_frames(
        self,
        frames: list[FrameData],
        job_id: str = "",
    ) -> list[dict[str, Any]]:
        """
        Process multiple frames and return all results.

        Args:
            frames: List of frames to process.
            job_id: Associated job ID.

        Returns:
            List of inference result dicts.
        """
        results = []
        for frame in frames:
            result = self.process_frame(frame, job_id=job_id)
            results.append(result)
        return results

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def model_status(self) -> dict[str, Any]:
        """Return status of loaded models."""
        if not self._initialized:
            return {"initialized": False, "error": self._init_error}

        return {
            "initialized": True,
            "yolo_loaded": (
                self._model_registry.detector_loaded
                if self._model_registry
                else False
            ),
            "ocr_loaded": (
                self._model_registry.ocr_loaded
                if self._model_registry
                else False
            ),
            "llm_available": (
                self._model_registry.llm_available
                if self._model_registry
                else False
            ),
        }
