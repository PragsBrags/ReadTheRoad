"""
API Routes — FastAPI endpoint handlers for the plate detection pipeline.

Provides:
  - / — API root info
  - /status — System health check
  - /results/{job_id} — Query detection results
  - /metrics — Prometheus metrics endpoint
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse


def create_api_router(services: dict[str, Any]) -> APIRouter:
    """
    Create the core API router with status, results, and metrics endpoints.

    Args:
        services: Dictionary of initialized service instances.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter()

    @router.get("/")
    async def root():
        """API root — basic info."""
        return {
            "service": "License Plate Detection Pipeline",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/status",
            "metrics": "/metrics",
        }

    @router.get("/status")
    async def system_status():
        """System health check."""
        cache = services.get("cache")
        ingestion = services.get("ingestion")
        dispatcher = services.get("dispatcher")

        from services.config import load_config
        config = load_config()

        return {
            "status": "healthy",
            "inference_mode": config.inference.mode.value,
            "ocr_engine": config.ocr.engine.value,
            "processing_mode": dispatcher.mode if dispatcher else "unknown",
            "components": {
                "ffmpeg_decoder": "ready",
                "frame_sampler": f"strategy={config.frame_sampler.strategy.value}",
                "redis": "connected" if (cache and cache.is_connected) else "disconnected",
                "dispatcher": dispatcher.status if dispatcher else "not initialized",
                "active_streams": len(ingestion.active_streams) if ingestion else 0,
            },
            "config": {
                "mode": config.system.mode.value,
                "yolo_model": config.model_registry.yolo.path,
                "max_streams": config.video_ingestion.max_streams,
                "llm_enabled": config.llm.enabled,
            },
        }

    @router.get("/results/{job_id}")
    async def get_results(job_id: str):
        """Query detection results for a job."""
        cache = services.get("cache")
        aggregation = services.get("aggregation")
        ingestion = services.get("ingestion")

        results = None
        status = "unknown"
        frames_sampled = 0
        frames_processed = 0

        # Try to get progress from cache
        if cache and cache.is_connected:
            progress = cache.get_job_progress(job_id)
            frames_sampled = progress.get("total_frames", 0)
            frames_processed = progress.get("processed_frames", 0)

            # Check if this job has results
            results = cache.get_job_results(job_id)

        # Determine status
        if ingestion and job_id in ingestion.active_streams:
            status = "streaming"
        elif frames_sampled > 0:
            if frames_processed >= frames_sampled:
                status = "completed"
            else:
                status = "processing"
        elif results is not None:
            # Fallback for when results exist but progress was not explicitly set (e.g. local mode cached results)
            status = "completed"
            frames_processed = len(results)
            frames_sampled = len(results)

        # If job is completed, finalize aggregation and cache
        if status == "completed":
            if results is None:
                results = []
            if len(results) > 0 and "frame_count" not in results[0] and aggregation:
                aggregation.flush(job_id)  # Clear state
                aggregation.add_result(job_id, {"plates": results})
                results = aggregation.flush(job_id)
                if cache and cache.is_connected:
                    cache.store_job_results(job_id, results)
            elif len(results) == 0 and cache and cache.is_connected:
                # Cache empty list so future calls don't hit fallback list lookups
                cache.store_job_results(job_id, [])

        # If it is still processing/streaming, aggregate on-the-fly for preview
        elif status in ("processing", "streaming"):
            if results is None:
                results = []
            if len(results) > 0 and "frame_count" not in results[0] and aggregation:
                aggregation.flush(job_id)  # Clear state
                aggregation.add_result(job_id, {"plates": results})
                results = aggregation.flush(job_id)

        # Fallback to local aggregation or raising 404
        if results is None and status == "unknown":
            # If aggregation service has it in active windows
            if aggregation and job_id in aggregation._windows:
                results = aggregation.flush(job_id)
            if results is None:
                raise HTTPException(status_code=404, detail=f"No results for job {job_id}")

        if results is None:
            results = []

        return {
            "job_id": job_id,
            "status": status,
            "frames_sampled": frames_sampled,
            "frames_processed": frames_processed,
            "plates": results,
            "count": len(results),
        }


    @router.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from fastapi.responses import Response
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            return JSONResponse(
                content={"error": "prometheus_client not installed"},
                status_code=501,
            )

    return router
