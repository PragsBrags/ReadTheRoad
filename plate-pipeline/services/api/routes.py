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

        results = None

        # Try cache first
        if cache and cache.is_connected:
            results = cache.get_job_results(job_id)

        # Try aggregation service
        if results is None and aggregation:
            results = aggregation.flush(job_id)

        if results is None:
            raise HTTPException(status_code=404, detail=f"No results for job {job_id}")

        return {
            "job_id": job_id,
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
