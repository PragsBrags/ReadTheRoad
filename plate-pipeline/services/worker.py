"""
Celery Worker — Stateless inference execution.

Workers consume frame payloads from RabbitMQ, route them through
the inference pipeline, and store results in Redis.

All state lives in Redis. Workers are fully stateless and horizontally
scalable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from celery import Celery
from celery.signals import worker_init, worker_shutdown

from services.config import load_config

logger = logging.getLogger(__name__)


# CELERY APP CONFIGURATION

def create_celery_app() -> Celery:
    """
    Create and configure the Celery application.

    All settings are loaded from config.yaml — no hardcoded values.
    """
    config = load_config()
    celery_cfg = config.workers.celery

    app = Celery(
        "plate_pipeline",
        broker=celery_cfg.broker_url,
        backend=celery_cfg.result_backend,
    )

    app.conf.update(
        task_serializer=celery_cfg.task_serializer,
        result_serializer=celery_cfg.result_serializer,
        accept_content=celery_cfg.accept_content,
        task_acks_late=celery_cfg.task_acks_late,
        worker_prefetch_multiplier=celery_cfg.worker_prefetch_multiplier,
        worker_concurrency=celery_cfg.concurrency,
        worker_autoscale=(celery_cfg.autoscale.max, celery_cfg.autoscale.min),
        task_track_started=True,
        task_time_limit=300,
        task_soft_time_limit=240, 
    )

    return app


# Create the Celery app instance
celery_app = create_celery_app()


# LAZY-LOADED SINGLETONS (per worker process)

_inference_router = None
_cache_service = None
_metrics = None


def _get_inference_router():
    """Lazy-load inference router (once per worker process)."""
    global _inference_router
    if _inference_router is None:
        from services.inference import InferenceRouter
        from services.models import ModelRegistry
        config = load_config()
        registry = ModelRegistry(config)
        registry.load_all()
        _inference_router = InferenceRouter(config, registry)
        logger.info("Inference router initialized in worker")
    return _inference_router


def _get_cache_service():
    """Lazy-load cache service."""
    global _cache_service
    if _cache_service is None:
        from services.cache import CacheService
        config = load_config()
        _cache_service = CacheService(config.redis)
        logger.info("Cache service initialized in worker")
    return _cache_service


def _get_metrics():
    """Lazy-load metrics collector."""
    global _metrics
    if _metrics is None:
        from services.monitoring import MetricsCollector
        _metrics = MetricsCollector()
    return _metrics


# CELERY SIGNALS

@worker_init.connect
def on_worker_init(**kwargs):
    """Initialize resources when worker starts."""
    logger.info("Worker initializing — loading models and connections...")
    try:
        _get_inference_router()
        _get_cache_service()
        logger.info("Worker initialization complete")
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Cleanup when worker stops."""
    logger.info("Worker shutting down...")
    global _inference_router, _cache_service
    _inference_router = None
    _cache_service = None


# CELERY TASKS

@celery_app.task(
    name="plate_pipeline.process_frame",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def process_frame(self, frame_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single frame through the inference pipeline.

    This is the main Celery task. It:
    1. Checks deduplication cache
    2. Routes frame through the configured inference mode
    3. Stores result in cache
    4. Records metrics

    Args:
        frame_payload: Serialized frame data from the queue.

    Returns:
        Inference result dictionary.
    """
    frame_id = frame_payload.get("frame_id", "unknown")
    job_id = frame_payload.get("job_id", "")
    start = time.time()

    try:
        cache = _get_cache_service()
        router = _get_inference_router()
        metrics = _get_metrics()

        # --- Deduplication check ---
        content_hash = frame_payload.get("content_hash", "")
        if content_hash and cache:
            cached = cache.get_result(content_hash)
            if cached:
                logger.debug(f"[{frame_id}] Cache hit for hash {content_hash[:8]}")
                metrics.increment_frames_processed(1)
                return cached

        # --- Deserialize frame ---
        from services.queue import deserialize_frame
        frame = deserialize_frame(frame_payload)

        # --- Run inference ---
        result = router.process(frame)

        # --- Enrich result ---
        result["frame_id"] = frame_id
        result["job_id"] = job_id
        result["processed_at"] = time.time()
        result["processing_time_ms"] = (time.time() - start) * 1000

        # --- Cache result ---
        if content_hash and cache:
            cache.store_result(content_hash, result)

        # --- Store by job_id atomically (so /results/{job_id} works) ---
        if job_id and cache:
            plates = result.get("plates", [])
            if plates:
                cache.append_job_plates(job_id, plates)
            cache.increment_job_frames_processed(job_id)

        # --- Metrics ---
        if metrics:
            metrics.record_inference_latency(
                result.get("processing_time_ms", 0) / 1000,
                mode=result.get("inference_mode", "unknown"),
            )
            metrics.increment_frames_processed(1)

        logger.info(
            f"[{frame_id}] Processed in {result['processing_time_ms']:.1f}ms "
            f"mode={result.get('inference_mode', 'unknown')} "
            f"plates={len(result.get('plates', []))}"
        )

        return result

    except Exception as e:
        logger.error(f"[{frame_id}] Processing failed: {e}")

        # Retry with exponential backoff
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"[{frame_id}] Max retries exceeded")
            if job_id and cache:
                cache.increment_job_frames_processed(job_id)
            return {
                "frame_id": frame_id,
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "processing_time_ms": (time.time() - start) * 1000,
            }


@celery_app.task(name="plate_pipeline.health_check")
def health_check() -> dict[str, Any]:
    """Worker health check task."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "inference_router": _inference_router is not None,
        "cache_service": _cache_service is not None,
    }
