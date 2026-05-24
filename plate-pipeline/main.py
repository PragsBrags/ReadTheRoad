"""
Main Application — FastAPI entrypoint for the plate detection pipeline.

Provides:
  - /ingest/* — Video ingestion endpoints
  - /status — System health
  - /results/{job_id} — Query detection results
  - /metrics — Prometheus metrics endpoint
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plate_pipeline")


# SERVICE INITIALIZATION

# Global service references
_services: dict[str, Any] = {}


def _init_services(app: FastAPI) -> None:
    """
    Initialize all pipeline services during startup.

    Loads config, creates service instances, and mounts routers.
    """
    from services.config import load_config
    config = load_config()
    logger.info(f"Config loaded (mode={config.system.mode.value})")

    # Monitoring
    from services.monitoring import MetricsCollector
    metrics = MetricsCollector(config.monitoring)
    _services["metrics"] = metrics

    # FFmpeg decoder
    from services.decoder import FFmpegDecoder
    decoder = FFmpegDecoder(config.decoder, config.video_ingestion)
    _services["decoder"] = decoder
    logger.info("FFmpeg decoder initialized")

    # Frame sampler
    from services.sampler import FrameSampler
    sampler = FrameSampler(config.frame_sampler)
    _services["sampler"] = sampler

    # Redis cache (optional — works without it)
    from services.cache import CacheService
    cache = CacheService(config.redis)
    _services["cache"] = cache

    # Database service
    from services.database import ResultPersistenceService
    persistence = ResultPersistenceService(config.database)
    _services["persistence"] = persistence

    # Aggregation service
    from services.aggregation import AggregationService
    aggregation = AggregationService(config.aggregation)
    _services["aggregation"] = aggregation

    # Task dispatcher (decides local vs distributed)
    from services.task_dispatcher import TaskDispatcher
    dispatcher = TaskDispatcher(config, cache=cache, metrics=metrics)
    dispatcher.configure_distributed()
    dispatcher.configure_local()
    _services["dispatcher"] = dispatcher

    # Ingestion service
    from services.ingestion import IngestionService, create_ingestion_router
    ingestion = IngestionService(
        config=config,
        decoder=decoder,
        sampler=sampler,
        dispatcher=dispatcher,
        metrics=metrics,
        cache=cache,
        aggregation=aggregation,
        persistence=persistence,
    )
    _services["ingestion"] = ingestion

    # Mount ingestion router
    app.include_router(create_ingestion_router(ingestion))

    # Mount core API router (status, results, metrics)
    from services.api import create_api_router
    app.include_router(create_api_router(_services))

    # Set system info metric
    if metrics.enabled:
        metrics.system_info.info({
            "version": "1.0.0",
            "mode": config.system.mode.value,
            "inference_mode": config.inference.mode.value,
            "ocr_engine": config.ocr.engine.value,
            "decoder": config.decoder.engine,
            "processing": dispatcher.mode,
        })

    _log_startup_banner(config, cache, dispatcher)


def _log_startup_banner(config, cache, dispatcher) -> None:
    """Log the startup banner with system status."""
    logger.info("=" * 60)
    logger.info("  PLATE DETECTION PIPELINE — READY")
    logger.info(f"  Inference mode : {config.inference.mode.value}")
    logger.info(f"  OCR engine     : {config.ocr.engine.value}")
    logger.info(f"  YOLO model     : {config.model_registry.yolo.path}")
    logger.info(f"  Redis          : {'connected' if cache.is_connected else 'disconnected'}")
    logger.info(f"  Processing     : {dispatcher.mode.upper()}")
    if dispatcher.mode == "distributed":
        logger.info("    → Frames dispatched via Celery to RabbitMQ workers")
    else:
        logger.info("    → Frames processed inline (no RabbitMQ needed)")
    logger.info("=" * 60)


# APPLICATION LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    logger.info("=" * 60)
    logger.info("  PLATE DETECTION PIPELINE — STARTING")
    logger.info("=" * 60)

    try:
        _init_services(app)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down pipeline...")

    ingestion = _services.get("ingestion")
    if ingestion:
        await ingestion.shutdown()

    cache = _services.get("cache")
    if cache:
        cache.close()

    logger.info("Pipeline shutdown complete")


# FASTAPI APPLICATION

app = FastAPI(
    title="License Plate Detection Pipeline",
    description=(
        "license plate detection system using FFmpeg, "
   
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MAIN

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
