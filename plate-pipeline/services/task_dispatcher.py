"""
Task Dispatcher — Routes frames to either Celery workers or local processor.

This is the single decision point for HOW frames get processed:
  - If RabbitMQ is available → dispatch as Celery tasks (distributed)
  - If not → process directly via LocalProcessor (local)


"""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.config import PipelineConfig
from services.decoder import FrameData
from services.queue import serialize_frame

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """
    Dispatches frame processing to the appropriate backend.

    In distributed mode (Docker): sends Celery tasks via send_task()
    In local mode: processes inline via LocalProcessor
    """

    def __init__(self, config: PipelineConfig, cache=None, metrics=None):
        self._config = config
        self._cache = cache
        self._metrics = metrics
        self._celery_app = None
        self._local_processor = None
        self._mode: str = "none"  # "distributed" | "local" | "none"

    def configure_distributed(self) -> bool:
        """
        Try to connect to RabbitMQ via Celery.
        Returns True if distributed mode is available.
        """
        try:
            from celery import Celery

            celery_cfg = self._config.workers.celery
            self._celery_app = Celery(
                "plate_pipeline",
                broker=celery_cfg.broker_url,
                backend=celery_cfg.result_backend,
            )

            # Test the connection by pinging the broker
            conn = self._celery_app.connection()
            conn.ensure_connection(max_retries=1, timeout=3)
            conn.close()

            self._mode = "distributed"
            logger.info(
                f"TaskDispatcher: DISTRIBUTED mode — "
                f"Celery connected to {celery_cfg.broker_url}"
            )
            return True

        except Exception as e:
            logger.warning(f"TaskDispatcher: Celery/RabbitMQ not available — {e}")
            self._celery_app = None
            return False

    def configure_local(self) -> bool:
        """
        Set up local processing (fallback).
        Returns True if local mode is ready.
        """
        try:
            from services.local_processor import LocalProcessor
            self._local_processor = LocalProcessor(
                self._config,
                cache=self._cache,
                metrics=self._metrics,
            )
            if self._mode != "distributed":
                self._mode = "local"
            logger.info("TaskDispatcher: LOCAL mode available as fallback")
            return True
        except Exception as e:
            logger.error(f"TaskDispatcher: Failed to create local processor — {e}")
            return False

    def dispatch_frame(
        self,
        frame: FrameData,
        job_id: str = "",
    ) -> Optional[dict[str, Any]]:
        """
        Dispatch a single frame for processing.

        In distributed mode: sends a Celery task (async, returns None).
        In local mode: processes inline (sync, returns result).

        Returns:
            Result dict in local mode, None in distributed mode.
        """
        use_redis = self._mode == "distributed" and self._celery_app and self._cache and self._cache.is_connected()
        payload = serialize_frame(frame, job_id, include_bytes=not use_redis)

        if self._mode == "distributed" and self._celery_app:
            try:
                if use_redis:
                    redis_key = f"frame:bytes:{frame.frame_id}"
                    try:
                        self._cache._client.setex(redis_key, 300, frame.frame_bytes)
                        payload["redis_key"] = redis_key
                    except Exception as redis_err:
                        logger.warning(f"Failed to store frame bytes in Redis: {redis_err} — falling back to base64 payload")
                        import base64
                        payload["frame_b64"] = base64.b64encode(frame.frame_bytes).decode("utf-8")

                self._celery_app.send_task(
                    "plate_pipeline.process_frame",
                    args=[payload],
                    queue="celery",
                )
                logger.debug(
                    f"Dispatched frame {frame.frame_id} as Celery task"
                )
                return None  # Async — result comes later via /results/
            except Exception as e:
                logger.error(f"Celery dispatch failed: {e} — falling back to local")
                # Fall through to local processing

        if self._local_processor:
            return self._local_processor.process_frame(frame, job_id)

        logger.error("No processing backend available!")
        return {
            "frame_id": frame.frame_id,
            "job_id": job_id,
            "status": "error",
            "error": "No processing backend (no Celery, no local processor)",
            "plates": [],
        }

    def dispatch_frames(
        self,
        frames: list[FrameData],
        job_id: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Dispatch multiple frames.

        Returns:
            Tuple of (processing_mode, list of results).
            In distributed mode, results list is empty (async processing).
        """
        results = []
        mode = self._mode

        for frame in frames:
            result = self.dispatch_frame(frame, job_id)
            if result is not None:
                results.append(result)
                mode = "local"  # If we got results back, we're local

        return mode, results



    @property
    def mode(self) -> str:
        return self._mode

    @property
    def local_processor(self):
        return self._local_processor

    @property
    def status(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "celery_available": self._celery_app is not None,
            "local_available": self._local_processor is not None,
            "local_models": (
                self._local_processor.model_status
                if self._local_processor
                else None
            ),
        }
