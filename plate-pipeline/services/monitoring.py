"""
Monitoring Service — Prometheus metrics for observability.

Tracks:
  - ingestion_latency_seconds (Histogram)
  - fps_sampling_rate (Gauge)
  - queue_depth (Gauge)
  - inference_latency_seconds (Histogram, labeled by mode)
  - ocr_latency_seconds (Histogram)
  - llm_calls_total (Counter)
  - llm_errors_total (Counter)
  - frames_processed_total (Counter)
  - active_streams (Gauge)

Exposes /metrics endpoint for Prometheus scraping.
"""

from __future__ import annotations

import logging
from typing import Optional

from services.config import MonitoringConfig

logger = logging.getLogger(__name__)


# METRICS COLLECTOR

class MetricsCollector:
    """
    Prometheus metrics collector.

    All metrics are registered once and shared across the process.
    Thread-safe via prometheus_client internals.
    """

    _instance: Optional["MetricsCollector"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[MonitoringConfig] = None):
        """Singleton pattern for metrics collector."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[MonitoringConfig] = None):
        if self._initialized:
            return

        self._config = config
        self._enabled = False

        try:
            from prometheus_client import (
                Counter,
                Gauge,
                Histogram,
                Info,
            )

            # --- System Info ---
            self.system_info = Info(
                "plate_pipeline",
                "License plate detection pipeline info",
            )

            # --- Ingestion Metrics ---
            self.ingestion_latency = Histogram(
                "plate_ingestion_latency_seconds",
                "Time to ingest and process a video source",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            )

            self.active_streams_gauge = Gauge(
                "plate_active_streams",
                "Number of currently active video streams",
            )

            # --- Frame Metrics ---
            self.frames_processed = Counter(
                "plate_frames_processed_total",
                "Total number of frames processed",
            )

            self.frames_sampled = Counter(
                "plate_frames_sampled_total",
                "Total frames that passed sampling filter",
            )

            self.fps_sampling_rate = Gauge(
                "plate_fps_sampling_rate",
                "Current effective sampling FPS",
            )

            # --- Queue Metrics ---
            self.queue_depth = Gauge(
                "plate_queue_depth",
                "Current depth of the frame processing queue",
            )

            self.queue_publish_total = Counter(
                "plate_queue_publish_total",
                "Total messages published to queue",
            )

            self.queue_publish_errors = Counter(
                "plate_queue_publish_errors_total",
                "Total queue publish errors",
            )

            # --- Inference Metrics ---
            self.inference_latency = Histogram(
                "plate_inference_latency_seconds",
                "Time for inference processing per frame",
                labelnames=["mode"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            )

            self.detections_total = Counter(
                "plate_detections_total",
                "Total license plate detections",
            )

            # --- OCR Metrics ---
            self.ocr_latency = Histogram(
                "plate_ocr_latency_seconds",
                "Time for OCR text extraction",
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            )

            self.ocr_extractions_total = Counter(
                "plate_ocr_extractions_total",
                "Total OCR text extractions performed",
            )

            # --- LLM Metrics ---
            self.llm_calls = Counter(
                "plate_llm_calls_total",
                "Total LLM API calls made",
            )

            self.llm_errors = Counter(
                "plate_llm_errors_total",
                "Total LLM API call errors",
            )

            self.llm_latency = Histogram(
                "plate_llm_latency_seconds",
                "Time for LLM inference/correction",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            )

            self.llm_tokens_used = Counter(
                "plate_llm_tokens_used_total",
                "Total LLM tokens consumed",
            )

            self.circuit_breaker_state = Gauge(
                "plate_circuit_breaker_state",
                "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            )

            # --- Cache Metrics ---
            self.cache_hits = Counter(
                "plate_cache_hits_total",
                "Total cache hits",
            )

            self.cache_misses = Counter(
                "plate_cache_misses_total",
                "Total cache misses",
            )

            self.dedup_skipped = Counter(
                "plate_dedup_skipped_total",
                "Total frames skipped due to deduplication",
            )

            # --- Aggregation Metrics ---
            self.aggregation_windows = Gauge(
                "plate_aggregation_active_windows",
                "Number of active aggregation windows",
            )

            self.aggregated_plates = Counter(
                "plate_aggregated_plates_total",
                "Total aggregated plate results produced",
            )

            self._enabled = True
            MetricsCollector._initialized = True
            logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning(
                "prometheus_client not installed. Metrics disabled. "
                "Install with: pip install prometheus-client"
            )
            MetricsCollector._initialized = True

    # RECORDING METHODS

    def record_ingestion_latency(self, seconds: float) -> None:
        """Record video ingestion latency."""
        if self._enabled:
            self.ingestion_latency.observe(seconds)

    def set_active_streams(self, count: int) -> None:
        """Set the number of active streams."""
        if self._enabled:
            self.active_streams_gauge.set(count)

    def increment_frames_processed(self, count: int = 1) -> None:
        """Increment frames processed counter."""
        if self._enabled:
            self.frames_processed.inc(count)

    def increment_frames_sampled(self, count: int = 1) -> None:
        """Increment frames sampled counter."""
        if self._enabled:
            self.frames_sampled.inc(count)

    def set_fps_rate(self, fps: float) -> None:
        """Set current effective sampling FPS."""
        if self._enabled:
            self.fps_sampling_rate.set(fps)

    def set_queue_depth(self, depth: int) -> None:
        """Set current queue depth."""
        if self._enabled:
            self.queue_depth.set(depth)

    def record_inference_latency(self, seconds: float, mode: str = "unknown") -> None:
        """Record inference latency with mode label."""
        if self._enabled:
            self.inference_latency.labels(mode=mode).observe(seconds)

    def increment_detections(self, count: int = 1) -> None:
        """Increment plate detections counter."""
        if self._enabled:
            self.detections_total.inc(count)

    def record_ocr_latency(self, seconds: float) -> None:
        """Record OCR extraction latency."""
        if self._enabled:
            self.ocr_latency.observe(seconds)

    def increment_llm_calls(self) -> None:
        """Increment LLM call counter."""
        if self._enabled:
            self.llm_calls.inc()

    def increment_llm_errors(self) -> None:
        """Increment LLM error counter."""
        if self._enabled:
            self.llm_errors.inc()

    def record_llm_latency(self, seconds: float) -> None:
        """Record LLM call latency."""
        if self._enabled:
            self.llm_latency.observe(seconds)

    def add_llm_tokens(self, count: int) -> None:
        """Add to LLM token usage counter."""
        if self._enabled:
            self.llm_tokens_used.inc(count)

    def set_circuit_breaker_state(self, state: str) -> None:
        """Set circuit breaker state gauge."""
        if self._enabled:
            state_map = {"closed": 0, "half_open": 1, "open": 2}
            self.circuit_breaker_state.set(state_map.get(state, -1))

    def increment_cache_hit(self) -> None:
        if self._enabled:
            self.cache_hits.inc()

    def increment_cache_miss(self) -> None:
        if self._enabled:
            self.cache_misses.inc()

    def increment_dedup_skipped(self) -> None:
        if self._enabled:
            self.dedup_skipped.inc()

    def set_aggregation_windows(self, count: int) -> None:
        if self._enabled:
            self.aggregation_windows.set(count)

    def increment_aggregated_plates(self, count: int = 1) -> None:
        if self._enabled:
            self.aggregated_plates.inc(count)

    @property
    def enabled(self) -> bool:
        return self._enabled
