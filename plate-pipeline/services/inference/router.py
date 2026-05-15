"""
Inference Router — Config-driven inference mode routing.

Implements the Router Pattern to dynamically select the correct
inference pipeline based on config.yaml.

Supported modes:
  1. yolo_only — Detect plates, return bounding boxes
  2. ocr_only — OCR on pre-cropped images
  3. yolo_ocr — Detect → crop → OCR (default)
  4. yolo_ocr_llm — YOLO+OCR + LLM correction (hybrid)
  5. direct_llm — LLM vision inference on cropped plates

Each mode is composed of Pipeline Stages (Pipeline Pattern).
LLM calls include a Circuit Breaker for fault tolerance.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from services.config import InferenceMode, PipelineConfig
from services.decoder import FrameData
from services.inference.circuit_breaker import CircuitBreaker
from services.inference.stages import (
    DetectionStage,
    DirectLLMStage,
    LLMCorrectionStage,
    OCRStage,
    Stage,
)
from services.models import ModelRegistry

logger = logging.getLogger(__name__)


class InferenceRouter:
    """
    Router Pattern — routes frames to the correct inference pipeline.

    Reads config.inference.mode and constructs the appropriate
    pipeline of stages. All routing is config-driven.
    """

    def __init__(self, config: PipelineConfig, registry: ModelRegistry):
        self._config = config
        self._registry = registry
        self._config_debug = config.debug

        # Initialize circuit breaker from config
        cb_config = config.llm.circuit_breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.failure_threshold,
            recovery_timeout=cb_config.recovery_timeout_seconds,
            half_open_max_calls=cb_config.half_open_max_calls,
        )

        # Build pipelines for each mode
        self._pipelines = self._build_pipelines()

        logger.info(
            f"InferenceRouter initialized — active mode: {config.inference.mode.value}"
        )

    def _build_pipelines(self) -> dict[InferenceMode, list[Stage]]:
        """Build pipeline stages for each inference mode."""
        detection = DetectionStage(self._registry, self._config)
        ocr = OCRStage(self._registry, self._config, self._config_debug)
        llm_correction = LLMCorrectionStage(
            self._registry, self._circuit_breaker, self._config
        )
        direct_llm = DirectLLMStage(
            self._registry, self._circuit_breaker
        )

        return {
            InferenceMode.YOLO_ONLY: [detection],
            InferenceMode.OCR_ONLY: [ocr],
            InferenceMode.YOLO_OCR: [detection, ocr],
            InferenceMode.YOLO_OCR_LLM: [detection, ocr, llm_correction],
            InferenceMode.DIRECT_LLM: [detection, direct_llm],
        }

    def process(self, frame: FrameData) -> dict[str, Any]:
        """
        Route a frame through the configured inference pipeline.

        Args:
            frame: Input frame data.

        Returns:
            Inference result dictionary.
        """
        mode = self._config.inference.mode
        pipeline = self._pipelines.get(mode)

        if pipeline is None:
            return {
                "status": "error",
                "error": f"Unknown inference mode: {mode}",
                "plates": [],
            }

        # Convert frame bytes to numpy image
        image = np.array(Image.open(io.BytesIO(frame.frame_bytes)))

        # Initialize pipeline data
        data: dict[str, Any] = {
            "frame_id": frame.frame_id,
            "source": frame.source,
            "frame_index": frame.frame_index,
            "image": image,
            "inference_mode": mode.value,
        }

        # Execute pipeline stages sequentially
        start = time.time()
        for stage in pipeline:
            data = stage.process(data)

        total_ms = (time.time() - start) * 1000

        # Build standardized output
        return self._build_result(data, mode, total_ms)

    def _build_result(
        self,
        data: dict[str, Any],
        mode: InferenceMode,
        total_ms: float,
    ) -> dict[str, Any]:
        """Build standardized result from pipeline data."""
        plates = []

        if mode == InferenceMode.YOLO_ONLY:
            for det in data.get("detections", []):
                plates.append({
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "text": None,
                })

        elif mode == InferenceMode.OCR_ONLY:
            for ocr in data.get("ocr_results", []):
                plates.append({
                    "bbox": ocr.get("bbox"),
                    "confidence": ocr["confidence"],
                    "text": ocr["text"],
                })

        elif mode == InferenceMode.YOLO_OCR:
            for ocr in data.get("ocr_results", []):
                plates.append({
                    "bbox": ocr["bbox"],
                    "confidence": (
                        ocr["detection_confidence"] * 0.5 +
                        ocr["confidence"] * 0.5
                    ),
                    "text": ocr["text"],
                })

        elif mode == InferenceMode.YOLO_OCR_LLM:
            corrections = data.get("llm_corrections", [])
            if corrections:
                for cor in corrections:
                    plates.append({
                        "bbox": cor["bbox"],
                        "confidence": cor["confidence"],
                        "text": cor["corrected_text"],
                        "original_ocr": cor["original_text"],
                        "llm_used": cor.get("llm_used", False),
                    })
            else:
                # Fallback to OCR results
                for ocr in data.get("ocr_results", []):
                    plates.append({
                        "bbox": ocr["bbox"],
                        "confidence": ocr["confidence"],
                        "text": ocr["text"],
                    })

        elif mode == InferenceMode.DIRECT_LLM:
            for llm_res in data.get("direct_llm_results", []):
                plates.append({
                    "bbox": llm_res["bbox"],
                    "confidence": llm_res["confidence"],
                    "text": llm_res["text"],
                    "mode": "direct_llm",
                })

        result = {
            "status": "success",
            "inference_mode": mode.value,
            "plates": plates,
            "plate_count": len(plates),
            "processing_time_ms": total_ms,
            "timings": {
                "detection_ms": data.get("detection_time_ms", 0),
                "ocr_ms": data.get("ocr_time_ms", 0),
                "llm_ms": data.get("llm_time_ms", 0) + data.get("direct_llm_time_ms", 0),
            },
            "circuit_breaker_state": self._circuit_breaker.state,
        }

        # Remove numpy image from result (not serializable)
        return result

    def route(self, frame: FrameData, config_override: Optional[dict] = None) -> dict[str, Any]:
        """
        Route with optional config override (for testing).

        In production, always uses config.inference.mode.
        """
        return self.process(frame)
