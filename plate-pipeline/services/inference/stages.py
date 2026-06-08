"""
Pipeline Stages — Composable inference stages for the detection pipeline.

Each stage implements the Stage interface and processes a data dictionary,
reading what it needs and adding its outputs.
"""

from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from PIL import Image
from pathlib import Path

from services.config import PipelineConfig, DebugConfig
from services.inference.circuit_breaker import CircuitBreaker
from services.models import ModelRegistry
from services.models.preprocessing import crop_resize_plate, preprocess_for_ocr

logger = logging.getLogger(__name__)


# ABSTRACT STAGE

class Stage(ABC):
    """
    Abstract pipeline stage.

    Every inference step implements this interface.
    Stages are composable and form the inference pipeline.
    """

    @abstractmethod
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process data through this stage.

        Args:
            data: Pipeline data dictionary. Each stage reads what it
                  needs and adds its outputs.

        Returns:
            Enriched data dictionary.
        """
        ...


# CONCRETE STAGES

class DetectionStage(Stage):
    """YOLO license plate detection stage."""

    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[PipelineConfig] = None,
        debug_config: Optional[DebugConfig] = None,
    ):
        self._registry = registry
        from services.config import load_config
        self._config = config if config is not None else load_config()
        self._config_debug = debug_config if debug_config is not None else self._config.debug

    def _expand_bbox(self, image, bbox: list[int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        pad = self._config.preprocessing.plate_crop_padding_ratio
        dx = int((x2 - x1) * pad)
        dy = int((y2 - y1) * pad)
        return [
            max(0, x1 - dx),
            max(0, y1 - dy),
            min(w, x2 + dx),
            min(h, y2 + dy),
        ]

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        image = data.get("image")
        if image is None:
            data["detections"] = []
            return data

        start = time.time()
        detector = self._registry.detector
        detections = detector.detect(image) if detector else []
        data["detections"] = detections
        data["detection_time_ms"] = (time.time() - start) * 1000

        # Crop detected plates
        crops = []
        for det in detections:
            x1, y1, x2, y2 = self._expand_bbox(image, det["bbox"])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                if self._config.preprocessing.enabled:
                    crop = crop_resize_plate(
                        crop,
                        min_width = self._config.preprocessing.min_plate_width,
                    )
                crops.append({
                    "image": crop,
                    "bbox": det["bbox"],
                    "crop_bbox": [x1, y1, x2, y2],
                    "confidence": det["confidence"],
                })
        data["plate_crops"] = crops

        logger.info(
            f"DetectionStage: Found {len(detections)} plates in {data['detection_time_ms']:.1f}ms "
            f"(Confidences: {[det['confidence'] for det in detections]})"
        )
        return data


class OCRStage(Stage):
    """OCR text extraction stage."""

    def __init__(
        self,
        registry: ModelRegistry,
        config: Optional[PipelineConfig] = None,
        debug_config: Optional[DebugConfig] = None,
    ):
        # Keep a single source of config; debug settings live on config.debug
        self._registry = registry
        from services.config import load_config
        self._config = config if config is not None else load_config()
        self._config_debug = debug_config if debug_config is not None else self._config.debug

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        crops = data.get("plate_crops", [])
        ocr = self._registry.ocr
        if not crops or not ocr:
            data["ocr_results"] = []
            return data

        start = time.time()
        ocr_results = []

        for crop_info in crops:
            crop_img = crop_info['image']
            # Optionally preprocess the crop for better OCR results
            if self._config.preprocessing.enabled and self._config.preprocessing.enhance_image_ocr:
                debug_dir = Path(self._config_debug.output_dir) if self._config_debug.output_dir else None
                crop_img = preprocess_for_ocr(
                    crop_img,
                    save_debug=self._config.debug.save_intermediate_images,
                    debug_dir=debug_dir,
                    debug_prefix=data.get('frame_id', 'unknown'),
                )

            # Pass the (possibly preprocessed) image to the OCR reader
            texts = ocr.read_text(crop_img)
            combined_text = " ".join(t["text"] for t in texts).strip()
            max_conf = max((t["confidence"] for t in texts), default=0.0)

            ocr_results.append({
                "text": combined_text,
                "confidence": max_conf,
                "bbox": crop_info["bbox"],
                "detection_confidence": crop_info["confidence"],
                "raw_ocr": texts,
            })

        data["ocr_results"] = ocr_results
        data["ocr_time_ms"] = (time.time() - start) * 1000

        logger.info(
            f"OCRStage: Extracted {len(ocr_results)} plates in {data['ocr_time_ms']:.1f}ms "
            f"(Texts: {[res['text'] for res in ocr_results]})"
        )
        return data


class LLMCorrectionStage(Stage):
    """LLM-based text correction stage (gated)."""

    def __init__(
        self,
        registry: ModelRegistry,
        circuit_breaker: CircuitBreaker,
        config: Optional[PipelineConfig] = None,
    ):
        self._registry = registry
        self._breaker = circuit_breaker
        from services.config import load_config
        cfg = config if config is not None else load_config()
        self._config = cfg.llm
        # Track LLM calls per job_id to enforce limits independently
        self._calls_per_job: dict[str, int] = {}

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        ocr_results = data.get("ocr_results", [])
        llm = self._registry.llm
        if not ocr_results or not llm:
            data["llm_corrections"] = []
            return data

        # Check circuit breaker
        if not self._breaker.allow_request():
            logger.warning("Circuit breaker OPEN — skipping LLM correction")
            data["llm_corrections"] = []
            data["llm_skipped"] = True
            return data

        # Check per-job call limit
        job_id = data.get("frame_id", "__default__")
        job_calls = self._calls_per_job.get(job_id, 0)
        if job_calls >= self._config.max_calls_per_job:
            data["llm_corrections"] = []
            data["llm_call_limit_reached"] = True
            return data

        start = time.time()
        corrections = []

        for ocr_result in ocr_results:
            # Encode crop image for LLM
            crop_info = None
            for crop in data.get("plate_crops", []):
                if crop["bbox"] == ocr_result["bbox"]:
                    crop_info = crop
                    break

            image_bytes = None
            if crop_info is not None:
                pil_img = Image.fromarray(crop_info["image"])
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                image_bytes = buf.getvalue()

            try:
                result = llm.correct_plate_text(
                    ocr_text=ocr_result["text"],
                    image_bytes=image_bytes,
                )
                self._breaker.record_success()
                self._calls_per_job[job_id] = job_calls + 1

                corrections.append({
                    "original_text": ocr_result["text"],
                    "corrected_text": result["corrected_text"],
                    "confidence": result["confidence"],
                    "bbox": ocr_result["bbox"],
                    "llm_used": True,
                })

            except Exception as e:
                self._breaker.record_failure()
                logger.error(f"LLM correction failed: {e}")

                # Fallback to OCR result
                if self._config.fallback_to_ocr:
                    corrections.append({
                        "original_text": ocr_result["text"],
                        "corrected_text": ocr_result["text"],
                        "confidence": ocr_result["confidence"],
                        "bbox": ocr_result["bbox"],
                        "llm_used": False,
                        "fallback": True,
                    })

        data["llm_corrections"] = corrections
        data["llm_time_ms"] = (time.time() - start) * 1000

        logger.info(
            f"LLMCorrectionStage: Processed {len(corrections)} corrections in {data['llm_time_ms']:.1f}ms "
            f"(Original -> Corrected: {[(c['original_text'], c['corrected_text']) for c in corrections]})"
        )
        return data


class DirectLLMStage(Stage):
    """Direct LLM vision inference — reads plate text from image directly."""

    def __init__(
        self,
        registry: ModelRegistry,
        circuit_breaker: CircuitBreaker,
    ):
        self._registry = registry
        self._breaker = circuit_breaker

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        crops = data.get("plate_crops", [])
        llm = self._registry.llm
        if not crops or not llm:
            data["direct_llm_results"] = []
            return data

        if not self._breaker.allow_request():
            logger.warning("Circuit breaker OPEN — skipping direct LLM")
            data["direct_llm_results"] = []
            return data

        start = time.time()
        results = []

        for crop_info in crops:
            pil_img = Image.fromarray(crop_info["image"])
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

            try:
                result = llm.correct_plate_text(
                    ocr_text="",
                    image_bytes=image_bytes,
                )
                self._breaker.record_success()

                results.append({
                    "text": result["corrected_text"],
                    "confidence": result["confidence"],
                    "bbox": crop_info["bbox"],
                    "mode": "direct_llm",
                })

            except Exception as e:
                self._breaker.record_failure()
                logger.error(f"Direct LLM failed: {e}")

        data["direct_llm_results"] = results
        data["direct_llm_time_ms"] = (time.time() - start) * 1000
        logger.info(
            f"DirectLLMStage: Extracted {len(results)} plates directly in {data['direct_llm_time_ms']:.1f}ms "
            f"(Texts: {[res['text'] for res in results]})"
        )
        return data
