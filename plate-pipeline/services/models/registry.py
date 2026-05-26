"""
Model Registry — Stores and retrieves loaded model instances.

Singleton-like registry that holds all loaded models for the
lifetime of the process.
"""

from __future__ import annotations

import logging
from typing import Optional

from services.config import PipelineConfig, InferenceMode
from services.models.base import DetectionModel, LLMClient, OCRModel
from services.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Stores and retrieves loaded model instances.

    Singleton-like registry that holds all loaded models for the
    lifetime of the process.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._detector: Optional[DetectionModel] = None
        self._ocr: Optional[OCRModel] = None
        self._llm: Optional[LLMClient] = None

    def load_all(self) -> None:
        """Load only the models needed for the active inference mode."""
        mode = self._config.inference.mode
        logger.info(f"Selective load: active mode is {mode.value}")

        # Map active modes to their required model dependencies
        needed = {
            InferenceMode.YOLO_ONLY: {"detector"},
            InferenceMode.OCR_ONLY: {"ocr"},
            InferenceMode.YOLO_OCR: {"detector", "ocr"},
            InferenceMode.YOLO_OCR_LLM: {"detector", "ocr", "llm"},
            InferenceMode.DIRECT_LLM: {"detector", "llm"},
        }.get(mode, {"detector", "ocr", "llm"})

        # Load YOLO detector if needed
        if "detector" in needed:
            _ = self.detector

        # Load OCR if needed and enabled in config
        if "ocr" in needed and self._config.ocr.enabled:
            _ = self.ocr

        # Load LLM if needed and enabled in config
        if "llm" in needed and self._config.llm.enabled:
            _ = self.llm

        logger.info(
            f"Selective load completed — YOLO loaded: {self.detector_loaded}, "
            f"OCR loaded: {self.ocr_loaded}, "
            f"LLM available: {self.llm_available}"
        )

    @property
    def detector(self) -> Optional[DetectionModel]:
        if self._detector is None:
            logger.info("Lazy-loading YOLO detector...")
            self._detector = ModelFactory.create_detector(self._config)
        return self._detector

    @property
    def ocr(self) -> Optional[OCRModel]:
        if self._ocr is None and self._config.ocr.enabled:
            logger.info("Lazy-loading OCR reader...")
            self._ocr = ModelFactory.create_ocr(self._config)
        return self._ocr

    @property
    def llm(self) -> Optional[LLMClient]:
        if self._llm is None and self._config.llm.enabled:
            logger.info("Lazy-loading LLM client...")
            self._llm = ModelFactory.create_llm(self._config)
        return self._llm

    @property
    def detector_loaded(self) -> bool:
        """Check if detector is loaded without triggering lazy-loading."""
        return self._detector is not None and self._detector.is_loaded()

    @property
    def ocr_loaded(self) -> bool:
        """Check if OCR is loaded without triggering lazy-loading."""
        return self._ocr is not None and self._ocr.is_loaded()

    @property
    def llm_available(self) -> bool:
        """Check if LLM is available without triggering lazy-loading."""
        return self._llm is not None and self._llm.is_available()
