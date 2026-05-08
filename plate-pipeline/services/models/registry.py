"""
Model Registry — Stores and retrieves loaded model instances.

Singleton-like registry that holds all loaded models for the
lifetime of the process.
"""

from __future__ import annotations

import logging
from typing import Optional

from services.config import PipelineConfig
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
        """Load all models based on config."""
        logger.info("Loading all models...")

        # Always load YOLO detector
        self._detector = ModelFactory.create_detector(self._config)

        # Load OCR if enabled
        if self._config.ocr.enabled:
            self._ocr = ModelFactory.create_ocr(self._config)

        # Load LLM if enabled
        if self._config.llm.enabled:
            self._llm = ModelFactory.create_llm(self._config)

        logger.info(
            f"Models loaded — YOLO: {self._detector.is_loaded()}, "
            f"OCR: {self._ocr.is_loaded() if self._ocr else False}, "
            f"LLM: {self._llm.is_available() if self._llm else False}"
        )

    @property
    def detector(self) -> Optional[DetectionModel]:
        return self._detector

    @property
    def ocr(self) -> Optional[OCRModel]:
        return self._ocr

    @property
    def llm(self) -> Optional[LLMClient]:
        return self._llm
