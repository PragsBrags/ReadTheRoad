"""
OCR Engines — EasyOCR and PaddleOCR implementations.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from services.config import PipelineConfig
from services.models.base import OCRModel

logger = logging.getLogger(__name__)


class EasyOCRReader(OCRModel):
    """EasyOCR-based text extraction."""

    def __init__(self, config: PipelineConfig):
        self._config = config.ocr.easyocr
        self._reader = None
        self._loaded = False

    def load(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self._config.languages,
                gpu=self._config.gpu,
            )
            self._loaded = True
            logger.info(
                f"EasyOCR loaded (languages={self._config.languages}, "
                f"gpu={self._config.gpu})"
            )
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            self._loaded = False

    def read_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Extract text from a plate image using EasyOCR."""
        if not self._loaded or self._reader is None:
            return []

        results = self._reader.readtext(
            image,
            detail=self._config.detail,
        )

        texts = []
        for detection in results:
            if self._config.detail:
                bbox, text, conf = detection
                texts.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": bbox,
                })
            else:
                texts.append({
                    "text": detection,
                    "confidence": 1.0,
                })

        return texts

    def is_loaded(self) -> bool:
        return self._loaded


class PaddleOCRReader(OCRModel):
    """PaddleOCR-based text extraction."""

    def __init__(self, config: PipelineConfig):
        self._config = config.ocr.paddleocr
        self._reader = None
        self._loaded = False

    def load(self) -> None:
        """Initialize PaddleOCR reader."""
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                lang=self._config.lang,
                use_angle_cls=self._config.use_angle_cls,
                use_gpu=self._config.use_gpu,
                show_log=False,
            )
            self._loaded = True
            logger.info(
                f"PaddleOCR loaded (lang={self._config.lang}, "
                f"gpu={self._config.use_gpu})"
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            self._loaded = False

    def read_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Extract text from a plate image using PaddleOCR."""
        if not self._loaded or self._reader is None:
            return []

        result = self._reader.ocr(image, cls=self._config.use_angle_cls)

        texts = []
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                conf = line[1][1]
                texts.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": bbox,
                })

        return texts

    def is_loaded(self) -> bool:
        return self._loaded
