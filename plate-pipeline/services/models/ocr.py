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
            
            # Map use_gpu setting to device parameter
            device = "gpu" if self._config.use_gpu else "cpu"
            
            self._reader = PaddleOCR(
                lang=self._config.lang,
                device=device,
                use_textline_orientation=self._config.use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            self._loaded = True
            logger.info(
                f"PaddleOCR loaded (lang={self._config.lang}, "
                f"device={device})"
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            self._loaded = False

    def read_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Extract text from a plate image using PaddleOCR."""
        if not self._loaded or self._reader is None:
            return []

        try:
            result = self._reader.predict(image)
        except Exception as e:
            logger.error(f"PaddleOCR prediction failed: {e}")
            return []

        texts = []
        if result and len(result) > 0:
            ocr_res = result[0]
            
            # Extract fields handling both object attribute and dictionary formats
            rec_texts = getattr(ocr_res, "rec_texts", None)
            if rec_texts is None and isinstance(ocr_res, dict):
                rec_texts = ocr_res.get("rec_texts", [])
            elif rec_texts is None:
                rec_texts = []

            rec_scores = getattr(ocr_res, "rec_scores", None)
            if rec_scores is None and isinstance(ocr_res, dict):
                rec_scores = ocr_res.get("rec_scores", [])
            elif rec_scores is None:
                rec_scores = []

            dt_polys = getattr(ocr_res, "dt_polys", None)
            if dt_polys is None and isinstance(ocr_res, dict):
                dt_polys = ocr_res.get("dt_polys", [])
            elif dt_polys is None:
                dt_polys = []

            for i, text in enumerate(rec_texts):
                conf = rec_scores[i] if i < len(rec_scores) else 1.0
                poly = dt_polys[i] if i < len(dt_polys) else None
                
                bbox = None
                if poly is not None:
                    if hasattr(poly, "tolist"):
                        bbox = poly.tolist()
                    else:
                        bbox = list(poly)

                texts.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": bbox,
                })

        return texts


    def is_loaded(self) -> bool:
        return self._loaded
