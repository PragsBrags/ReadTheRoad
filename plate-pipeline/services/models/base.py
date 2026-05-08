"""
Model Interfaces — Abstract base classes for detection, OCR, and LLM models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class DetectionModel(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect license plates in an image.

        Args:
            image: Input image as numpy array (BGR or RGB).

        Returns:
            List of detections, each with:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_name: str
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...


class OCRModel(ABC):
    """Abstract interface for OCR engines."""

    @abstractmethod
    def read_text(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Extract text from a cropped plate image.

        Args:
            image: Cropped plate image as numpy array.

        Returns:
            List of text results, each with:
                - text: str
                - confidence: float
                - bbox: optional bounding box
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def correct_plate_text(
        self,
        ocr_text: str,
        image_bytes: Optional[bytes] = None,
    ) -> dict[str, Any]:
        """
        Correct or extract plate text using LLM.

        Args:
            ocr_text: OCR-extracted text (may be empty for direct mode).
            image_bytes: Optional cropped plate image bytes.

        Returns:
            Dict with:
                - corrected_text: str
                - confidence: float
                - reasoning: str (optional)
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
