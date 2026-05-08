"""
Models Package — Model loading, registry, and factory patterns.

Re-exports all public symbols so consumers can do:
    from services.models import ModelRegistry, YOLODetector, DetectionModel
"""

from services.models.base import DetectionModel, LLMClient, OCRModel
from services.models.detection import YOLODetector
from services.models.factory import ModelFactory
from services.models.llm import OllamaClient, OpenAIClient
from services.models.ocr import EasyOCRReader, PaddleOCRReader
from services.models.registry import ModelRegistry

__all__ = [
    # Interfaces
    "DetectionModel",
    "OCRModel",
    "LLMClient",
    # Implementations
    "YOLODetector",
    "EasyOCRReader",
    "PaddleOCRReader",
    "OpenAIClient",
    "OllamaClient",
    # Factory & Registry
    "ModelFactory",
    "ModelRegistry",
]
