"""
Model Factory — Creates model instances based on config.

Centralizes all model instantiation logic.
"""

from __future__ import annotations

from services.config import LLMProvider, OCREngine, PipelineConfig
from services.models.base import DetectionModel, LLMClient, OCRModel
from services.models.detection import YOLODetector
from services.models.llm import OllamaClient, OpenAIClient
from services.models.ocr import EasyOCRReader, PaddleOCRReader


class ModelFactory:
    """
    Factory Pattern — creates model instances based on config.

    Centralizes all model instantiation logic.
    """

    @staticmethod
    def create_detector(config: PipelineConfig) -> DetectionModel:
        """Create a YOLO detector."""
        detector = YOLODetector(config)
        detector.load()
        return detector

    @staticmethod
    def create_ocr(config: PipelineConfig) -> OCRModel:
        """Create an OCR reader based on config."""
        engine = config.ocr.engine

        if engine == OCREngine.EASYOCR:
            reader = EasyOCRReader(config)
        elif engine == OCREngine.PADDLEOCR:
            reader = PaddleOCRReader(config)
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

        reader.load()
        return reader

    @staticmethod
    def create_llm(config: PipelineConfig) -> LLMClient:
        """Create an LLM client based on config."""
        provider = config.llm.provider

        if provider == LLMProvider.OPENAI:
            client = OpenAIClient(config)
        elif provider == LLMProvider.OLLAMA:
            client = OllamaClient(config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        client.load()
        return client
