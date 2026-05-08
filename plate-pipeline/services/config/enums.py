"""
Configuration Enums — All enum types used across the pipeline config.
"""

from __future__ import annotations

from enum import Enum


class SystemMode(str, Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"


class InferenceMode(str, Enum):
    YOLO_ONLY = "yolo_only"
    OCR_ONLY = "ocr_only"
    YOLO_OCR = "yolo_ocr"
    YOLO_OCR_LLM = "yolo_ocr_llm"
    DIRECT_LLM = "direct_llm"


class SamplerStrategy(str, Enum):
    FIXED_FPS = "fixed_fps"
    MOTION = "motion"
    HYBRID = "hybrid"
    NYQUIST = "nyquist"


class OCREngine(str, Enum):
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMMode(str, Enum):
    GATED = "gated"
    ALWAYS = "always"
    DISABLED = "disabled"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
