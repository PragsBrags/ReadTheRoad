"""
Inference Package — Config-driven inference pipeline routing.

Re-exports all public symbols so consumers can do:
    from services.inference import InferenceRouter, CircuitBreaker
"""

from services.inference.circuit_breaker import CircuitBreaker
from services.inference.router import InferenceRouter
from services.inference.stages import (
    DetectionStage,
    DirectLLMStage,
    LLMCorrectionStage,
    OCRStage,
    Stage,
)

__all__ = [
    "CircuitBreaker",
    "InferenceRouter",
    "Stage",
    "DetectionStage",
    "OCRStage",
    "LLMCorrectionStage",
    "DirectLLMStage",
]
