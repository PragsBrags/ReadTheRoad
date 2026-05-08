"""
Config Package — Type-safe, validated configuration from YAML.

Re-exports all public symbols so consumers can do:
    from services.config import load_config, PipelineConfig, InferenceMode
"""

# Enums
from services.config.enums import (
    DeviceType,
    InferenceMode,
    LLMMode,
    LLMProvider,
    OCREngine,
    SamplerStrategy,
    SystemMode,
)

# Pydantic models
from services.config.models import (
    AggregationConfig,
    AnthropicSettings,
    CeleryAutoscaleConfig,
    CelerySettings,
    CircuitBreakerConfig,
    DecoderConfig,
    DSPyConfig,
    EasyOCRSettings,
    FFmpegSettings,
    FrameSamplerConfig,
    GrafanaConfig,
    InferenceConfig,
    InferenceModeEntry,
    InferenceModesConfig,
    LLMConfig,
    ModelRegistryConfig,
    MonitoringConfig,
    OCRConfig,
    OllamaSettings,
    OpenAISettings,
    PaddleOCRSettings,
    PipelineConfig,
    PrometheusConfig,
    QueueConfig,
    RedisConfig,
    SystemConfig,
    VideoIngestionConfig,
    WorkersConfig,
    YOLOModelConfig,
)

# Loader
from services.config.loader import load_config, reload_config

__all__ = [
    # Enums
    "DeviceType",
    "InferenceMode",
    "LLMMode",
    "LLMProvider",
    "OCREngine",
    "SamplerStrategy",
    "SystemMode",
    # Models
    "AggregationConfig",
    "AnthropicSettings",
    "CeleryAutoscaleConfig",
    "CelerySettings",
    "CircuitBreakerConfig",
    "DecoderConfig",
    "DSPyConfig",
    "EasyOCRSettings",
    "FFmpegSettings",
    "FrameSamplerConfig",
    "GrafanaConfig",
    "InferenceConfig",
    "InferenceModeEntry",
    "InferenceModesConfig",
    "LLMConfig",
    "ModelRegistryConfig",
    "MonitoringConfig",
    "OCRConfig",
    "OllamaSettings",
    "OpenAISettings",
    "PaddleOCRSettings",
    "PipelineConfig",
    "PrometheusConfig",
    "QueueConfig",
    "RedisConfig",
    "SystemConfig",
    "VideoIngestionConfig",
    "WorkersConfig",
    "YOLOModelConfig",
    # Loader
    "load_config",
    "reload_config",
]
