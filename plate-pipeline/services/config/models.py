"""
Configuration Models — Pydantic models for type-safe pipeline config.

All system behavior is controlled via config/config.yaml.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from services.config.enums import (
    DeviceType,
    InferenceMode,
    LLMMode,
    LLMProvider,
    OCREngine,
    SamplerStrategy,
    SystemMode,
)


# SYSTEM

class SystemConfig(BaseModel):
    mode: SystemMode = SystemMode.PRODUCTION


# VIDEO INGESTION

class VideoIngestionConfig(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["file", "rtsp", "rtmp"])
    max_streams: int = Field(default=20, ge=1, le=100)
    upload_dir: str = "/tmp/plate-pipeline/uploads"
    frame_output_dir: str = "/tmp/plate-pipeline/frames"


# DECODER

class FFmpegSettings(BaseModel):
    binary_path: str = "ffmpeg"
    frame_filter: str = "eq(pict_type,P)"
    vsync: str = "vfr"
    output_format: str = "jpg"
    output_quality: int = Field(default=2, ge=1, le=31)


class DecoderConfig(BaseModel):
    engine: str = "ffmpeg"
    ffmpeg: FFmpegSettings = Field(default_factory=FFmpegSettings)
    hardware_acceleration: bool = False
    timeout_seconds: int = Field(default=300, ge=10)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        if v != "ffmpeg":
            raise ValueError("Only 'ffmpeg' is allowed as the decoder engine")
        return v


# FRAME SAMPLER

class FrameSamplerConfig(BaseModel):
    strategy: SamplerStrategy = SamplerStrategy.HYBRID
    fixed_fps: float = Field(default=2.0, gt=0)
    motion_threshold: float = Field(default=0.2, ge=0, le=1)
    nyquist_event_frequency: float = Field(default=1.0, gt=0)
    downscale_factor: float = Field(default=1.0, gt=0, le=1)

    motion_area_threshold: float = Field(default=100, gt=0)
    cooldown_frames: int = Field(default=5, ge=0)
    warmup_frames: int = Field(default=10, ge=0)


# INFERENCE

class InferenceModeEntry(BaseModel):
    enabled: bool = False
    description: str = ""


class InferenceModesConfig(BaseModel):
    yolo_only: InferenceModeEntry = Field(default_factory=InferenceModeEntry)
    ocr_only: InferenceModeEntry = Field(default_factory=InferenceModeEntry)
    yolo_ocr: InferenceModeEntry = Field(default_factory=InferenceModeEntry)
    yolo_ocr_llm: InferenceModeEntry = Field(default_factory=InferenceModeEntry)
    direct_llm: InferenceModeEntry = Field(default_factory=InferenceModeEntry)


class InferenceConfig(BaseModel):
    mode: InferenceMode = InferenceMode.YOLO_OCR
    modes: InferenceModesConfig = Field(default_factory=InferenceModesConfig)

    @field_validator("mode")
    @classmethod
    def validate_active_mode(cls, v: InferenceMode) -> InferenceMode:
        return v


# MODEL REGISTRY
class YOLOModelConfig(BaseModel):
    path: str = "/models/plate.pt"
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    device: DeviceType = DeviceType.CPU
    imgsz: int = Field(default=640, ge=32)
    # Matches config.yaml/config.local.yaml key: 'selection_policy'
    selection_policy: str = "all"


class ModelRegistryConfig(BaseModel):
    yolo: YOLOModelConfig = Field(default_factory=YOLOModelConfig)


# OCR

class EasyOCRSettings(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["en"])
    gpu: bool = False
    detail: int = 1


class PaddleOCRSettings(BaseModel):
    lang: str = "en"
    use_angle_cls: bool = True
    use_gpu: bool = False


class OCRConfig(BaseModel):
    engine: OCREngine = OCREngine.EASYOCR
    enabled: bool = True
    easyocr: EasyOCRSettings = Field(default_factory=EasyOCRSettings)
    paddleocr: PaddleOCRSettings = Field(default_factory=PaddleOCRSettings)

class PreprocessingConfig(BaseModel):
    enabled: bool = True
    min_plate_width: int = Field(default=100, ge=1)
    enhance_image_ocr: bool = True

class DebugConfig(BaseModel):
    save_intermediate_images: bool = False
    output_dir: str = "tmp/plate-pipeline/debug"


# LLM

class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: int = Field(default=60, ge=1)
    half_open_max_calls: int = Field(default=2, ge=1)


class OpenAISettings(BaseModel):
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = Field(default=100, ge=1)
    temperature: float = Field(default=0.1, ge=0, le=2)


class AnthropicSettings(BaseModel):
    model: str = "claude-3-haiku-20240307"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = Field(default=100, ge=1)


class OllamaSettings(BaseModel):
    model: str = "llava"
    base_url: str = "http://localhost:11434"


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: LLMProvider = LLMProvider.OPENAI
    mode: LLMMode = LLMMode.GATED
    max_calls_per_job: int = Field(default=1, ge=0)
    fallback_to_ocr: bool = True
    timeout_seconds: int = Field(default=30, ge=1)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


# DSPy

class DSPyConfig(BaseModel):
    enabled: bool = True
    prompt_compression: bool = True
    optimizer: str = "bootstrap_fewshot"
    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=16, ge=0)
    lm_model: str = "openai/gpt-4o-mini"


# QUEUE / WORKERS

class QueueConfig(BaseModel):
    broker: str = "rabbitmq"
    broker_url: str = "amqp://guest:guest@rabbitmq:5672//"
    prefetch_count: int = Field(default=10, ge=1)
    frame_queue: str = "plate_frames"
    result_queue: str = "plate_results"
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: int = Field(default=5, ge=1)


class CeleryAutoscaleConfig(BaseModel):
    min: int = Field(default=2, ge=1)
    max: int = Field(default=16, ge=1)


class CelerySettings(BaseModel):
    broker_url: str = "amqp://guest:guest@rabbitmq:5672//"
    result_backend: str = "redis://redis:6379/0"
    concurrency: int = Field(default=4, ge=1)
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = Field(default_factory=lambda: ["json"])
    task_acks_late: bool = True
    worker_prefetch_multiplier: int = Field(default=1, ge=1)
    autoscale: CeleryAutoscaleConfig = Field(default_factory=CeleryAutoscaleConfig)


class WorkersConfig(BaseModel):
    celery: CelerySettings = Field(default_factory=CelerySettings)


# REDIS

class RedisConfig(BaseModel):
    enabled: bool = True
    url: str = "redis://redis:6379/0"
    ttl_seconds: int = Field(default=300, ge=1)
    deduplication: bool = True
    dedup_ttl_seconds: int = Field(default=60, ge=1)
    key_prefix: str = "plate_pipeline"


# AGGREGATION

class AggregationConfig(BaseModel):
    window_ms: int = Field(default=3000, ge=100)
    dedup_threshold: float = Field(default=0.85, ge=0, le=1)
    min_confidence: float = Field(default=0.4, ge=0, le=1)
    max_results_per_window: int = Field(default=50, ge=1)


# MONITORING

class PrometheusConfig(BaseModel):
    enabled: bool = True
    port: int = Field(default=8000, ge=1)
    path: str = "/metrics"


class GrafanaConfig(BaseModel):
    enabled: bool = True
    port: int = Field(default=3000, ge=1)


class MonitoringConfig(BaseModel):
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    grafana: GrafanaConfig = Field(default_factory=GrafanaConfig)


# ROOT CONFIG

class PipelineConfig(BaseModel):


    system: SystemConfig = Field(default_factory=SystemConfig)
    video_ingestion: VideoIngestionConfig = Field(default_factory=VideoIngestionConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    frame_sampler: FrameSamplerConfig = Field(default_factory=FrameSamplerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dspy: DSPyConfig = Field(default_factory=DSPyConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
