from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class JobStartedCreate(BaseModel):
    job_id: str
    source: str
    inference_mode: str | None = None
    status: str = "started"
    detection_model: str | None = None


class JobCompletedUpdate(BaseModel):
    job_id: str
    inference_mode: str | None = None
    processing_mode: str | None = None

    video_total_frames: int = 0
    video_width: int = 0
    video_height: int = 0

    frames_extracted: int = 0
    frames_sampled: int = 0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobFailedUpdate(BaseModel):
    job_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlateDetectionCreate(BaseModel):
    job_id: str
    frame_result_id: int | None = None
    frame_id: str | None = None
    frame_index: int | None = None

    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None

    plate_text: str | None = None
    vehicle_class: str | None = None
    confidence: float = 0.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    raw_plate: dict[str, Any] = Field(default_factory=dict)


class FrameResultCreate(BaseModel):
    job_id: str
    frame_id: str
    frame_index: int | None = None
    source: str
    timestamp_ms: float = 0.0

    inference_mode: str | None = None
    plate_count: int = 0

    processing_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    llm_time_ms: float = 0.0

class ResourceLoggingCreate(BaseModel):
    job_id: str
    cpu_percent: float | None = None
    ram_usage: float | None = None
    gpu_percent: float | None = None
    gpu_ram_usage: float | None = None
    gpu_power_usage: float | None = None
    gpu_name: str | None = None
