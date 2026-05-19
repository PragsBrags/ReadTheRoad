from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class JobStartedCreate(BaseModel):
    job_id: str
    source: str
    inference_mode: str | None = None


class JobCompletedUpdate(BaseModel):
    job_id: str
    inference_mode: str | None = None
    processing_mode: str | None = None
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

    plate_text: str | None = None
    vehicle_class: str | None = None
    confidence: float = 0.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    raw_plate: dict[str, Any] = Field(default_factory=dict)


class FrameResultCreate(BaseModel):
    job_id: str
    frame_id: str
    source: str
    timestamp_ms: float = 0.0

    inference_mode: str | None = None
    plate_count: int = 0

    processing_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    llm_time_ms: float = 0.0
