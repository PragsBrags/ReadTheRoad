from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase

class Base(DeclarativeBase):
    pass

class IngestionDetails(Base):
    __tablename__ = "ingestion_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    source: Mapped[str] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(32), default="started", nullable=False)

    inference_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)

    video_total_fps: Mapped[int] = mapped_column(Integer, default=0)
    video_width: Mapped[int] = mapped_column(Integer, default=0)
    video_height: Mapped[int] = mapped_column(Integer, default=0)
    frames_extracted: Mapped[int] = mapped_column(Integer, default=0)
    frames_sampled: Mapped[int] = mapped_column(Integer, default=0)

    detection_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    frame_results: Mapped[list["FrameResult"]] = relationship(back_populates="job")

class FrameResult(Base):
    __tablename__ = "frame_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("ingestion_details.job_id"), index=True)
    frame_id: Mapped[str] = mapped_column(String(64), index=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(Text)
    timestamp_ms: Mapped[float] = mapped_column(Float, default=0.0)

    inference_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plate_count: Mapped[int] = mapped_column(Integer, default=0)

    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    detection_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ocr_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    job: Mapped[IngestionDetails] = relationship(back_populates="frame_results")
    plates: Mapped[list["PlateDetection"]] = relationship(back_populates="frame_result")

class PlateDetection(Base):
    __tablename__ = "plate_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    frame_result_id: Mapped[int | None] = mapped_column(ForeignKey("frame_results.id"), nullable=True)
    frame_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frame_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    bbox_x1: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_x2: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_y1: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_y2: Mapped[float] = mapped_column(Float, default=0.0)

    plate_text: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    raw_plate: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    frame_result: Mapped[FrameResult | None] = relationship(back_populates="plates")

class ResourceLogging(Base):
    __tablename__ = "logging_resource"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("ingestion_details.job_id"), index=True)

    cpu_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)
    ram_usage: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)

    gpu_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)
    gpu_ram_usage: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)
    gpu_power_usage: Mapped[float] = mapped_column(Float, default=0.0, nullable=True)
    gpu_name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
