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

    inference_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    frame_results: Mapped[list["FrameResult"]] = relationship(back_populates="job")

class FrameResult(Base):
    __tablename__ = "frame_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), ForeignKey("ingestion_details.job_id"), index=True)
    frame_id: Mapped[str] = mapped_column(String(64), index=True)
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

    plate_text: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vehicle_class: Mapped[str] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    raw_plate: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    frame_result: Mapped[FrameResult | None] = relationship(back_populates="plates")