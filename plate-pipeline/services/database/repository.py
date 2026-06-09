from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from services.database.models import FrameResult, IngestionDetails, PlateDetection, ResourceLogging
from services.database.schema import FrameResultCreate, JobCompletedUpdate, JobFailedUpdate, JobStartedCreate, PlateDetectionCreate, ResourceLoggingCreate

class ResultRepository:
    def create_job(self, db: Session, data: JobStartedCreate) -> IngestionDetails:
        existing = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if existing:
            return existing

        row = IngestionDetails(
            job_id=data.job_id,
            source=data.source,
            inference_mode=data.inference_mode,
            status=data.status,
            detection_model=data.detection_model,
        )
        db.add(row)
        return row

    def complete_job(self, db: Session, data: JobCompletedUpdate) -> None:
        row = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if row is None:
            return
        
        row.inference_mode = data.inference_mode
        row.processing_mode = data.processing_mode
        row.created_at = data.created_at
        row.status = "completed"

    def failed_job(self, db: Session, data: JobFailedUpdate) -> None:
        row = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if row is None:
            return
        
        row.created_at = data.created_at
        row.status = "failed"

    def save_resource_log(self, db: Session, data: ResourceLoggingCreate) -> None:
        db.add(ResourceLogging(**data.model_dump()))


    def save_resource_logs(self, db: Session, rows: list[ResourceLoggingCreate]) -> None:
        db.add_all(ResourceLogging(**row.model_dump()) for row in rows)


    def update_job_metrics(self, db: Session, data: JobCompletedUpdate) -> None:
        row = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if row is None:
            return

        for key, value in data.model_dump(exclude={"job_id"}).items():
            setattr(row, key, value)

    def save_frame(
    self,
    db: Session,
    data: FrameResultCreate,
    plates: list[PlateDetectionCreate],
        ) -> None:
        row = FrameResult(
            job_id=data.job_id,
            frame_id=data.frame_id,
            source=data.source,
            timestamp_ms=data.timestamp_ms,
            inference_mode=data.inference_mode,
            plate_count=len(plates),
            processing_time_ms=data.processing_time_ms,
            detection_time_ms=data.detection_time_ms,
            ocr_time_ms=data.ocr_time_ms,
            llm_time_ms=data.llm_time_ms,
        )
        db.add(row)
        db.flush()

        for plate in plates:
            self.save_plate(
                db,
                plate.model_copy(
                    update={
                        "frame_result_id": row.id,
                        "frame_id": data.frame_id,
                    }
                ),
            )

    def save_plate(self, db: Session, data: PlateDetectionCreate) -> None:
        row = PlateDetection(
            job_id=data.job_id,
            frame_result_id=data.frame_result_id,
            frame_id=data.frame_id,
            plate_text=data.plate_text,
            vehicle_class=data.vehicle_class,
            confidence=data.confidence,
            raw_plate=data.raw_plate,
            created_at=data.created_at,
        )
        db.add(row)
        