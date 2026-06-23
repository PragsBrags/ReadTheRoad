from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from services.database.models import FrameResult, IngestionDetails, PlateDetection, ResourceLogging
from services.database.schema import (
    FrameResultCreate,
    JobCompletedUpdate,
    JobFailedUpdate,
    JobStartedCreate,
    PlateDetectionCreate,
    ResourceLoggingCreate,
)

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

        row.video_total_fps = data.video_total_frames

        row.video_width = data.video_width
        row.video_height = data.video_height

        # Do not overwrite original created_at (set at insert time by DB).
        # Only set completed_at to mark job completion time.
        row.completed_at = data.completed_at
        row.frames_extracted = data.frames_extracted
        row.frames_sampled = data.frames_sampled
        row.status = "completed"

    def failed_job(self, db: Session, data: JobFailedUpdate) -> None:
        row = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if row is None:
            return
        # Preserve original created_at; only update status on failure.
        row.status = "failed"

    def save_resource_log(self, db: Session, data: ResourceLoggingCreate) -> None:
        db.query(ResourceLogging).filter_by(job_id=data.job_id).delete(
            synchronize_session=False
        )
        db.add(ResourceLogging(**data.model_dump()))


    def save_resource_logs(self, db: Session, rows: list[ResourceLoggingCreate]) -> None:
        job_ids = {row.job_id for row in rows}
        if job_ids:
            db.query(ResourceLogging).filter(ResourceLogging.job_id.in_(job_ids)).delete(
                synchronize_session=False
            )
        db.add_all(ResourceLogging(**row.model_dump()) for row in rows)


    def update_job_metrics(self, db: Session, data: JobCompletedUpdate) -> None:
        row = db.query(IngestionDetails).filter_by(job_id=data.job_id).one_or_none()
        if row is None:
            return

        # Avoid overwriting `created_at` here; only apply other metrics.
        for key, value in data.model_dump(exclude={"job_id", "created_at"}).items():
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
            frame_index=data.frame_index,
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
                        "frame_index": data.frame_index,
                    }
                ),
            )

    def save_plate(self, db: Session, data: PlateDetectionCreate) -> None:
        # Determine bbox values: prefer explicit fields, fallback to raw_plate["bbox"] if present
        bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = 0.0
        if getattr(data, "bbox_x1", None) is not None and getattr(data, "bbox_y1", None) is not None and getattr(data, "bbox_x2", None) is not None and getattr(data, "bbox_y2", None) is not None:
            bbox_x1 = float(data.bbox_x1)
            bbox_y1 = float(data.bbox_y1)
            bbox_x2 = float(data.bbox_x2)
            bbox_y2 = float(data.bbox_y2)
        else:
            rp = data.raw_plate or {}
            bbox = None
            if isinstance(rp, dict):
                bbox = rp.get("bbox") or rp.get("bounding_box") or rp.get("bbox_xyxy")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(float, bbox[:4])

        row = PlateDetection(
            job_id=data.job_id,
            frame_result_id=data.frame_result_id,
            frame_id=data.frame_id,
            frame_index=data.frame_index,
            bbox_x1=bbox_x1,
            bbox_x2=bbox_x2,
            bbox_y1=bbox_y1,
            bbox_y2=bbox_y2,
            plate_text=data.plate_text,
            vehicle_class=data.vehicle_class,
            confidence=data.confidence,
            raw_plate=data.raw_plate,
            created_at=data.created_at,
        )
        db.add(row)
