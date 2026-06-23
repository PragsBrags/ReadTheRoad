from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.database.models import Base, FrameResult, PlateDetection
from services.database.repository import ResultRepository
from services.database.schema import FrameResultCreate, PlateDetectionCreate


def test_save_frame_persists_frame_index_on_frame_and_plates():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as db:
        repo = ResultRepository()
        repo.save_frame(
            db,
            FrameResultCreate(
                job_id="job-1",
                frame_id="frame-1",
                frame_index=7,
                source="video.mp4",
                timestamp_ms=123.0,
            ),
            [
                PlateDetectionCreate(
                    job_id="job-1",
                    frame_id="will-be-overwritten",
                    frame_index=99,
                    plate_text="ABC123",
                    confidence=0.9,
                    raw_plate={"bbox": [1, 2, 3, 4]},
                )
            ],
        )
        db.commit()

        frame = db.query(FrameResult).one()
        plate = db.query(PlateDetection).one()

        assert frame.frame_index == 7
        assert plate.frame_result_id == frame.id
        assert plate.frame_id == "frame-1"
        assert plate.frame_index == 7

