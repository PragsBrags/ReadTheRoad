from services.database.models import Base, FrameResult, IngestionDetails, PlateDetection
from services.database.persistence_service import ResultPersistenceService

__all__ = [
    "Base",
    "IngestionDetails",
    "FrameResult",
    "PlateDetection",
    "ResultPersistenceService",
]