from __future__ import annotations

import logging
from typing import Any

from services.config import DatabaseConfig
from services.database.repository import ResultRepository
from services.database.session import DatabaseSessionManager
from services.database.schema import (
    JobStartedCreate,
    JobCompletedUpdate,
    JobFailedUpdate,
    FrameResultCreate,
    PlateDetectionCreate,
)

logger = logging.getLogger(__name__)


class ResultPersistenceService:
    def __init__(self, config: DatabaseConfig):
        self.enabled = config.enabled
        self._manager = DatabaseSessionManager(config)
        self._repo = ResultRepository()

        if self.enabled:
            self._manager.create_tables()
            logger.info("Database persistence enabled")

    def save_job_started(
        self,
        job: JobStartedCreate,
    ) -> None:
        if not self.enabled:
            return

        with self._manager.session() as db:
            self._repo.create_job(db, job)

    def save_job_completed(self, *, job: JobCompletedUpdate) -> None:
        if not self.enabled:
            return

        with self._manager.session() as db:
            self._repo.complete_job(db, job)

    def save_job_failed(self, *, job: JobFailedUpdate) -> None:
        if not self.enabled:
            return

        with self._manager.session() as db:
            self._repo.failed_job(db, job)

    def save_frame_result(
    self,
    *,
    job: FrameResultCreate,
    plates: list[PlateDetectionCreate],
        ) -> None:
        if not self.enabled:
            return

        with self._manager.session() as db:
            self._repo.save_frame(db, job, plates)
