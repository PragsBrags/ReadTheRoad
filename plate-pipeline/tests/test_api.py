"""
Tests for API Routes, ingestion response differentiation, and results progress tracking.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

import main
from services.decoder import FrameData
from services.ingestion import ActiveStream


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with patched FFmpeg and Database dependencies."""
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.run") as mock_run, \
         patch("services.database.ResultPersistenceService") as mock_persistence:
        mock_run.return_value = MagicMock(stdout="ffmpeg version 6.0", returncode=0)
        with TestClient(main.app) as c:
            yield c


class TestAPIResults:

    def test_results_endpoint_unknown_job(self, client):
        """Endpoint raises 404 for unknown jobs."""
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        mock_cache.get_job_progress.return_value = {"total_frames": 0, "processed_frames": 0}
        mock_cache.get_job_results.return_value = None

        original_cache = main._services.get("cache")
        main._services["cache"] = mock_cache

        try:
            response = client.get("/results/unknown-job")
            assert response.status_code == 404
        finally:
            main._services["cache"] = original_cache

    def test_results_endpoint_processing(self, client):
        """Endpoint returns status 'processing' and aggregates on-the-fly."""
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        mock_cache.get_job_progress.return_value = {"total_frames": 10, "processed_frames": 5}
        mock_cache.get_job_results.return_value = [
            {"text": "ABC1234", "confidence": 0.9, "bbox": [0, 0, 100, 50]}
        ]

        original_cache = main._services.get("cache")
        main._services["cache"] = mock_cache

        try:
            response = client.get("/results/test-job-processing")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-processing"
            assert data["status"] == "processing"
            assert data["frames_sampled"] == 10
            assert data["frames_processed"] == 5
            assert len(data["plates"]) == 1
            # Aggregated on-the-fly, so frame_count exists
            assert data["plates"][0]["frame_count"] == 1
            assert data["plates"][0]["text"] == "ABC1234"
        finally:
            main._services["cache"] = original_cache

    def test_results_endpoint_completed(self, client):
        """Endpoint returns status 'completed' and caches finalized aggregated results."""
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        mock_cache.get_job_progress.return_value = {"total_frames": 10, "processed_frames": 10}
        mock_cache.get_job_results.return_value = [
            {"text": "ABC1234", "confidence": 0.9, "bbox": [0, 0, 100, 50]}
        ]

        original_cache = main._services.get("cache")
        main._services["cache"] = mock_cache

        try:
            response = client.get("/results/test-job-completed")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-completed"
            assert data["status"] == "completed"
            assert data["frames_sampled"] == 10
            assert data["frames_processed"] == 10
            # Ensure finalized results were written back to cache
            mock_cache.store_job_results.assert_called_once()
        finally:
            main._services["cache"] = original_cache

    def test_results_endpoint_streaming(self, client):
        """Endpoint returns status 'streaming' for active stream ingestions."""
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        mock_cache.get_job_progress.return_value = {"total_frames": 0, "processed_frames": 5}
        mock_cache.get_job_results.return_value = [
            {"text": "ABC1234", "confidence": 0.9, "bbox": [0, 0, 100, 50]}
        ]

        original_cache = main._services.get("cache")
        original_ingestion = main._services.get("ingestion")

        # Mock ingestion active streams
        mock_ingestion = MagicMock()
        mock_ingestion.active_streams = {
            "test-stream": ActiveStream(stream_id="test-stream", source="rtsp://test")
        }

        main._services["cache"] = mock_cache
        main._services["ingestion"] = mock_ingestion

        try:
            response = client.get("/results/test-stream")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "streaming"
        finally:
            main._services["cache"] = original_cache
            main._services["ingestion"] = original_ingestion


class TestAPIIngestion:

    @patch("services.ingestion.IngestionService._save_sampled_frame")
    @patch("services.decoder.FFmpegDecoder.extract_frames_sync")
    def test_ingest_file_local_mode(self, mock_extract, mock_save_frame, client):
        """Ingest file returns completed response synchronously in local mode."""
        # Create dummy FrameData
        frames = [
            FrameData(frame_id="f1", source="dummy", frame_bytes=b"dummy", frame_index=0, timestamp_ms=0.0),
            FrameData(frame_id="f2", source="dummy", frame_bytes=b"dummy", frame_index=1, timestamp_ms=100.0),
        ]
        mock_extract.return_value = frames

        original_ingestion = main._services.get("ingestion")
        ingestion = original_ingestion

        # Mock sampler to return the same frames (bypassing image conversion)
        mock_sampler = MagicMock()
        mock_sampler.sample.side_effect = lambda x: x

        # Mock dispatcher to return local mode with mock results
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_frames.return_value = ("local", [
            {"frame_id": "f1", "plates": [{"text": "ABC1234", "confidence": 0.95, "bbox": [0,0,10,10]}]}
        ])
        
        # Patch dispatcher, sampler, and cache on the ingestion service
        original_dispatcher = ingestion._dispatcher
        original_sampler = ingestion._sampler
        ingestion._dispatcher = mock_dispatcher
        ingestion._sampler = mock_sampler
        
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        original_cache = ingestion._cache
        ingestion._cache = mock_cache

        try:
            # Construct a dummy multipart file upload request
            files = {"file": ("test.mp4", b"fake-video-bytes", "video/mp4")}
            response = client.post("/ingest/file", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["processing_mode"] == "local"
            assert data["frames_sampled"] == 2
            assert data["frames_processed"] == 1
            assert len(data["plates"]) == 1
            assert "plates detected" in data["message"]
            
            # Verify cache methods called
            mock_cache.set_job_total_frames.assert_called_once()
            mock_cache.store_job_results.assert_called_once()
        finally:
            ingestion._dispatcher = original_dispatcher
            ingestion._sampler = original_sampler
            ingestion._cache = original_cache

    @patch("services.ingestion.IngestionService._save_sampled_frame")
    @patch("services.decoder.FFmpegDecoder.extract_frames_sync")
    def test_ingest_file_distributed_mode(self, mock_extract, mock_save_frame, client):
        """Ingest file returns processing response asynchronously in distributed mode."""
        frames = [
            FrameData(frame_id="f1", source="dummy", frame_bytes=b"dummy", frame_index=0, timestamp_ms=0.0),
            FrameData(frame_id="f2", source="dummy", frame_bytes=b"dummy", frame_index=1, timestamp_ms=100.0),
        ]
        mock_extract.return_value = frames

        original_ingestion = main._services.get("ingestion")
        ingestion = original_ingestion

        # Mock sampler to return the same frames (bypassing image conversion)
        mock_sampler = MagicMock()
        mock_sampler.sample.side_effect = lambda x: x

        # Mock dispatcher to return distributed mode (results are empty)
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_frames.return_value = ("distributed", [])
        
        # Patch dispatcher, sampler, and cache on the ingestion service
        original_dispatcher = ingestion._dispatcher
        original_sampler = ingestion._sampler
        ingestion._dispatcher = mock_dispatcher
        ingestion._sampler = mock_sampler
        
        mock_cache = MagicMock()
        mock_cache.is_connected = True
        original_cache = ingestion._cache
        ingestion._cache = mock_cache

        try:
            files = {"file": ("test.mp4", b"fake-video-bytes", "video/mp4")}
            response = client.post("/ingest/file", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["processing_mode"] == "distributed"
            assert data["frames_sampled"] == 2
            assert data["frames_processed"] == 0
            assert data["plates"] == []
            assert "queued" in data["message"]
            
            # Verify total frames set in cache
            mock_cache.set_job_total_frames.assert_called_once_with(data["job_id"], 2)
            # Verify store_job_results is NOT called (since it's async)
            mock_cache.store_job_results.assert_not_called()
        finally:
            ingestion._dispatcher = original_dispatcher
            ingestion._sampler = original_sampler
            ingestion._cache = original_cache
