"""
Tests for InferenceRouter and pipeline stages.
"""

import io
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image
from services.config import InferenceMode, PipelineConfig
from services.decoder import FrameData
from services.inference import (
    CircuitBreaker, DetectionStage, InferenceRouter, OCRStage, Stage,
)


def make_test_frame(w=200, h=100):
    img = Image.new("RGB", (w, h), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return FrameData(
        frame_id="test-frame", source="test.mp4",
        frame_bytes=buf.getvalue(), frame_index=0, timestamp_ms=0.0,
    )


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_half_open_after_recovery(self):
        import time
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half_open"

    def test_closes_after_half_open_success(self):
        import time
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.02)
        cb.record_success()
        cb.record_success()
        assert cb.state == "closed"


class TestStageInterface:
    def test_stage_is_abstract(self):
        with pytest.raises(TypeError):
            Stage()


class TestDetectionStage:
    def test_no_model_returns_empty(self):
        registry = MagicMock()
        registry.detector = None
        stage = DetectionStage(registry)
        result = stage.process({"image": np.zeros((100, 100, 3), dtype=np.uint8)})
        assert result["detections"] == []

    def test_with_mock_model(self):
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            {"bbox": [10, 20, 100, 60], "confidence": 0.95, "class_name": "license_plate"}
        ]
        registry = MagicMock()
        registry.detector = mock_det
        stage = DetectionStage(registry)
        result = stage.process({"image": np.zeros((100, 200, 3), dtype=np.uint8)})
        assert len(result["detections"]) == 1


class TestInferenceRouter:
    def test_pipelines_for_all_modes(self):
        registry = MagicMock()
        config = PipelineConfig()
        router = InferenceRouter(config, registry)
        for mode in InferenceMode:
            assert mode in router._pipelines

    def test_process_returns_result(self):
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        registry = MagicMock()
        registry.detector = mock_det
        registry.ocr = MagicMock()
        registry.ocr.read_text.return_value = []
        registry.llm = MagicMock()
        config = PipelineConfig()
        router = InferenceRouter(config, registry)
        result = router.process(make_test_frame())
        assert result["status"] == "success"
        assert "plates" in result
