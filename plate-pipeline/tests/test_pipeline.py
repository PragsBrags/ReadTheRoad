"""
Tests for end-to-end pipeline validation.
"""

import pytest
from services.config import load_config, reload_config, PipelineConfig, AggregationConfig
from services.aggregation import AggregationService, string_similarity


class TestConfigLoader:
    def test_loads_valid_config(self):
        reload_config()
        config = load_config()
        assert isinstance(config, PipelineConfig)

    def test_decoder_is_ffmpeg(self):
        config = load_config()
        assert config.decoder.engine == "ffmpeg"

    def test_inference_mode_is_valid(self):
        config = load_config()
        assert config.inference.mode.value in [
            "yolo_only", "ocr_only", "yolo_ocr", "yolo_ocr_llm", "direct_llm"
        ]

    def test_active_mode_is_enabled(self):
        config = load_config()
        mode = config.inference.mode.value
        mode_cfg = getattr(config.inference.modes, mode)
        assert mode_cfg.enabled is True


class TestAggregation:
    def test_string_similarity_identical(self):
        assert string_similarity("ABC1234", "ABC1234") == 1.0

    def test_string_similarity_different(self):
        sim = string_similarity("ABC1234", "XYZ9999")
        assert sim < 0.5

    def test_dedup_merges_similar(self):
        config = AggregationConfig(dedup_threshold=0.8, window_ms=5000)
        agg = AggregationService(config)

        agg.add_result("job1", {"plates": [
            {"text": "ABC1234", "confidence": 0.9, "bbox": [0, 0, 100, 50]}
        ]})
        agg.add_result("job1", {"plates": [
            {"text": "ABC1234", "confidence": 0.95, "bbox": [0, 0, 100, 50]}
        ]})

        results = agg.flush("job1")
        assert len(results) == 1
        assert results[0]["frame_count"] == 2

    def test_distinct_plates_not_merged(self):
        config = AggregationConfig(dedup_threshold=0.9, window_ms=5000)
        agg = AggregationService(config)

        agg.add_result("job1", {"plates": [
            {"text": "ABC1234", "confidence": 0.9, "bbox": [0, 0, 100, 50]},
            {"text": "XYZ5678", "confidence": 0.85, "bbox": [200, 0, 300, 50]},
        ]})

        results = agg.flush("job1")
        assert len(results) == 2
