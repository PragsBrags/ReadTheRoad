"""
Tests for lazy loading and selective model loading.
"""

from unittest.mock import MagicMock, patch
import pytest
from services.config import PipelineConfig, InferenceMode
from services.models.registry import ModelRegistry


def test_no_loading_on_init():
    """Verify that ModelRegistry initialization does not load any models."""
    config = PipelineConfig()
    with patch("services.models.factory.ModelFactory.create_detector") as mock_create_detector, \
         patch("services.models.factory.ModelFactory.create_ocr") as mock_create_ocr, \
         patch("services.models.factory.ModelFactory.create_llm") as mock_create_llm:

        registry = ModelRegistry(config)
        assert registry._detector is None
        assert registry._ocr is None
        assert registry._llm is None

        assert registry.detector_loaded is False
        assert registry.ocr_loaded is False
        assert registry.llm_available is False

        mock_create_detector.assert_not_called()
        mock_create_ocr.assert_not_called()
        mock_create_llm.assert_not_called()


def test_lazy_loading_on_property_access():
    """Verify that properties lazy-load the models on first access."""
    config = PipelineConfig()
    config.ocr.enabled = True
    config.llm.enabled = True

    mock_det = MagicMock()
    mock_det.is_loaded.return_value = True

    mock_ocr = MagicMock()
    mock_ocr.is_loaded.return_value = True

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True

    with patch("services.models.factory.ModelFactory.create_detector", return_value=mock_det) as mock_create_detector, \
         patch("services.models.factory.ModelFactory.create_ocr", return_value=mock_ocr) as mock_create_ocr, \
         patch("services.models.factory.ModelFactory.create_llm", return_value=mock_llm) as mock_create_llm:

        registry = ModelRegistry(config)

        # Access detector property
        assert registry.detector == mock_det
        mock_create_detector.assert_called_once_with(config)
        assert registry.detector_loaded is True

        # Access ocr property
        assert registry.ocr == mock_ocr
        mock_create_ocr.assert_called_once_with(config)
        assert registry.ocr_loaded is True

        # Access llm property
        assert registry.llm == mock_llm
        mock_create_llm.assert_called_once_with(config)
        assert registry.llm_available is True


def test_status_helpers_do_not_trigger_load():
    """Verify that status properties do not trigger lazy-loading."""
    config = PipelineConfig()
    with patch("services.models.factory.ModelFactory.create_detector") as mock_create_detector, \
         patch("services.models.factory.ModelFactory.create_ocr") as mock_create_ocr, \
         patch("services.models.factory.ModelFactory.create_llm") as mock_create_llm:

        registry = ModelRegistry(config)

        # Check status flags
        assert registry.detector_loaded is False
        assert registry.ocr_loaded is False
        assert registry.llm_available is False

        # Verify no models were loaded
        mock_create_detector.assert_not_called()
        mock_create_ocr.assert_not_called()
        mock_create_llm.assert_not_called()


@pytest.mark.parametrize(
    "mode,expected_needed",
    [
        (InferenceMode.YOLO_ONLY, {"detector"}),
        (InferenceMode.OCR_ONLY, {"ocr"}),
        (InferenceMode.YOLO_OCR, {"detector", "ocr"}),
        (InferenceMode.YOLO_OCR_LLM, {"detector", "ocr", "llm"}),
        (InferenceMode.DIRECT_LLM, {"detector", "llm"}),
    ]
)
def test_selective_loading_in_load_all(mode, expected_needed):
    """Verify that load_all only loads the models required for the active mode."""
    config = PipelineConfig()
    config.inference.mode = mode
    config.ocr.enabled = True
    config.llm.enabled = True

    mock_det = MagicMock()
    mock_det.is_loaded.return_value = True

    mock_ocr = MagicMock()
    mock_ocr.is_loaded.return_value = True

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True

    with patch("services.models.factory.ModelFactory.create_detector", return_value=mock_det) as mock_create_detector, \
         patch("services.models.factory.ModelFactory.create_ocr", return_value=mock_ocr) as mock_create_ocr, \
         patch("services.models.factory.ModelFactory.create_llm", return_value=mock_llm) as mock_create_llm:

        registry = ModelRegistry(config)
        registry.load_all()

        if "detector" in expected_needed:
            mock_create_detector.assert_called_once_with(config)
            assert registry.detector_loaded is True
        else:
            mock_create_detector.assert_not_called()
            assert registry.detector_loaded is False

        if "ocr" in expected_needed:
            mock_create_ocr.assert_called_once_with(config)
            assert registry.ocr_loaded is True
        else:
            mock_create_ocr.assert_not_called()
            assert registry.ocr_loaded is False

        if "llm" in expected_needed:
            mock_create_llm.assert_called_once_with(config)
            assert registry.llm_available is True
        else:
            mock_create_llm.assert_not_called()
            assert registry.llm_available is False
