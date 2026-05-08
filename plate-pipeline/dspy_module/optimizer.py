"""
DSPy Optimizer — Prompt optimization for plate text correction.

Uses DSPy's declarative framework to define and optimize prompts
for license plate text correction. Supports:
  - ChainOfThought reasoning
  - BootstrapFewShot optimization
  - Prompt compression

Requires training data for optimization. Provides a stub dataset
and metric for initial development.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.config import DSPyConfig

logger = logging.getLogger(__name__)


# DSPy SIGNATURES & MODULES

def _create_plate_correction_module(config: DSPyConfig):
    """
    Create a DSPy module for plate text correction.

    Signature: ocr_text, context -> corrected_plate

    Returns the module and the configured LM, or None if DSPy
    is not available.
    """
    try:
        import dspy
    except ImportError:
        logger.warning("DSPy not installed. Install with: pip install dspy-ai")
        return None, None

    # Configure the LM
    try:
        lm = dspy.LM(config.lm_model)
        dspy.configure(lm=lm)
        logger.info(f"DSPy configured with LM: {config.lm_model}")
    except Exception as e:
        logger.error(f"Failed to configure DSPy LM: {e}")
        return None, None

    # Define the signature
    class PlateCorrection(dspy.Signature):
        """Correct OCR-extracted license plate text.

        Common OCR errors on license plates:
        - 0/O confusion
        - 1/I/L confusion
        - 5/S confusion
        - 8/B confusion
        - 2/Z confusion

        Given the raw OCR text and optional context, produce the
        corrected license plate string.
        """
        ocr_text: str = dspy.InputField(
            desc="Raw OCR-extracted text from a license plate image"
        )
        context: str = dspy.InputField(
            desc="Additional context (country, plate format, etc.)",
            default="",
        )
        corrected_plate: str = dspy.OutputField(
            desc="Corrected license plate text"
        )

    # Create the module
    class PlateCorrector(dspy.Module):
        def __init__(self):
            super().__init__()
            self.correct = dspy.ChainOfThought(PlateCorrection)

        def forward(self, ocr_text: str, context: str = "") -> dspy.Prediction:
            return self.correct(ocr_text=ocr_text, context=context)

    module = PlateCorrector()
    return module, lm


# STUB TRAINING DATA

STUB_TRAINING_DATA = [
    {"ocr_text": "ABC 1234", "corrected_plate": "ABC 1234", "context": ""},
    {"ocr_text": "AB0 I234", "corrected_plate": "ABO 1234", "context": ""},
    {"ocr_text": "XY2 5S8B", "corrected_plate": "XYZ 5S8B", "context": ""},
    {"ocr_text": "L0L OIO1", "corrected_plate": "LOL 0101", "context": ""},
    {"ocr_text": "MNO P4S6", "corrected_plate": "MNO P456", "context": ""},
    {"ocr_text": "8AZ I23S", "corrected_plate": "BAZ 1235", "context": ""},
    {"ocr_text": "QR5 TUV7", "corrected_plate": "QRS TUV7", "context": ""},
    {"ocr_text": "DE1 G4S6", "corrected_plate": "DEF G456", "context": ""},
]


def _plate_metric(example, prediction, trace=None) -> float:
    """
    Metric for evaluating plate correction accuracy.

    Returns 1.0 if the corrected plate matches the gold standard,
    0.0 otherwise. Can be extended with partial matching.
    """
    predicted = prediction.corrected_plate.strip().upper()
    expected = example.corrected_plate.strip().upper()

    if predicted == expected:
        return 1.0

    # Partial credit for close matches
    matches = sum(a == b for a, b in zip(predicted, expected))
    max_len = max(len(predicted), len(expected), 1)
    return matches / max_len


# OPTIMIZER

class PlateOptimizer:
    """
    DSPy-based optimizer for plate text correction prompts.

    Wraps the DSPy compilation process:
    1. Define the plate correction module
    2. Provide training data + metric
    3. Compile with BootstrapFewShot
    4. Use the optimized module for inference
    """

    def __init__(self, config: DSPyConfig):
        self._config = config
        self._module = None
        self._compiled_module = None
        self._lm = None
        self._ready = False

    def initialize(self) -> bool:
        """Initialize the DSPy module."""
        if not self._config.enabled:
            logger.info("DSPy disabled in config")
            return False

        self._module, self._lm = _create_plate_correction_module(self._config)
        if self._module is None:
            return False

        self._ready = True
        logger.info("PlateOptimizer initialized")
        return True

    def optimize(
        self,
        training_data: Optional[list[dict[str, str]]] = None,
    ) -> bool:
        """
        Run DSPy optimization on the plate correction module.

        Args:
            training_data: List of dicts with 'ocr_text', 'corrected_plate',
                          and optional 'context'. Uses stub data if None.

        Returns:
            True if optimization succeeded.
        """
        if not self._ready or self._module is None:
            logger.warning("Cannot optimize — module not initialized")
            return False

        try:
            import dspy

            data = training_data or STUB_TRAINING_DATA

            # Convert to DSPy examples
            examples = [
                dspy.Example(
                    ocr_text=d["ocr_text"],
                    context=d.get("context", ""),
                    corrected_plate=d["corrected_plate"],
                ).with_inputs("ocr_text", "context")
                for d in data
            ]

            # Split train/dev
            split = max(1, len(examples) * 3 // 4)
            train_set = examples[:split]
            dev_set = examples[split:] or examples[:2]

            # Compile with BootstrapFewShot
            optimizer = dspy.BootstrapFewShot(
                metric=_plate_metric,
                max_bootstrapped_demos=self._config.max_bootstrapped_demos,
                max_labeled_demos=self._config.max_labeled_demos,
            )

            self._compiled_module = optimizer.compile(
                self._module,
                trainset=train_set,
            )

            logger.info("DSPy optimization complete")
            return True

        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
            return False

    def correct(self, ocr_text: str, context: str = "") -> dict[str, Any]:
        """
        Correct plate text using the (optionally compiled) module.

        Args:
            ocr_text: Raw OCR text.
            context: Additional context.

        Returns:
            Dict with corrected_plate and metadata.
        """
        module = self._compiled_module or self._module

        if module is None:
            return {
                "corrected_plate": ocr_text,
                "optimized": False,
                "error": "DSPy not initialized",
            }

        try:
            result = module(ocr_text=ocr_text, context=context)
            return {
                "corrected_plate": result.corrected_plate,
                "optimized": self._compiled_module is not None,
                "reasoning": getattr(result, "rationale", ""),
            }
        except Exception as e:
            logger.error(f"DSPy correction failed: {e}")
            return {
                "corrected_plate": ocr_text,
                "optimized": False,
                "error": str(e),
            }

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_optimized(self) -> bool:
        return self._compiled_module is not None
