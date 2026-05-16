"""
YOLO Detector — License plate detection using Ultralytics YOLO.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from services.config import PipelineConfig
from services.models.base import DetectionModel

logger = logging.getLogger(__name__)


class YOLODetector(DetectionModel):
    """YOLO-based license plate detector using Ultralytics."""

    def __init__(self, config: PipelineConfig):
        self._config = config.model_registry.yolo
        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the YOLO model from the configured path."""
        model_path = self._config.path

        if not os.path.exists(model_path):
            logger.warning(
                f"YOLO model not found at '{model_path}'. "
                f"Plate detection will not work until a valid model is provided. "
                f"Update model_registry.yolo.path in config.yaml."
            )
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._loaded = True
            logger.info(
                f"YOLO model loaded from '{model_path}' "
                f"(device={self._config.device.value})"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._loaded = False

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO inference on an image."""
        if not self._loaded or self._model is None:
            logger.warning("YOLO model not loaded — skipping detection")
            return []

        results = self._model(
            image,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.imgsz,
            device=self._config.device.value,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": "license_plate",
                })
                
        if self._config.selection_policy == "best" and detections:
            detections = [max(detections, key=lambda d: d["confidence"])]

        logger.debug(f"YOLO detected {len(detections)} plates")
        return detections

    def is_loaded(self) -> bool:
        return self._loaded
