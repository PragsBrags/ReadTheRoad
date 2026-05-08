"""
Aggregation Service — Multi-frame result merging and deduplication.

Aggregates detections from multiple frames within a configurable
time window. Uses string similarity for deduplication and
confidence-weighted voting for final plate text.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Optional

from services.config import AggregationConfig

logger = logging.getLogger(__name__)


# STRING SIMILARITY

def string_similarity(a: str, b: str) -> float:
    """
    Compute string similarity ratio between two strings.

    Uses SequenceMatcher for robust similarity scoring.
    Returns value between 0.0 (completely different) and 1.0 (identical).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()


# AGGREGATION RESULT

class AggregatedPlate:
    """Represents an aggregated plate detection across multiple frames."""

    def __init__(self, text: str, confidence: float, bbox: list[int]):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.frame_count = 1
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.all_texts: list[str] = [text]
        self.all_confidences: list[float] = [confidence]
        self.source_frames: list[str] = []

    def merge(
        self, text: str, confidence: float, bbox: list[int],
        frame_id: str = "",
    ) -> None:
        """Merge a new detection into this aggregated plate."""
        self.all_texts.append(text)
        self.all_confidences.append(confidence)
        self.frame_count += 1
        self.last_seen = time.time()
        if frame_id:
            self.source_frames.append(frame_id)

        # Confidence-weighted text selection
        if confidence > self.confidence:
            self.text = text
            self.confidence = confidence
            self.bbox = bbox

    @property
    def average_confidence(self) -> float:
        """Average confidence across all detections."""
        if not self.all_confidences:
            return 0.0
        return sum(self.all_confidences) / len(self.all_confidences)

    @property
    def weighted_text(self) -> str:
        """Get the text with highest confidence-weighted vote."""
        if not self.all_texts:
            return self.text

        # Vote: count text variants weighted by confidence
        votes: dict[str, float] = defaultdict(float)
        for text, conf in zip(self.all_texts, self.all_confidences):
            normalized = text.strip().upper()
            votes[normalized] += conf

        if not votes:
            return self.text

        best_text = max(votes, key=lambda k: votes[k])
        return best_text

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "text": self.weighted_text,
            "confidence": self.average_confidence,
            "best_single_confidence": self.confidence,
            "bbox": self.bbox,
            "frame_count": self.frame_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "all_variants": list(set(self.all_texts)),
            "source_frames": self.source_frames,
        }


# AGGREGATION SERVICE

class AggregationService:
    """
    Multi-frame result aggregation service.

    Merges results from multiple frames within a time window.
    Deduplicates using string similarity threshold from config.
    Uses confidence-weighted voting for final plate text.
    """

    def __init__(self, config: AggregationConfig):
        self._config = config
        self._window_ms = config.window_ms
        self._dedup_threshold = config.dedup_threshold
        self._min_confidence = config.min_confidence

        # Active aggregation windows keyed by job_id
        self._windows: dict[str, list[AggregatedPlate]] = defaultdict(list)
        self._window_start: dict[str, float] = {}

        # Lock for thread-safe access from concurrent async coroutines
        self._lock = asyncio.Lock()

    def add_result(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> Optional[list[dict[str, Any]]]:
        """
        Add an inference result to the aggregation window.

        If the window has expired, returns the finalized aggregated
        results. Otherwise returns None (window still collecting).

        Args:
            job_id: Job/stream identifier.
            result: Inference result from a worker.

        Returns:
            List of aggregated plate dicts if window expired, else None.
        """
        now = time.time()

        # Initialize window if needed
        if job_id not in self._window_start:
            self._window_start[job_id] = now

        # Check if window has expired
        window_elapsed_ms = (now - self._window_start[job_id]) * 1000
        if window_elapsed_ms >= self._window_ms and self._windows[job_id]:
            # Finalize current window and start new one
            finalized = self._finalize_window(job_id)
            self._window_start[job_id] = now
            self._windows[job_id] = []

            # Process current result into new window
            self._merge_result(job_id, result)
            return finalized

        # Add to current window
        self._merge_result(job_id, result)
        return None

    async def add_result_async(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> Optional[list[dict[str, Any]]]:
        """Thread-safe version of add_result for async contexts."""
        async with self._lock:
            return self.add_result(job_id, result)

    def _merge_result(self, job_id: str, result: dict[str, Any]) -> None:
        """Merge a result into the current aggregation window."""
        plates = result.get("plates", [])
        frame_id = result.get("frame_id", "")

        for plate in plates:
            text = plate.get("text") or ""
            confidence = plate.get("confidence", 0.0)
            bbox = plate.get("bbox", [0, 0, 0, 0])

            # Skip low-confidence detections
            if confidence < self._min_confidence:
                continue

            # Try to merge with existing aggregated plate
            merged = False
            for agg in self._windows[job_id]:
                # Compare against weighted_text for stable dedup
                compare_text = agg.weighted_text if agg.all_texts else agg.text
                if text and compare_text:
                    sim = string_similarity(text, compare_text)
                    if sim >= self._dedup_threshold:
                        agg.merge(text, confidence, bbox, frame_id=frame_id)
                        merged = True
                        break

            if not merged:
                agg = AggregatedPlate(text, confidence, bbox)
                if frame_id:
                    agg.source_frames.append(frame_id)
                self._windows[job_id].append(agg)

                # Limit results per window
                if len(self._windows[job_id]) > self._config.max_results_per_window:
                    # Remove lowest confidence
                    self._windows[job_id].sort(
                        key=lambda x: x.average_confidence, reverse=True
                    )
                    self._windows[job_id] = self._windows[job_id][
                        :self._config.max_results_per_window
                    ]

    def _finalize_window(self, job_id: str) -> list[dict[str, Any]]:
        """Finalize an aggregation window and return results."""
        window = self._windows.get(job_id, [])

        results = [agg.to_dict() for agg in window]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(
            f"[{job_id}] Aggregation window finalized: "
            f"{len(results)} unique plates from "
            f"{sum(r['frame_count'] for r in results)} detections"
        )

        return results

    def flush(self, job_id: str) -> list[dict[str, Any]]:
        """Force-flush a window (e.g., on job completion)."""
        results = self._finalize_window(job_id)
        self._windows.pop(job_id, None)
        self._window_start.pop(job_id, None)
        return results

    def flush_all(self) -> dict[str, list[dict[str, Any]]]:
        """Flush all windows."""
        all_results = {}
        for job_id in list(self._windows.keys()):
            all_results[job_id] = self.flush(job_id)
        return all_results

    @property
    def active_windows(self) -> int:
        return len(self._windows)
