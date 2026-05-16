"""
Frame Sampler — Config-driven frame sampling strategies.

Implements the Strategy pattern for frame selection. The active strategy
is determined entirely by config.yaml.

Strategies:
  - FixedFPS: sample every Nth frame to achieve target FPS
  - Motion: SSIM-based motion detection, skip static frames
  - Hybrid: fixed FPS + motion threshold combined
  - Nyquist: sample at 2x expected event frequency
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image
import cv2

from services.config import FrameSamplerConfig, SamplerStrategy
from services.decoder import FrameData

logger = logging.getLogger(__name__)


# SSIM UTILITY

def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute a simplified SSIM between two grayscale images.

    This is a lightweight approximation. 
    """
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2)
    except ImportError:
        pass

    # Fallback: simplified SSIM
    if img1.shape != img2.shape:
        return 0.0

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def frame_to_grayscale(frame: FrameData) -> np.ndarray:
    """Convert frame bytes to grayscale numpy array."""
    img = Image.open(io.BytesIO(frame.frame_bytes)).convert("L")
    return np.array(img)

#Sharpness estimation for motion detection
def calculate_sharpness(img: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# BASE STRATEGY

class FrameSamplerStrategy(ABC):
    """Abstract base for frame sampling strategies."""

    @abstractmethod
    def should_sample(self, frame: FrameData, index: int) -> bool:
        """
        Determine whether a frame should be included in the sample.

        Args:
            frame: The candidate frame.
            index: Sequential index of this frame in the stream.

        Returns:
            True if the frame should be processed, False to skip.
        """
        ...
    def sample_frames(self, frames: list[FrameData]) -> list[FrameData]:
        self.reset()
        sampled=[]

        for idx, frame in enumerate(frames):
            if self.should_sample(frame, idx):
                sampled.append(frame)
        
        return sampled  

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g., between streams)."""
        ...


# CONCRETE STRATEGIES

class FixedFPSSampler(FrameSamplerStrategy):
    """
    Sample every Nth frame to achieve a target FPS.

    Given the source FPS and target FPS, computes the skip interval.
    If source FPS is unknown, defaults to keeping every frame.
    """

    def __init__(self, target_fps: float, source_fps: float = 30.0):
        self._target_fps = target_fps
        self._source_fps = source_fps
        self._interval = max(1, int(source_fps / target_fps))
        self._count = 0
        logger.info(
            f"FixedFPSSampler: target={target_fps}fps, "
            f"source={source_fps}fps, interval={self._interval}"
        )

    def should_sample(self, frame: FrameData, index: int) -> bool:
        self._count += 1
        return (self._count - 1) % self._interval == 0

    def reset(self) -> None:
        self._count = 0


class MotionSampler(FrameSamplerStrategy):
    """
    Sample frames based on motion detection via SSIM.

    Compares each frame to the previous one. If SSIM is below the
    threshold (meaning frames are different), the frame is sampled.
    """

    def __init__(self, motion_threshold: float = 0.2):
        self._threshold = motion_threshold
        self._prev_gray: Optional[np.ndarray] = None
        logger.info(f"MotionSampler: threshold={motion_threshold}")

    def should_sample(self, frame: FrameData, index: int) -> bool:
        current_gray = frame_to_grayscale(frame)

        if self._prev_gray is None:
            self._prev_gray = current_gray
            return True  # Always sample the first frame

        ssim = compute_ssim_simple(self._prev_gray, current_gray)
        has_motion = ssim < (1.0 - self._threshold)

        if has_motion:
            self._prev_gray = current_gray

        return has_motion

    def reset(self) -> None:
        self._prev_gray = None


class HybridSampler(FrameSamplerStrategy):
    """
    Combines fixed FPS + motion detection.

    First applies FPS-based filtering, then checks for motion.
    This is the default strategy.
    """

    def __init__(
        self,
        target_fps: float = 2.0,
        motion_threshold: float = 0.2,
        source_fps: float = 30.0,
    ):
        self._fps_sampler = FixedFPSSampler(target_fps, source_fps)
        self._motion_sampler = MotionSampler(motion_threshold)
        logger.info(
            f"HybridSampler: fps={target_fps}, motion_threshold={motion_threshold}"
        )

    def should_sample(self, frame: FrameData, index: int) -> bool:
        # First gate: FPS filter
        if not self._fps_sampler.should_sample(frame, index):
            return False

        # Second gate: motion detection
        return self._motion_sampler.should_sample(frame, index)

    def reset(self) -> None:
        self._fps_sampler.reset()
        self._motion_sampler.reset()


class NyquistSampler(FrameSamplerStrategy):
    """
    Sample at 2x the expected event frequency (Nyquist theorem).

    If license plates are expected to appear at `f` events/second,
    sample at `2f` frames/second to avoid aliasing.
    """

    def __init__(
        self,
        event_frequency: float = 1.0,
        source_fps: float = 30.0,
    ):
        nyquist_rate = 2.0 * event_frequency
        self._interval = max(1, int(source_fps / nyquist_rate))
        self._count = 0
        logger.info(
            f"NyquistSampler: event_freq={event_frequency}Hz, "
            f"nyquist_rate={nyquist_rate}fps, interval={self._interval}"
        )

    def should_sample(self, frame: FrameData, index: int) -> bool:
        self._count += 1
        return (self._count - 1) % self._interval == 0

    def reset(self) -> None:
        self._count = 0

class MotionSharpnessSampler(FrameSamplerStrategy):
    def __init__(
            self,
            motion_area_threshold: float = 100,
            cooldown_frames: int = 10,
            warmup_frames: int = 100,
    ):
        self._threshold = motion_area_threshold
        self.cooldown_frames = cooldown_frames
        self.warmup_frames = warmup_frames
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
        self.reset()

    def should_sample(self, frame: FrameData, index: int) -> bool:
        img = np.array(Image.open(io.BytesIO(frame.frame_bytes)).convert("RGB"))

        self._frame_count += 1
        mask = self.object_detector.apply(img)

        if self._frame_count < self.warmup_frames:
            return False

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        active_motion = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self._threshold:
                active_motion = True
                score = calculate_sharpness(img) * area

                if score > self._max_score:
                    self._max_score = score
                    self._best_frame_id = frame.frame_id

        if not active_motion and self._best_frame_id is not None:
            self._frames_since_motion += 1

            if self._frames_since_motion > self.cooldown_frames:
                should_emit = frame.frame_id == self._best_frame_id

                self._best_frame_id = None
                self._max_score = -1
                self._frames_since_motion = 0

                return should_emit

        return frame.frame_id == self._best_frame_id
    
    def reset(self) -> None:
        self._best_frame_id = None
        self._max_score = -1
        self._frames_since_motion = 0
        self._frame_count = 0



# SAMPLER FACTORY

class FrameSampler:
    """
    Frame sampler with pluggable strategies.

    Strategy is selected from config. The sampler filters a stream
    of frames, yielding only those that pass the active strategy.
    """

    _STRATEGY_MAP = {
        SamplerStrategy.FIXED_FPS: lambda cfg: FixedFPSSampler(
            target_fps=cfg.fixed_fps,
        ),
        SamplerStrategy.MOTION: lambda cfg: MotionSampler(
            motion_threshold=cfg.motion_threshold,
        ),
        SamplerStrategy.HYBRID: lambda cfg: HybridSampler(
            target_fps=cfg.fixed_fps,
            motion_threshold=cfg.motion_threshold,
        ),
        SamplerStrategy.NYQUIST: lambda cfg: NyquistSampler(
            event_frequency=cfg.nyquist_event_frequency,
        ),
        SamplerStrategy.MOTION_SHARPNESS: lambda cfg: MotionSharpnessSampler(
            motion_area_threshold=cfg.motion_area_threshold,
            cooldown_frames=cfg.cooldown_frames,
            warmup_frames=cfg.warmup_frames,
        ),
    }

    def __init__(self, config: FrameSamplerConfig):
        self._config = config
        factory = self._STRATEGY_MAP.get(config.strategy)
        if factory is None:
            raise ValueError(
                f"Unknown sampling strategy: {config.strategy}. "
                f"Valid options: {list(self._STRATEGY_MAP.keys())}"
            )
        self._strategy: FrameSamplerStrategy = factory(config)
        logger.info(f"FrameSampler initialized with strategy: {config.strategy.value}")

    @property
    def strategy(self) -> FrameSamplerStrategy:
        return self._strategy

    def sample(self, frames: list[FrameData]) -> list[FrameData]:
        """
        Filter a list of frames using the active strategy.

        Args:
            frames: List of FrameData to filter.

        Returns:
            Filtered list of frames that passed the sampling strategy.
        """
        
        sampled = self._strategy.sample_frames(frames)

        logger.info(
            f"Sampled {len(sampled)}/{len(frames)} frames "
            f"({len(sampled)/max(len(frames),1)*100:.1f}%)"
        )
        return sampled

    def filter_stream(self, frames):
        """
        Generator: filter a stream of frames using the active strategy.

        Args:
            frames: Iterable of FrameData.

        Yields:
            FrameData objects that pass the sampling strategy.
        """
        self._strategy.reset()
        total = 0
        sampled = 0

        for idx, frame in enumerate(frames):
            total += 1
            if self._strategy.should_sample(frame, idx):
                sampled += 1
                yield frame

        logger.info(
            f"Stream sampling complete: {sampled}/{total} frames kept"
        )
