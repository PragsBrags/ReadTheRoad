"""
Tests for FrameSampler strategies.

Tests fixed FPS, motion-based, hybrid, and Nyquist sampling.
"""

import io


from PIL import Image

from services.config import FrameSamplerConfig, SamplerStrategy
from services.decoder import FrameData
from services.sampler import (
    FixedFPSSampler,
    FrameSampler,
    HybridSampler,
    MotionSharpnessSampler,
    MotionSampler,
    NyquistSampler,
)


# HELPERS

def make_frame(index: int, color: int = 128) -> FrameData:
    """Create a dummy frame with a solid-color JPEG."""
    img = Image.new("RGB", (100, 100), (color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return FrameData(
        frame_id=f"frame_{index}",
        source="test.mp4",
        frame_bytes=buf.getvalue(),
        frame_index=index,
        timestamp_ms=float(index * 33),  # ~30fps
    )


def make_frames(count: int, vary: bool = False) -> list[FrameData]:
    """Create a list of dummy frames."""
    return [
        make_frame(i, color=(i * 30 % 256) if vary else 128)
        for i in range(count)
    ]


# TESTS

class TestFixedFPSSampler:

    def test_samples_at_interval(self):
        """Should sample every Nth frame."""
        sampler = FixedFPSSampler(target_fps=1.0, source_fps=30.0)
        # Interval = 30/1 = 30
        results = [
            sampler.should_sample(make_frame(i), i)
            for i in range(90)
        ]
        sampled_count = sum(results)
        assert sampled_count == 3  # frames 0, 30, 60

    def test_high_target_fps(self):
        """High target FPS should keep most frames."""
        sampler = FixedFPSSampler(target_fps=30.0, source_fps=30.0)
        results = [
            sampler.should_sample(make_frame(i), i)
            for i in range(30)
        ]
        assert sum(results) == 30

    def test_reset(self):
        """Reset should restart the counter."""
        sampler = FixedFPSSampler(target_fps=1.0, source_fps=10.0)
        for i in range(5):
            sampler.should_sample(make_frame(i), i)
        sampler.reset()
        # After reset, first frame should be sampled
        assert sampler.should_sample(make_frame(0), 0) is True


class TestMotionSampler:

    def test_first_frame_always_sampled(self):
        """First frame should always be sampled."""
        sampler = MotionSampler(motion_threshold=0.2)
        frame = make_frame(0)
        assert sampler.should_sample(frame, 0) is True

    def test_identical_frames_skipped(self):
        """Identical frames should be skipped (no motion)."""
        sampler = MotionSampler(motion_threshold=0.2)
        frame1 = make_frame(0, color=128)
        frame2 = make_frame(1, color=128)

        sampler.should_sample(frame1, 0)  # First always True
        result = sampler.should_sample(frame2, 1)
        assert result == False

    def test_different_frames_sampled(self):
        """Very different frames should be sampled."""
        sampler = MotionSampler(motion_threshold=0.2)
        frame1 = make_frame(0, color=0)
        frame2 = make_frame(1, color=255)

        sampler.should_sample(frame1, 0)
        result = sampler.should_sample(frame2, 1)
        assert result == True


class TestNyquistSampler:

    def test_nyquist_rate(self):
        """Should sample at 2x event frequency."""
        # event_freq=1Hz, source=30fps → nyquist=2fps → interval=15
        sampler = NyquistSampler(event_frequency=1.0, source_fps=30.0)
        results = [
            sampler.should_sample(make_frame(i), i)
            for i in range(60)
        ]
        assert sum(results) == 4  # 60/15 = 4


class TestFrameSampler:

    def test_factory_fixed_fps(self):
        """Factory should create FixedFPS strategy."""
        config = FrameSamplerConfig(strategy=SamplerStrategy.FIXED_FPS)
        sampler = FrameSampler(config)
        assert isinstance(sampler.strategy, FixedFPSSampler)

    def test_factory_motion(self):
        """Factory should create Motion strategy."""
        config = FrameSamplerConfig(strategy=SamplerStrategy.MOTION)
        sampler = FrameSampler(config)
        assert isinstance(sampler.strategy, MotionSampler)

    def test_factory_hybrid(self):
        """Factory should create Hybrid strategy."""
        config = FrameSamplerConfig(strategy=SamplerStrategy.HYBRID)
        sampler = FrameSampler(config)
        assert isinstance(sampler.strategy, HybridSampler)

    def test_factory_nyquist(self):
        """Factory should create Nyquist strategy."""
        config = FrameSamplerConfig(strategy=SamplerStrategy.NYQUIST)
        sampler = FrameSampler(config)
        assert isinstance(sampler.strategy, NyquistSampler)

    def test_factory_motion_sharpness(self):
        """Factory should create OpenCV motion+sharpness strategy."""
        config = FrameSamplerConfig(strategy=SamplerStrategy.MOTION_SHARPNESS)
        sampler = FrameSampler(config)
        assert isinstance(sampler.strategy, MotionSharpnessSampler)

    def test_motion_sharpness_flushes_buffered_best_frame(self):
        """The strategy should emit its buffered best frame at video end."""
        sampler = MotionSharpnessSampler(
            motion_area_threshold=1,
            cooldown_frames=1,
            warmup_frames=1,
            downscale_factor=0.5,
        )
        sampled = sampler.sample_frames(make_frames(5))

        assert len(sampled) >= 1
        assert sampled[0].frame_id == "frame_0"

    def test_sample_reduces_frames(self):
        """Sampling should reduce total frame count."""
        config = FrameSamplerConfig(
            strategy=SamplerStrategy.FIXED_FPS,
            fixed_fps=1.0,
        )
        sampler = FrameSampler(config)
        frames = make_frames(30)
        sampled = sampler.sample(frames)
        assert len(sampled) < len(frames)
        assert len(sampled) > 0

    def test_filter_stream_generator(self):
        """filter_stream should work as a generator."""
        config = FrameSamplerConfig(
            strategy=SamplerStrategy.FIXED_FPS,
            fixed_fps=10.0,
        )
        sampler = FrameSampler(config)
        frames = make_frames(30)
        sampled = list(sampler.filter_stream(frames))
        assert len(sampled) > 0
        assert all(isinstance(f, FrameData) for f in sampled)
