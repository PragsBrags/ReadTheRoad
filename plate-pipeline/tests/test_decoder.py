"""
Tests for FFmpegDecoder.

Tests FFmpeg command building, frame extraction, and error handling.
"""

import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.config import (
    DecoderConfig,
    FFmpegSettings,
    VideoIngestionConfig,
)
from services.decoder import FFmpegDecoder, FrameData


# FIXTURES

@pytest.fixture
def decoder_config():
    return DecoderConfig(
        engine="ffmpeg",
        ffmpeg=FFmpegSettings(
            binary_path="ffmpeg",
            frame_filter="eq(pict_type,P)",
            vsync="vfr",
            output_format="jpg",
            output_quality=2,
        ),
        hardware_acceleration=False,
        timeout_seconds=60,
    )


@pytest.fixture
def ingestion_config(tmp_path):
    return VideoIngestionConfig(
        upload_dir=str(tmp_path / "uploads"),
        frame_output_dir=str(tmp_path / "frames"),
    )


# TESTS

class TestFFmpegDecoder:

    def test_engine_must_be_ffmpeg(self):
        """Decoder config rejects non-FFmpeg engines."""
        with pytest.raises(ValueError, match="Only 'ffmpeg' is allowed"):
            DecoderConfig(engine="opencv")

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_validate_ffmpeg_binary(self, mock_run, mock_which, decoder_config, ingestion_config):
        """Decoder validates FFmpeg binary on init."""
        mock_run.return_value = MagicMock(
            stdout="ffmpeg version 6.0", returncode=0
        )
        decoder = FFmpegDecoder(decoder_config, ingestion_config)
        mock_which.assert_called_once_with("ffmpeg")

    @patch("shutil.which", return_value=None)
    def test_missing_ffmpeg_raises(self, mock_which, decoder_config, ingestion_config):
        """Decoder raises if FFmpeg binary not found."""
        with pytest.raises(RuntimeError, match="FFmpeg binary not found"):
            FFmpegDecoder(decoder_config, ingestion_config)

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_build_command_includes_pframe_filter(
        self, mock_run, mock_which, decoder_config, ingestion_config
    ):
        """Built command must include P-frame filter."""
        mock_run.return_value = MagicMock(stdout="ffmpeg version 6.0", returncode=0)
        decoder = FFmpegDecoder(decoder_config, ingestion_config)
        cmd = decoder._build_command("test.mp4", "/tmp/output")

        # Must contain the mandatory P-frame filter
        assert any("eq(pict_type,P)" in arg for arg in cmd)
        # Must contain vsync vfr
        assert "vfr" in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_build_command_rtsp_transport(
        self, mock_run, mock_which, decoder_config, ingestion_config
    ):
        """RTSP sources should include TCP transport flag."""
        mock_run.return_value = MagicMock(stdout="ffmpeg version 6.0", returncode=0)
        decoder = FFmpegDecoder(decoder_config, ingestion_config)
        cmd = decoder._build_command("rtsp://stream.example.com/live", "/tmp/output")

        assert "-rtsp_transport" in cmd
        assert "tcp" in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_build_command_hwaccel(
        self, mock_run, mock_which, ingestion_config
    ):
        """Hardware acceleration flag should be included when enabled."""
        config = DecoderConfig(
            engine="ffmpeg",
            hardware_acceleration=True,
        )
        mock_run.return_value = MagicMock(stdout="ffmpeg version 6.0", returncode=0)
        decoder = FFmpegDecoder(config, ingestion_config)
        cmd = decoder._build_command("test.mp4", "/tmp/output")

        assert "-hwaccel" in cmd
        assert "auto" in cmd

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_extract_frames_yields_while_process_runs(
        self, mock_run, mock_which, decoder_config, ingestion_config, tmp_path, monkeypatch
    ):
        """Live sources should yield frames before FFmpeg exits."""
        mock_run.return_value = MagicMock(stdout="ffmpeg version 6.0", returncode=0)
        decoder = FFmpegDecoder(decoder_config, ingestion_config)

        done_event = asyncio.Event()

        class FakeStderr:
            async def read(self, _size: int) -> bytes:
                await done_event.wait()
                return b""

        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.stderr = FakeStderr()

            async def wait(self):
                await done_event.wait()
                self.returncode = 0
                return 0

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            output_path = Path(cmd[-1])
            output_dir = output_path.parent

            async def write_frames():
                (output_dir / "frame_000001.jpg").write_bytes(b"frame-1")
                await asyncio.sleep(0.05)
                (output_dir / "frame_000002.jpg").write_bytes(b"frame-2")
                await asyncio.sleep(0.05)
                done_event.set()

            asyncio.create_task(write_frames())
            return FakeProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

        async def collect_frames() -> list[FrameData]:
            frames: list[FrameData] = []
            async for frame in decoder.extract_frames("rtsp://example.com/live", output_dir=str(tmp_path / "frames")):
                frames.append(frame)
            return frames

        frames = asyncio.run(collect_frames())

        assert len(frames) == 2
        assert [frame.frame_index for frame in frames] == [0, 1]


class TestFrameData:

    def test_frame_data_auto_hash(self):
        """FrameData should auto-compute content hash."""
        frame = FrameData(
            frame_id="",
            source="test.mp4",
            frame_bytes=b"fake jpeg data",
            frame_index=0,
            timestamp_ms=0.0,
        )
        assert frame.content_hash != ""
        assert frame.frame_id != ""

    def test_frame_data_preserves_id(self):
        """FrameData should preserve explicit frame_id."""
        frame = FrameData(
            frame_id="custom-id",
            source="test.mp4",
            frame_bytes=b"data",
            frame_index=0,
            timestamp_ms=0.0,
        )
        assert frame.frame_id == "custom-id"
