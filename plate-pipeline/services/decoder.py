"""
FFmpeg Decoder —  video decoder for the pipeline.

FFmpeg is the ONLY allowed decoder. All video ingestion go through
this module. Supports RTSP, RTMP, and local file sources.

Responsibilities:
  - RTSP / RTMP stream decoding
  - Frame extraction
  - Keyframe filtering (P-frame selection)
  - FPS reduction pre-sampling
  - Outputting normalized frame images
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

from services.config import DecoderConfig, VideoIngestionConfig

logger = logging.getLogger(__name__)



# DATA MODELS

@dataclass
class FrameData:
    """Represents a single extracted frame with metadata."""

    frame_id: str
    source: str
    frame_bytes: bytes
    frame_index: int
    timestamp_ms: float
    width: int = 0
    height: int = 0
    content_hash: str = ""
    extraction_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.frame_id:
            self.frame_id = str(uuid.uuid4())
        if not self.content_hash and self.frame_bytes and len(self.frame_bytes) > 0:
            self.content_hash = hashlib.blake2b(
                self.frame_bytes, digest_size=16
            ).hexdigest()


# FFmpeg DECODER

class FFmpegDecoder:
    """
    FFmpeg-based video decoder.

    This is the ONLY allowed decoder in the system. All video sources
    (file, RTSP, RTMP) are processed through FFmpeg subprocess calls.

    Uses the mandatory base command pattern:
        ffmpeg -i <source> -vf "select='eq(pict_type,P)'" -vsync vfr output%04d.jpg
    """

    def __init__(
        self,
        decoder_config: DecoderConfig,
        ingestion_config: VideoIngestionConfig,
    ):
        self._config = decoder_config
        self._ingestion_config = ingestion_config
        self._ffmpeg_bin = decoder_config.ffmpeg.binary_path
        self._validate_ffmpeg()

    def _validate_ffmpeg(self) -> None:
        """Validate that FFmpeg binary is available."""
        if not shutil.which(self._ffmpeg_bin):
            raise RuntimeError(
                f"FFmpeg binary not found at '{self._ffmpeg_bin}'. "
                f"FFmpeg is MANDATORY for this pipeline. Install it or "
                f"update decoder.ffmpeg.binary_path in config."
            )

        # Verify version
        try:
            result = subprocess.run(
                [self._ffmpeg_bin, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            logger.info(f"FFmpeg validated: {version_line}")
        except Exception as e:
            raise RuntimeError(f"FFmpeg validation failed: {e}")

    def probe_video_metadata(self, source: str) -> dict:
        """
        Probe a video source using ffprobe (if available) and return
        a small metadata dict with width, height, fps and nb_frames.

        Returns empty dict if probing is not available or fails.
        """
        try:
            import json
            import subprocess

            # try to find ffprobe next to ffmpeg or on PATH
            ffprobe_bin = shutil.which("ffprobe")
            if not ffprobe_bin:
                # guess alongside configured ffmpeg
                try:
                    ffprobe_guess = os.path.join(os.path.dirname(self._ffmpeg_bin), "ffprobe")
                    if os.path.exists(ffprobe_guess):
                        ffprobe_bin = ffprobe_guess
                except Exception:
                    ffprobe_bin = None

            if not ffprobe_bin:
                return {}

            cmd = [
                ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                source,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0 or not result.stdout:
                return {}

            info = json.loads(result.stdout)
            streams = info.get("streams", [])
            if not streams:
                return {}

            s = streams[0]
            width = int(s.get("width") or 0)
            height = int(s.get("height") or 0)

            fps = 0.0
            fps_str = s.get("r_frame_rate") or s.get("avg_frame_rate")
            if fps_str and fps_str != "0/0":
                try:
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = float(num) / float(den) if float(den) != 0 else float(num)
                    else:
                        fps = float(fps_str)
                except Exception:
                    fps = 0.0

            nb_frames = s.get("nb_frames")
            if nb_frames is None:
                # try duration-based fallback
                duration = s.get("duration")
                try:
                    if duration and fps > 0:
                        total_frames = int(round(float(duration) * fps))
                    else:
                        total_frames = 0
                except Exception:
                    total_frames = 0
            else:
                try:
                    total_frames = int(nb_frames)
                except Exception:
                    total_frames = 0

            return {"width": width, "height": height, "fps": fps, "nb_frames": total_frames}

        except Exception:
            return {}

    def _build_command(
        self,
        source: str,
        output_dir: str,
        output_pattern: str = "frame_%06d",
    ) -> list[str]:
        """
        Build the FFmpeg command from config.

        Required base command:
            ffmpeg -i <source>
                -vf "select='eq(pict_type,P)'"
                -vsync vfr
                output%04d.jpg
        """
        ff = self._config.ffmpeg
        cmd = [self._ffmpeg_bin]

        # Hardware acceleration
        if self._config.hardware_acceleration:
            cmd.extend(["-hwaccel", "auto"])

        # Input source
        if source.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        cmd.extend(["-i", source])

        # Video filter — P-frame selection (MANDATORY)
        vf_filter = f"select='{ff.frame_filter}'"
        cmd.extend(["-vf", vf_filter])

        # Variable frame rate (MANDATORY)
        cmd.extend(["-vsync", ff.vsync])

        # Output quality
        cmd.extend(["-q:v", str(ff.output_quality)])

        # Output pattern
        ext = ff.output_format
        output_path = os.path.join(output_dir, f"{output_pattern}.{ext}")
        cmd.append(output_path)

        # Overwrite without asking
        cmd.insert(1, "-y")

        return cmd

    def extract_frames_sync(
        self,
        source: str,
        output_dir: Optional[str] = None,
    ) -> list[FrameData]:
        """
        Extract frames synchronously from a video source.

        Args:
            source: Video file path, RTSP URL, or RTMP URL.
            output_dir: Directory to write frames. If None, uses a temp dir.

        Returns:
            List of FrameData objects for each extracted frame.
        """
        use_temp = output_dir is None
        if use_temp:
            output_dir = tempfile.mkdtemp(prefix="plate_frames_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        cmd = self._build_command(source, output_dir)
        logger.info(f"FFmpeg command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                # FFmpeg often returns non-zero even on partial success
                # Check if any frames were actually produced
                if not any(Path(output_dir).iterdir()):
                    raise RuntimeError(
                        f"FFmpeg failed with return code {result.returncode}: "
                        f"{result.stderr[:500]}"
                    )

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"FFmpeg timed out after {self._config.timeout_seconds}s "
                f"processing source: {source}"
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # Collect extracted frames
        frames: list[FrameData] = []
        frame_files = sorted(Path(output_dir).glob(f"frame_*.{self._config.ffmpeg.output_format}"))

        for idx, frame_path in enumerate(frame_files):
            frame_bytes = frame_path.read_bytes()
            frames.append(
                FrameData(
                    frame_id=str(uuid.uuid4()),
                    source=source,
                    frame_bytes=frame_bytes,
                    frame_index=idx,
                    timestamp_ms=0.0,  # Approximated from index
                    extraction_time_ms=elapsed_ms / max(len(frame_files), 1),
                    metadata={
                        "file_path": str(frame_path),
                        "file_size": len(frame_bytes),
                    },
                )
            )

        logger.info(
            f"Extracted {len(frames)} frames from '{source}' "
            f"in {elapsed_ms:.1f}ms"
        )

        return frames

    async def extract_frames(
        self,
        source: str,
        output_dir: Optional[str] = None,
    ) -> AsyncGenerator[FrameData, None]:
        """
        Extract frames asynchronously from a video source.

        Runs FFmpeg as an async subprocess and yields frames as they
        are produced.

        Args:
            source: Video file path, RTSP URL, or RTMP URL.
            output_dir: Directory to write frames.

        Yields:
            FrameData objects for each extracted frame.
        """
        use_temp = output_dir is None
        if use_temp:
            output_dir = tempfile.mkdtemp(prefix="plate_frames_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        cmd = self._build_command(source, output_dir)
        logger.info(f"FFmpeg async command: {' '.join(cmd)}")

        start_time = time.time()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        ext = self._config.ffmpeg.output_format
        yielded_files: set[str] = set()
        frame_index = 0
        stderr_chunks: list[bytes] = []

        async def _drain_stderr() -> None:
            if process.stderr is None:
                return

            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(_drain_stderr())

        try:
            while True:
                frame_files = sorted(Path(output_dir).glob(f"frame_*.{ext}"))

                for frame_path in frame_files:
                    if frame_path.name in yielded_files:
                        continue

                    frame_bytes = frame_path.read_bytes()
                    yielded_files.add(frame_path.name)
                    yield FrameData(
                        frame_id=str(uuid.uuid4()),
                        source=source,
                        frame_bytes=frame_bytes,
                        frame_index=frame_index,
                        timestamp_ms=0.0,
                        extraction_time_ms=(time.time() - start_time) * 1000,
                        metadata={
                            "file_path": str(frame_path),
                            "file_size": len(frame_bytes),
                        },
                    )
                    frame_index += 1

                if process.returncode is not None:
                    break

                await asyncio.sleep(0.25)

            await asyncio.wait_for(stderr_task, timeout=1)

            if process.returncode != 0:
                stderr = b"".join(stderr_chunks).decode(errors="replace")
                logger.warning(f"FFmpeg stderr: {stderr[:500]}")

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Async extracted {len(yielded_files)} frames from '{source}' "
                f"in {elapsed_ms:.1f}ms"
            )
        finally:
            if not stderr_task.done():
                stderr_task.cancel()

    def extract_frames_piped(
        self,
        source: str,
    ) -> Generator[FrameData, None, None]:
        """
        Extract frames via stdout pipe (no disk I/O).

        Pipes raw JPEG frames from FFmpeg stdout. More efficient for
        high-throughput scenarios but requires parsing JPEG boundaries.

        Args:
            source: Video file path or stream URL.

        Yields:
            FrameData objects with frame bytes from pipe.
        """
        ff = self._config.ffmpeg
        cmd = [self._ffmpeg_bin]

        if self._config.hardware_acceleration:
            cmd.extend(["-hwaccel", "auto"])

        if source.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend(["-i", source])
        cmd.extend(["-vf", f"select='{ff.frame_filter}'"])
        cmd.extend(["-vsync", ff.vsync])
        cmd.extend(["-f", "image2pipe"])
        cmd.extend(["-vcodec", "mjpeg"])
        cmd.extend(["-q:v", str(ff.output_quality)])
        cmd.append("pipe:1")

        logger.info(f"FFmpeg piped command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_index = 0
        buffer = b""
        SOI = b"\xff\xd8"  # JPEG Start of Image marker
        EOI = b"\xff\xd9"  # JPEG End of Image marker

        try:
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break

                buffer += chunk

                while True:
                    soi_pos = buffer.find(SOI)
                    if soi_pos == -1:
                        buffer = b""
                        break

                    eoi_pos = buffer.find(EOI, soi_pos + 2)
                    if eoi_pos == -1:
                        # Incomplete frame, wait for more data
                        buffer = buffer[soi_pos:]
                        break

                    # Extract complete JPEG
                    frame_bytes = buffer[soi_pos:eoi_pos + 2]
                    buffer = buffer[eoi_pos + 2:]

                    yield FrameData(
                        frame_id=str(uuid.uuid4()),
                        source=source,
                        frame_bytes=frame_bytes,
                        frame_index=frame_index,
                        timestamp_ms=0.0,
                        metadata={"pipe_mode": True},
                    )
                    frame_index += 1

        finally:
            process.terminate()
            process.wait(timeout=5)
            logger.info(
                f"Piped extraction complete: {frame_index} frames from '{source}'"
            )
