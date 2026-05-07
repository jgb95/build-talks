"""
ffmpeg subprocess helpers.

All direct ffmpeg/ffprobe invocations live here so the rest of the
codebase never calls subprocess directly.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from build_talks.config import (
    ACODEC,
    FPS,
    HEIGHT,
    SAMPLE_RATE,
    VCODEC_HW,
    VCODEC_SW,
    WIDTH,
)

log = logging.getLogger(__name__)

# Set to True when --verbose is passed so ffmpeg stderr streams live.
verbose: bool = False

# Seconds between progress log lines during a long render.
PROGRESS_INTERVAL = 30


def probe_duration_us(path: Path) -> int:
    """
    Return the duration of a media file in microseconds.

    Raises RuntimeError if the duration cannot be determined.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return int(float(result.stdout.strip()) * 1_000_000)
    except (ValueError, AttributeError):
        raise RuntimeError(f"Could not determine duration of {path}")


def normalize_video_filter() -> str:
    """Return the scale+pad+fps filter chain that produces the canonical output format."""
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},"
        f"settb=1/{FPS}"
    )


def vcodec_flags(vcodec: str) -> list[str]:
    """Return the encoder flags for the chosen video codec."""
    if vcodec == VCODEC_HW:
        # VideoToolbox quality-based VBR (1–100); avoids the very high default bitrate.
        return ["-c:v", VCODEC_HW, "-q:v", "65"]
    # libx264 constant-rate-factor: allocates bits where needed.
    return ["-c:v", VCODEC_SW, "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium"]


def run(cmd: list[str], label: str = "", duration_us: int | None = None) -> None:
    """
    Run an ffmpeg command, optionally streaming progress to the log.

    Verbose mode: ffmpeg stderr streams live (full stats).
    Normal mode:  a progress percentage is logged every PROGRESS_INTERVAL
                  seconds when duration_us is known; otherwise silent.
    """
    log.debug("Running: %s", " ".join(cmd))

    if verbose:
        result = subprocess.run(cmd, stderr=None, stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")
        return

    if duration_us and duration_us > 0:
        # Inject -progress pipe:1 so ffmpeg writes structured progress to stdout.
        prog_cmd: list[str] = []
        for arg in cmd:
            if arg == "-y":
                prog_cmd += ["-y", "-progress", "pipe:1", "-nostats"]
            else:
                prog_cmd.append(arg)

        proc = subprocess.Popen(
            prog_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        stderr_lines: list[str] = []
        t_start = time.monotonic()
        t_last_log = t_start

        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        elapsed_us = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    now = time.monotonic()
                    if now - t_last_log >= PROGRESS_INTERVAL:
                        pct = min(elapsed_us / duration_us * 100, 100.0)
                        wall = int(now - t_start)
                        log.info("[rendering] %s — %d%% (%ds)", label, int(pct), wall)
                        t_last_log = now
        finally:
            if proc.stderr:
                stderr_lines = proc.stderr.read().splitlines()
            proc.wait()

        if proc.returncode != 0:
            for line in stderr_lines:
                log.error("[rendering] ffmpeg: %s", line)
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
        return

    # No duration estimate — run silently.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("[rendering] %s — ffmpeg error: %s", label, result.stderr.strip())
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")


def anullsrc(duration: float) -> str:
    """Return an ffmpeg anullsrc filter expression trimmed to *duration* seconds."""
    return (
        f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
        f"atrim=end={duration},asetpts=PTS-STARTPTS"
    )
