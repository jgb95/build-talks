"""
Segment data model, preparation, and timeline utilities.

A Segment describes one clip in the final video timeline. This module owns:

  BLACK           -- sentinel Path used as source for a pure-black clip
  AudioSource     -- type alias for the audio field of a Segment
  Segment         -- dataclass describing one clip in the final timeline
  prepare_segment -- normalise a source (image/video/black) into a cached clip
  seg_duration    -- duration of a prepared (or raw) segment in seconds
  talk_offset     -- seconds into the final video where the talk audio begins
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from build_talks.config import (
    ACODEC,
    FADE_DURATION,
    FPS,
    HEIGHT,
    SAMPLE_RATE,
    WIDTH,
)
from build_talks.ffmpeg import (
    normalize_video_filter,
    probe_duration_us,
    run,
    ts_to_us,
    vcodec_flags,
)

log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
#  Segment model
# ════════════════════════════════════════════════════════════

# Sentinel Path: use as segment.source to request a pure-black video clip.
BLACK: Path = Path("__black__")

# Image extensions recognised as still images (not video).
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

AudioSource = Path | Literal["source", "silence"]


@dataclass
class Segment:
    """
    One clip in the final video timeline.

    source      -- path to an image, a video file, or the sentinel BLACK constant
    audio       -- "silence"  : generate silent audio for this segment's duration
                   "source"   : use the video file's own audio track
                   Path       : mix in this external audio file
    pad_start   -- seconds to hold/clone the first frame (or full duration for images)
    pad_end     -- seconds to hold/clone the last frame
    trim_start  -- HH:MM:SS timestamp to seek to in the source (raw segments only)
    trim_end    -- HH:MM:SS timestamp to stop at in the source (raw segments only)
    source_chunks -- optional ordered chunk files that should be treated as a
                     virtual concatenated source for raw segments
    raw         -- if True, skip pre-rendering; source is normalised inside the final
                   filtergraph (used for the talk recording to avoid a double-encode)
    """
    source: Path
    audio: AudioSource = "silence"
    pad_start: float = 0.0
    pad_end: float = 0.0
    trim_start: str | None = None
    trim_end: str | None = None
    source_chunks: list[Path] | None = None
    raw: bool = False


# ════════════════════════════════════════════════════════════
#  Segment preparation
# ════════════════════════════════════════════════════════════

def _cache_name(segment: Segment) -> str:
    """Derive a stable cache filename from the segment's source and padding params."""
    if segment.source == BLACK:
        return f"black_s{segment.pad_start}_e{segment.pad_end}.mp4"
    stem = segment.source.stem
    return f"{stem}_s{segment.pad_start}_e{segment.pad_end}.mp4"


def prepare_segment(segment: Segment, cache_dir: Path, vcodec: str) -> Path:
    """
    Normalize a segment source into a pipeline-ready video clip and cache it.

    - BLACK sentinel  : generates a pure black video of pad_start duration.
    - Image file      : loops the image for pad_start seconds (pad_end is ignored
                        since the whole clip is a still frame).
    - Video file      : re-encodes with tpad to clone first/last frames for fade
                        headroom and adds silent audio.

    All outputs are normalised to the canonical format (WIDTH×HEIGHT, FPS, ACODEC).
    Returns the path to the prepared (cached) file.
    Skips encoding if the cached file already exists.
    """
    if segment.raw:
        # Raw segments are normalised inside the final filtergraph; nothing to do here.
        return segment.source

    output = cache_dir / _cache_name(segment)

    if output.exists():
        log.debug("[prepare] cache hit: %s", output.name)
        return output

    norm = normalize_video_filter()
    silence = f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}"

    # ---- Black segment ----
    if segment.source == BLACK:
        duration = segment.pad_start or segment.pad_end
        log.info("[prepare] black %.2fs", duration)
        t0 = time.monotonic()
        run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-t", str(duration),
                "-i", f"color=black:size={WIDTH}x{HEIGHT}:rate={FPS}",
                "-f", "lavfi", "-t", str(duration),
                "-i", silence,
                *vcodec_flags(vcodec),
                "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
                "-shortest",
                str(output),
            ],
            label="black",
            duration_us=int(duration * 1_000_000),
        )
        log.info("[prepare] black done (%.1fs)", time.monotonic() - t0)
        return output

    # ---- Image segment ----
    if segment.source.suffix.lower() in _IMAGE_SUFFIXES:
        duration = segment.pad_start
        log.info("[prepare] image %s → %.2fs", segment.source.name, duration)
        t0 = time.monotonic()
        run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-t", str(duration), "-i", str(segment.source),
                "-f", "lavfi", "-t", str(duration),
                "-i", silence,
                "-vf", norm,
                *vcodec_flags(vcodec),
                "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
                "-shortest",
                str(output),
            ],
            label=segment.source.stem,
            duration_us=int(duration * 1_000_000),
        )
        log.info("[prepare] image done (%.1fs)", time.monotonic() - t0)
        return output

    # ---- Video segment ----
    log.info("[prepare] video %s (pad_start=%.2f pad_end=%.2f)", segment.source.name, segment.pad_start, segment.pad_end)
    t0 = time.monotonic()

    vfilter = (
        f"{norm},"
        f"tpad=start_mode=clone:start_duration={segment.pad_start}"
        f":stop_mode=clone:stop_duration={segment.pad_end}"
    )

    src_dur_us = probe_duration_us(segment.source)
    output_dur_us = int(
        (src_dur_us / 1_000_000 + segment.pad_start + segment.pad_end) * 1_000_000
    )

    run(
        [
            "ffmpeg", "-y",
            "-i", str(segment.source),
            "-f", "lavfi",
            "-i", silence,
            "-vf", vfilter,
            *vcodec_flags(vcodec),
            "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            str(output),
        ],
        label=segment.source.stem,
        duration_us=output_dur_us,
    )
    log.info("[prepare] video done (%.1fs)", time.monotonic() - t0)
    return output


# ════════════════════════════════════════════════════════════
#  Timeline utilities
# ════════════════════════════════════════════════════════════

def seg_duration(segment: Segment, prepared: Path) -> float:
    """Return the duration of a prepared (or raw) segment in seconds."""
    if segment.raw:
        # Raw talk segment: duration comes from the trim window.
        return (ts_to_us(segment.trim_end) - ts_to_us(segment.trim_start)) / 1_000_000
    return probe_duration_us(prepared) / 1_000_000


def talk_offset(recipe: list[Segment], cache_dir: Path, vcodec: str) -> float:
    """
    Compute how many seconds into the final video the talk audio begins.

    Accounts for all preceding segment durations minus xfade overlaps.
    This is a pure calculation — it prepares (or retrieves from cache) each
    segment only to read its duration, without building or running any ffmpeg command.
    """
    F = FADE_DURATION
    pairs = [(seg, prepare_segment(seg, cache_dir, vcodec)) for seg in recipe]
    durations = [seg_duration(seg, prepared) for seg, prepared in pairs]
    raw_idx = next((i for i, seg in enumerate(recipe) if seg.raw), None)
    if raw_idx is None:
        return 0.0
    return sum(durations[:raw_idx]) - raw_idx * F


def raw_offsets(recipe: list[Segment], cache_dir: Path, vcodec: str) -> list[float]:
    """
    Return timeline offsets (seconds) for every raw segment in recipe order.

    This is used for multi-talk timelines (e.g. livestream playlists) where each
    talk's transcript should be shifted to its start position in the final video.
    """
    F = FADE_DURATION
    pairs = [(seg, prepare_segment(seg, cache_dir, vcodec)) for seg in recipe]
    durations = [seg_duration(seg, prepared) for seg, prepared in pairs]

    offsets: list[float] = []
    for i, seg in enumerate(recipe):
        if seg.raw:
            offsets.append(sum(durations[:i]) - i * F)
    return offsets
