"""
Filtergraph construction and final video assembly.

This module takes a prepared list of Segments and produces the finished video:

  build_filtergraph -- generate the complete ffmpeg command from a segment list
  assemble          -- prepare segments, build the filtergraph, and run the encode
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from build_talks.config import (
    ACODEC,
    FADE_DURATION,
    FPS,
    SAMPLE_RATE,
)
from build_talks.ffmpeg import (
    duration_str,
    normalize_video_filter,
    run,
    vcodec_flags,
)
from build_talks.segment import Segment, prepare_segment, seg_duration

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
#  Filtergraph builder
# ════════════════════════════════════════════════════════════

def build_filtergraph(
    pairs: list[tuple[Segment, Path]],
    vcodec: str,
    output: Path,
) -> tuple[list[str], int]:
    """
    Build the full ffmpeg command for assembling the final video.

    pairs   -- list of (Segment, prepared_path) in timeline order
    vcodec  -- encoder choice (passed through to vcodec_flags)
    output  -- destination file for the assembled video

    Returns:
        cmd          -- complete ffmpeg command as a list of strings
        total_dur_us -- total output duration in microseconds (for progress tracking)
    """
    F = FADE_DURATION
    n = len(pairs)

    # --- Build input list and collect per-segment durations ---
    inputs: list[str] = []
    durations: list[float] = []

    for seg, prepared in pairs:
        if seg.raw:
            # Seek at input level; the segment is trimmed to exactly trim_start→trim_end.
            inputs += ["-ss", seg.trim_start, "-t", duration_str(seg.trim_start, seg.trim_end), "-i", str(seg.source)]
        else:
            inputs += ["-i", str(prepared)]
        durations.append(seg_duration(seg, prepared))

    # --- Compute xfade offsets ---
    # The offset for xfade N is the time at which it fires: sum of solo-play durations
    # of all preceding clips, where each clip that has already participated in an xfade
    # contributes (its duration - F) rather than its full duration.
    #
    # Offset formula (0-indexed):
    #   xf[0] = durations[0] - F
    #   xf[i] = xf[i-1] + (durations[i] - F)   for i >= 1
    xf_offsets: list[float] = []
    running = 0.0
    for dur in durations[:-1]:
        running += dur - F
        xf_offsets.append(running)

    # Total output duration = sum of all durations - (n-1) * F
    total_s = sum(durations) - (n - 1) * F
    total_us = int(total_s * 1_000_000)

    # --- Video filtergraph ---
    norm = normalize_video_filter()
    vf_parts: list[str] = []

    # Label each input's video stream; apply normalize filter to the raw segment.
    for i, (seg, _) in enumerate(pairs):
        if seg.raw:
            vf_parts.append(f"[{i}:v]{norm}[v{i}]")
        else:
            vf_parts.append(f"[{i}:v]settb=1/{FPS}[v{i}]")

    # Chain xfades pairwise.
    prev = "v0"
    for i, offset in enumerate(xf_offsets):
        next_label = f"vx{i}"
        vf_parts.append(
            f"[{prev}][v{i + 1}]xfade=transition=fade:duration={F}:offset={offset:.6f}[{next_label}]"
        )
        prev = next_label

    vf_parts.append(f"[{prev}]null[vout]")

    # --- Audio filtergraph ---
    # Build one audio node per segment, then chain them with acrossfade so this
    # works for timelines with multiple raw talk segments (e.g. livestreams).
    af_parts: list[str] = []
    audio_inputs: list[str] = []  # extra -i args appended after video inputs
    audio_base_idx = n  # first external-audio input index

    audio_labels: list[str] = []
    for i, (seg, _) in enumerate(pairs):
        label = f"a{i}"
        dur = durations[i]
        if seg.audio == "silence":
            af_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
                f"atrim=end={dur:.6f},asetpts=PTS-STARTPTS[{label}]"
            )
        elif seg.audio == "source":
            af_parts.append(f"[{i}:a]atrim=end={dur:.6f},asetpts=PTS-STARTPTS[{label}]")
        else:
            ext_idx = audio_base_idx + (len(audio_inputs) // 2)
            audio_inputs += ["-i", str(seg.audio)]
            af_parts.append(
                f"[{ext_idx}:a]atrim=end={dur:.6f},asetpts=PTS-STARTPTS[{label}]"
            )
        audio_labels.append(label)

    if len(audio_labels) == 1:
        af_parts.append(f"[{audio_labels[0]}]anull[aout]")
    else:
        prev_a = audio_labels[0]
        for i in range(1, len(audio_labels)):
            next_a = f"acx{i}"
            af_parts.append(
                f"[{prev_a}][{audio_labels[i]}]acrossfade=d={F}:c1=tri:c2=tri[{next_a}]"
            )
            prev_a = next_a
        af_parts.append(f"[{prev_a}]anull[aout]")

    # --- Assemble the full command ---
    filter_complex = ";".join(vf_parts) + ";" + ";".join(af_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + audio_inputs
        + ["-filter_complex", filter_complex]
        + ["-map", "[vout]", "-map", "[aout]"]
        + vcodec_flags(vcodec)
        + ["-c:a", ACODEC, "-ar", str(SAMPLE_RATE)]
        + [str(output)]
    )

    return cmd, total_us


# ════════════════════════════════════════════════════════════
#  Top-level assembler
# ════════════════════════════════════════════════════════════

def assemble(recipe: list[Segment], output: Path, vcodec: str, cache: Path) -> None:
    """
    Prepare all non-raw segments, build the filtergraph, and render the final video.

    Steps:
      1. For each segment, call prepare_segment() to produce (or retrieve from cache)
         a normalised clip.  Raw segments are passed through as-is.
      2. Call build_filtergraph() to generate the full ffmpeg command.
      3. Run the command via run().
    """
    label = output.stem

    if output.exists():
        log.debug("[rendering] %s — already exists, skipping", label)
        return

    pairs = [(seg, prepare_segment(seg, cache, vcodec)) for seg in recipe]
    cmd, total_us = build_filtergraph(pairs, vcodec, output)

    log.info("[rendering] %s — %ds video", label, total_us // 1_000_000)
    t0 = time.monotonic()
    run(cmd, label=label, duration_us=total_us)
    log.info("[rendering] %s — done (%.1fs)", label, time.monotonic() - t0)
