"""
Video rendering functions.

render_title_card   -- image → short video clip with silent audio
normalize_sponsor   -- re-encode sponsor reel with padding and silent audio
render_full_talk    -- assemble the complete final video in one filtergraph pass
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from build_talks.config import (
    ACODEC,
    BLACK_PAD,
    FADE_DURATION,
    FPS,
    HEIGHT,
    SAMPLE_RATE,
    SPONSOR_HOLD,
    TITLE_DURATION,
    WIDTH,
)
from build_talks import ffmpeg as ff

log = logging.getLogger(__name__)


# ---- Timestamp helpers ----

def ts_to_us(ts: str) -> int:
    """Convert HH:MM:SS[.mmm] timestamp to microseconds."""
    h, m, s = ts.split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1_000_000)


def duration_str(start: str, end: str) -> str:
    """
    Return the duration between two HH:MM:SS[.mmm] timestamps as an
    HH:MM:SS.mmm string suitable for ffmpeg's -t option.
    """
    us = ts_to_us(end) - ts_to_us(start)
    total_s = us / 1_000_000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h}:{m:02d}:{s:06.3f}"


# ---- Render functions ----

def render_title_card(image_path: Path, output: Path, vcodec: str, label: str = "") -> None:
    """
    Render a title card image as a video clip with silent audio.

    Duration = TITLE_DURATION + 2 * FADE_DURATION so that xfade filters
    can consume FADE_DURATION seconds on each end while still leaving
    TITLE_DURATION seconds of fully-visible title in between.
    """
    if output.exists():
        log.debug("[rendering] %s — title card already cached", label)
        return

    total = TITLE_DURATION + 2 * FADE_DURATION
    log.info("[rendering] %s — title card", label)
    t0 = time.monotonic()
    ff.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(total), "-i", str(image_path),
            "-f", "lavfi", "-t", str(total),
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
            "-vf", ff.normalize_video_filter(),
            *ff.vcodec_flags(vcodec),
            "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
            "-shortest",
            str(output),
        ],
        label=label,
        duration_us=int(total * 1_000_000),
    )
    log.info("[rendering] %s — title card done (%.1fs)", label, time.monotonic() - t0)


def normalize_sponsor(sponsor_path: Path, output: Path, vcodec: str) -> None:
    """
    Re-encode the sponsor reel with silent audio and frame padding.

    Prepends FADE_DURATION seconds of cloned first frame (for the incoming
    xfade) and appends (SPONSOR_HOLD + FADE_DURATION) seconds of frozen last
    frame (SPONSOR_HOLD seconds visible; FADE_DURATION consumed by the outgoing
    xfade/fade).
    """
    if output.exists():
        log.debug("[rendering] sponsor — already cached")
        return

    log.info("[rendering] sponsor")
    t0 = time.monotonic()

    vfilter = (
        f"{ff.normalize_video_filter()},"
        f"tpad=start_mode=clone:start_duration={FADE_DURATION}"
        f":stop_mode=clone:stop_duration={SPONSOR_HOLD + FADE_DURATION}"
    )

    src_dur_us = ff.probe_duration_us(sponsor_path)
    output_dur_us = int(
        (src_dur_us / 1_000_000 + FADE_DURATION + SPONSOR_HOLD + FADE_DURATION)
        * 1_000_000
    )

    ff.run(
        [
            "ffmpeg", "-y",
            "-i", str(sponsor_path),
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
            "-vf", vfilter,
            *ff.vcodec_flags(vcodec),
            "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            str(output),
        ],
        label="sponsor",
        duration_us=output_dur_us,
    )
    log.info("[rendering] sponsor — done (%.1fs)", time.monotonic() - t0)


def render_full_talk(
    title_video: Path,
    sponsor_video: Path,
    source: Path,
    start: str,
    end: str,
    output: Path,
    vcodec: str,
    sponsor_duration_us: int,
) -> None:
    """
    Render the complete final video in a single ffmpeg filtergraph pass.

    Timeline:
      BLACK_PAD → fade-in → title → xfade → sponsor →
      xfade+audio-fadein → talk → xfade+audio-fadeout →
      title → xfade → sponsor → fade-out → BLACK_PAD

    Inputs:
      [0] title_video   — pre-rendered, TITLE_DURATION + 2*FADE_DURATION long, used twice
      [1] sponsor_video — pre-rendered normalized sponsor, used twice
      [2] source        — raw recording, seeked to start/end via -ss/-t at input level

    No intermediate files — the source is normalised inside the filtergraph.
    """
    if output.exists():
        log.debug("[rendering] %s — already exists, skipping", output.name)
        return

    F = FADE_DURATION
    T = TITLE_DURATION + 2 * F          # title clip duration (5s)
    S = sponsor_duration_us / 1_000_000
    talk_dur_s = (ts_to_us(end) - ts_to_us(start)) / 1_000_000
    B = BLACK_PAD

    # xfade offset = how long the outgoing clip plays solo before blending
    xf0 = B
    xf1 = B + T - F
    xf2 = B + T + S - 2 * F
    xf3 = B + T + S + talk_dur_s - 3 * F
    xf4 = B + 2 * T + S + talk_dur_s - 4 * F
    fade_out_st = B + 2 * T + 2 * S + talk_dur_s - 5 * F

    total_s = fade_out_st + F + B
    total_us = int(total_s * 1_000_000)

    # Audio silence durations flanking the talk
    silence_before = xf2                              # B + T + S - 2F
    silence_after = total_s - silence_before - talk_dur_s

    norm = ff.normalize_video_filter()

    vf = (
        f"[0:v]settb=1/{FPS},split=2[title1v][title2v];"
        f"[1:v]settb=1/{FPS},split=2[spons1v][spons2v];"
        f"[2:v]{norm}[talkv];"
        f"color=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={B + F}[blackv_in];"
        f"[blackv_in][title1v]xfade=transition=fade:duration={F}:offset={xf0}[v01];"
        f"[v01][spons1v]xfade=transition=fade:duration={F}:offset={xf1}[v012];"
        f"[v012][talkv]xfade=transition=fade:duration={F}:offset={xf2}[v0123];"
        f"[v0123][title2v]xfade=transition=fade:duration={F}:offset={xf3}[v01234];"
        f"[v01234][spons2v]xfade=transition=fade:duration={F}:offset={xf4}[v012345];"
        f"[v012345]fade=t=out:st={fade_out_st}:d={F}[vfaded];"
        f"color=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={B}[blackv_out];"
        f"[vfaded][blackv_out]concat=n=2:v=1:a=0[vout]"
    )

    af = (
        f"[2:a]asplit=3[ta1][ta2][ta3];"
        f"[ta1]atrim=start=0:end={F},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={F}[a_fadein];"
        f"[ta2]atrim=start={F}:end={talk_dur_s - F},asetpts=PTS-STARTPTS[a_mid];"
        f"[ta3]atrim=start={talk_dur_s - F}:end={talk_dur_s},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st=0:d={F}[a_fadeout];"
        f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
        f"atrim=end={silence_before},asetpts=PTS-STARTPTS[a_sil_before];"
        f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
        f"atrim=end={silence_after},asetpts=PTS-STARTPTS[a_sil_after];"
        f"[a_sil_before][a_fadein][a_mid][a_fadeout][a_sil_after]concat=n=5:v=0:a=1[aout]"
    )

    label = output.stem
    log.info("[rendering] %s — full talk %.0fs", label, total_s)
    t0 = time.monotonic()
    ff.run(
        [
            "ffmpeg", "-y",
            "-i", str(title_video),
            "-i", str(sponsor_video),
            "-ss", start, "-t", duration_str(start, end), "-i", str(source),
            "-filter_complex", f"{vf};{af}",
            "-map", "[vout]", "-map", "[aout]",
            *ff.vcodec_flags(vcodec),
            "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
            str(output),
        ],
        label=label,
        duration_us=total_us,
    )
    log.info("[rendering] %s — 100%% (%.1fs)", label, time.monotonic() - t0)
