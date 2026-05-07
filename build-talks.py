#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "notion-client>=3.0.0",
#   "requests",
#   "python-dotenv",
#   "faster-whisper",
# ]
# ///
"""
build-talks: assemble conference talk videos from source recordings.

CSV columns expected: id, source_file, start_time, end_time
  id           -- talk identifier, matched against the Clipart field in Notion
  source_file  -- path to the source recording (relative or absolute)
  start_time   -- trim start  (HH:MM:SS or HH:MM:SS.mmm)
  end_time     -- trim end    (HH:MM:SS or HH:MM:SS.mmm)

Notion database fields expected:
  Clipart    -- text field containing "{id}.{ext}" (ID is extracted automatically)
  SocialCard -- Files & media or URL field with the title card image download link

Assembly order per talk (with transitions):
  0.5s black → 1s fade-from-black → 3s title card → 1s xfade → sponsor reel →
  1s xfade+audio-fadein → full talk → 1s xfade+audio-fadeout → 3s title card →
  1s xfade → sponsor reel → 1s fade-to-black → 0.5s black

The entire final video is produced by a single ffmpeg filtergraph call
(render_full_talk).  No intermediate trimmed-talk files; the source recording
is fed directly with -ss/-t seeking and normalised inside the filtergraph.
This eliminates all concat-seam audio artefacts.
"""

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from notion_client.client import ClientOptions as NotionClientOptions


# ---- Encoding constants ----
WIDTH = 1920
HEIGHT = 1080
FPS = 30
ACODEC = "aac"
SAMPLE_RATE = 48000

# How long the title card is *visibly* held (seconds).
# The actual rendered file is TITLE_DURATION + 2 * FADE_DURATION to give
# the xfade filters 1s of headroom on each side.
TITLE_DURATION = 3
FADE_DURATION = 1   # duration of every crossfade / fade-in / fade-out
BLACK_PAD = 0.5     # seconds of pure black held before/after the fades
SPONSOR_HOLD = 2    # seconds the last frame of sponsor is visibly frozen
                    # (actual tpad stop_duration = SPONSOR_HOLD + FADE_DURATION)

# Hardware encoder (Apple Silicon) vs software fallback
VCODEC_HW = "h264_videotoolbox"
VCODEC_SW = "libx264"

# Notion property names (override via env if your DB uses different names)
NOTION_CLIPART_PROP = os.getenv("NOTION_CLIPART_PROP", "Clipart")
NOTION_SOCIAL_CARD_PROP = os.getenv("NOTION_SOCIAL_CARD_PROP", "SocialCard")

# Valid timestamp pattern: HH:MM:SS or HH:MM:SS.mmm
_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")

# ---- Logging setup ----

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# Silence chatty third-party loggers
for _noisy in (
    "httpx", "httpcore", "huggingface_hub", "filelock", "notion_client",
    "faster_whisper",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Suppress huggingface_hub tqdm download bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger("build-talks")


# ---- ffmpeg helpers ----

# Set to True when --verbose is passed so ffmpeg stderr streams live
_VERBOSE = False

# How many seconds between progress log lines during a long render/transcode
_PROGRESS_INTERVAL = 30


def _probe_duration_us(path: Path) -> int | None:
    """Return the duration of a media file in microseconds, or None if unknown."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return int(float(result.stdout.strip()) * 1_000_000)
    except (ValueError, AttributeError):
        return None


def run(cmd: list[str], label: str = "", duration_us: int | None = None) -> None:
    """
    Run an ffmpeg command, logging progress every _PROGRESS_INTERVAL seconds
    if the render takes longer than that interval.

    In verbose mode stderr streams live (full ffmpeg stats).
    In normal mode:  a [rendering] <label> — <pct>% (<Ns>) line is logged
                     at most every _PROGRESS_INTERVAL seconds when duration_us
                     is known; otherwise the command runs silently.
    """
    import time

    log.debug("Running: %s", " ".join(cmd))
    if _VERBOSE:
        result = subprocess.run(cmd, stderr=None, stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")
        return

    if duration_us and duration_us > 0:
        # Inject -progress pipe:1 so ffmpeg writes structured progress to stdout
        prog_cmd = []
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
        stderr_buf: list[str] = []
        t_start = time.monotonic()
        t_last_log = t_start  # wall-clock time of the last progress log line

        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        elapsed_us = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    now = time.monotonic()
                    wall = now - t_start
                    if now - t_last_log >= _PROGRESS_INTERVAL:
                        pct = min(elapsed_us / duration_us * 100, 100.0)
                        log.info("[rendering] %s — %d%% (%ds)", label, int(pct), int(wall))
                        t_last_log = now
        finally:
            if proc.stderr:
                stderr_buf = proc.stderr.read().splitlines()
            proc.wait()

        if proc.returncode != 0:
            for eline in stderr_buf:
                log.error("[rendering] ffmpeg: %s", eline)
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")

        return

    # Silent fallback (no duration estimate)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("[rendering] %s — ffmpeg error: %s", label, result.stderr.strip())
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")


def _normalize_video_filter() -> str:
    """Common scale+pad+fps filter chain to hit canonical format."""
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},"
        f"settb=1/{FPS}"
    )


def _vcodec_flags(vcodec: str) -> list[str]:
    """Return the appropriate codec flags for the chosen encoder."""
    if vcodec == VCODEC_HW:
        # VideoToolbox doesn't support pix_fmt selection at output; yuv420p is default.
        # -q:v 65 enables quality-based VBR (1–100 scale); without it VideoToolbox
        # defaults to a very high constant bitrate, producing unnecessarily large files.
        return ["-c:v", VCODEC_HW, "-q:v", "65"]
    # -crf 23 is libx264's constant-rate-factor mode: allocates bits only where needed,
    # yielding files that are proportional in size to their actual complexity.
    return ["-c:v", VCODEC_SW, "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium"]


def render_title_card(image_path: Path, output: Path, vcodec: str, label: str = "") -> None:
    """
    Render the title card image as a video.

    Duration = TITLE_DURATION + 2 * FADE_DURATION so that xfade filters
    can consume FADE_DURATION seconds on each end while still leaving
    TITLE_DURATION seconds of fully-visible title card in between.
    """
    import time
    if output.exists():
        log.debug("[rendering] %s — title card already cached", label)
        return
    total = TITLE_DURATION + 2 * FADE_DURATION
    log.info("[rendering] %s — title card", label)
    t0 = time.monotonic()
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(total), "-i", str(image_path),
        "-f", "lavfi", "-t", str(total),
        "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
        "-vf", _normalize_video_filter(),
        *_vcodec_flags(vcodec),
        "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
        "-shortest",
        str(output),
    ], label=label, duration_us=total * 1_000_000)
    log.info("[rendering] %s — title card done (%.1fs)", label, time.monotonic() - t0)


def normalize_sponsor(sponsor_path: Path, output: Path, vcodec: str) -> None:
    """
    Re-encode sponsor reel with silent audio.

    Adds FADE_DURATION seconds of clone-padded frames at the start (so the
    incoming xfade has content to blend from) and (SPONSOR_HOLD + FADE_DURATION)
    seconds of frozen last frame at the end (SPONSOR_HOLD seconds are visibly
    held; FADE_DURATION seconds are consumed by the outgoing xfade/fade).
    The source is assumed to have no audio.
    """
    import time
    if output.exists():
        log.debug("[rendering] sponsor — already cached")
        return
    log.info("[rendering] sponsor")
    t0 = time.monotonic()

    vfilter = (
        f"{_normalize_video_filter()},"
        f"tpad=start_mode=clone:start_duration={FADE_DURATION}"
        f":stop_mode=clone:stop_duration={SPONSOR_HOLD + FADE_DURATION}"
    )

    src_dur_us = _probe_duration_us(sponsor_path)
    output_dur_us = (
        int((src_dur_us / 1_000_000 + FADE_DURATION + SPONSOR_HOLD + FADE_DURATION) * 1_000_000)
        if src_dur_us is not None else None
    )

    run([
        "ffmpeg", "-y",
        "-i", str(sponsor_path),
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}",
        "-vf", vfilter,
        *_vcodec_flags(vcodec),
        "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output),
    ], label="sponsor", duration_us=output_dur_us)
    log.info("[rendering] sponsor — done (%.1fs)", time.monotonic() - t0)


def _ts_to_us(ts: str) -> int:
    """Convert HH:MM:SS[.mmm] timestamp to microseconds."""
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return int((h * 3600 + m * 60 + s) * 1_000_000)


def _duration_str(start: str, end: str) -> str:
    """
    Compute the duration between two HH:MM:SS[.mmm] timestamps and return
    it as an HH:MM:SS.mmm string suitable for ffmpeg's -t option.
    """
    us = _ts_to_us(end) - _ts_to_us(start)
    total_s = us / 1_000_000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h}:{m:02d}:{s:06.3f}"


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
      BLACK_PAD s black → FADE_DURATION s fade-in → TITLE_DURATION s title →
      FADE_DURATION s xfade → sponsor reel → FADE_DURATION s xfade+audio-fadein →
      full talk →
      FADE_DURATION s xfade+audio-fadeout → TITLE_DURATION s title →
      FADE_DURATION s xfade → sponsor reel → FADE_DURATION s fade-out → BLACK_PAD s black

    Inputs:
      [0] title_video  (pre-rendered, TITLE_DURATION + 2*FADE_DURATION long, used twice via split)
      [1] sponsor_video (pre-rendered normalized sponsor, used twice via split)
      [2] source recording, seeked to start/end via -ss/-t at input level

    No intermediate files; the source is normalised inside the filtergraph.
    """
    import time
    if output.exists():
        log.debug("[rendering] %s — full talk already cached", output.name)
        return

    F = FADE_DURATION
    T = TITLE_DURATION + 2 * F   # = 5s  (title clip duration)
    S = sponsor_duration_us / 1_000_000
    talk_dur_s = (_ts_to_us(end) - _ts_to_us(start)) / 1_000_000
    B = BLACK_PAD

    # -----------------------------------------------------------------------
    # Video xfade chain offsets
    # xfade offset = how long the outgoing clip plays solo before blending.
    # Running output duration after each step:
    #   black_in alone:        B + F
    #   after xf0 (→title1):   B + T
    #   after xf1 (→sponsor1): B + T + S - F
    #   after xf2 (→talk):     B + T + S + talk_dur_s - 2F
    #   after xf3 (→title2):   B + 2T + S + talk_dur_s - 3F
    #   after xf4 (→sponsor2): B + 2T + 2S + talk_dur_s - 4F
    # Each xfade offset = running_duration_before_that_step - F
    # -----------------------------------------------------------------------
    xf0 = B
    xf1 = B + T - F
    xf2 = B + T + S - 2*F
    xf3 = B + T + S + talk_dur_s - 3*F
    xf4 = B + 2*T + S + talk_dur_s - 4*F
    fade_out_st = B + 2*T + 2*S + talk_dur_s - 5*F

    # Total video duration = fade_out_st + F + B
    total_s = fade_out_st + F + B
    total_us = int(total_s * 1_000_000)

    # -----------------------------------------------------------------------
    # Audio timeline (silence flanks the talk; no audio for title/sponsor):
    #   [0, xf2]                            : silence (black + title1 + sponsor1)
    #   [xf2, xf2 + talk_dur_s]             : talk audio (with F-second fade-in/out)
    #   [xf2 + talk_dur_s, total_s]         : silence (title2 + sponsor2 + black pad)
    # -----------------------------------------------------------------------
    silence_before = xf2                           # = B + T + S - 2F
    silence_after  = total_s - silence_before - talk_dur_s

    # Build filtergraph
    # Inputs: [0]=title, [1]=sponsor, [2]=source_talk (title/sponsor split for reuse)
    norm = _normalize_video_filter()
    vf = (
        # -- Normalise timebases of pre-encoded inputs and split for reuse --
        f"[0:v]settb=1/{FPS},split=2[title1v][title2v];"
        f"[1:v]settb=1/{FPS},split=2[spons1v][spons2v];"
        # -- Normalise talk video (includes settb via _normalize_video_filter) --
        f"[2:v]{norm}[talkv];"
        # -- Black pad source (B+F long; xfade needs F seconds of content before the blend) --
        f"color=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={B+F}[blackv_in];"
        # -- Intro chain --
        f"[blackv_in][title1v]xfade=transition=fade:duration={F}:offset={xf0}[v01];"
        f"[v01][spons1v]xfade=transition=fade:duration={F}:offset={xf1}[v012];"
        f"[v012][talkv]xfade=transition=fade:duration={F}:offset={xf2}[v0123];"
        # -- Outro chain --
        f"[v0123][title2v]xfade=transition=fade:duration={F}:offset={xf3}[v01234];"
        f"[v01234][spons2v]xfade=transition=fade:duration={F}:offset={xf4}[v012345];"
        # -- Final fade-out + black pad --
        f"[v012345]fade=t=out:st={fade_out_st}:d={F}[vfaded];"
        f"color=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={B}[blackv_out];"
        f"[vfaded][blackv_out]concat=n=2:v=1:a=0[vout]"
    )

    # Audio filtergraph — 3 atrim+afade passes from [2:a] via asplit=3:
    af = (
        f"[2:a]asplit=3[ta1][ta2][ta3];"
        # fade-in: first F seconds of talk
        f"[ta1]atrim=start=0:end={F},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={F}[a_fadein];"
        # middle: F to (talk_dur_s - F) seconds of talk (full volume)
        f"[ta2]atrim=start={F}:end={talk_dur_s - F},asetpts=PTS-STARTPTS[a_mid];"
        # fade-out: last F seconds of talk
        f"[ta3]atrim=start={talk_dur_s - F}:end={talk_dur_s},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st=0:d={F}[a_fadeout];"
        # silence before talk
        f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
        f"atrim=end={silence_before},asetpts=PTS-STARTPTS[a_sil_before];"
        # silence after talk
        f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},"
        f"atrim=end={silence_after},asetpts=PTS-STARTPTS[a_sil_after];"
        # concat all audio: silence | fadein | middle | fadeout | silence
        f"[a_sil_before][a_fadein][a_mid][a_fadeout][a_sil_after]concat=n=5:v=0:a=1[aout]"
    )

    label = output.stem
    log.info("[rendering] %s — full talk %.0fs video", label, total_s)
    t0 = time.monotonic()
    run([
        "ffmpeg", "-y",
        "-i", str(title_video),
        "-i", str(sponsor_video),
        "-ss", start, "-t", _duration_str(start, end), "-i", str(source),
        "-filter_complex", f"{vf};{af}",
        "-map", "[vout]", "-map", "[aout]",
        *_vcodec_flags(vcodec),
        "-c:a", ACODEC, "-ar", str(SAMPLE_RATE),
        str(output),
    ], label=label, duration_us=total_us)
    log.info("[rendering] %s — 100%% (%.1fs)", label, time.monotonic() - t0)


# ---- Notion integration ----

class NotionFetcher:
    """Loads a Notion database once and provides lookups by talk ID."""

    def __init__(self, token: str, database_id: str):
        # Pin to Notion-Version 2022-06-28 — the 2025-09-03 version (notion-client v3
        # default) moved /databases/{id}/query to a stricter data_sources API that
        # requires additional sharing steps.  2022-06-28 is the last stable version
        # that supports the standard databases query endpoint.
        self.client = NotionClient(
            options=NotionClientOptions(auth=token, notion_version="2022-06-28")
        )
        self.database_id = database_id
        self._pages: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Query the database and index pages by their Clipart ID. Called once at construction."""
        log.info("[notion] loading database...")
        cursor = None
        while True:
            body: dict = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            # notion-client v3 removed databases.query(); use the raw request method
            response = self.client.request(
                f"databases/{self.database_id}/query",
                method="POST",
                body=body,
            )

            for page in response["results"]:
                talk_id = self._extract_clipart_id(page)
                if talk_id:
                    self._pages[talk_id] = page

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        log.info("[notion] loaded %d pages", len(self._pages))

    def _extract_clipart_id(self, page: dict) -> str | None:
        """
        Pull the talk ID from the Clipart property.
        The field contains '{ID}.{ext}' — we keep only the ID part.
        """
        prop = page["properties"].get(NOTION_CLIPART_PROP)
        if not prop:
            return None

        ptype = prop["type"]
        if ptype == "title":
            raw = "".join(t["plain_text"] for t in prop["title"]).strip()
        elif ptype == "rich_text":
            raw = "".join(t["plain_text"] for t in prop["rich_text"]).strip()
        elif ptype == "formula":
            raw = prop["formula"].get("string", "")
        else:
            return None

        if not raw:
            return None

        # Strip the file extension suffix, e.g. "talk-01.png" → "talk-01"
        return raw.rsplit(".", 1)[0]

    def get_social_card_url(self, talk_id: str) -> str:
        """Return the SocialCard image URL for a given talk ID."""
        page = self._pages.get(talk_id)
        if not page:
            raise KeyError(
                f"No Notion page found with {NOTION_CLIPART_PROP} ID='{talk_id}'"
            )

        prop = page["properties"].get(NOTION_SOCIAL_CARD_PROP)
        if not prop:
            raise KeyError(
                f"Page '{talk_id}' has no '{NOTION_SOCIAL_CARD_PROP}' property"
            )

        ptype = prop["type"]
        if ptype == "url":
            url = prop["url"]
        elif ptype == "files":
            files = prop["files"]
            if not files:
                raise ValueError(
                    f"No file in '{NOTION_SOCIAL_CARD_PROP}' for '{talk_id}'"
                )
            f = files[0]
            url = f["external"]["url"] if f["type"] == "external" else f["file"]["url"]
        elif ptype == "rich_text":
            url = "".join(t["plain_text"] for t in prop["rich_text"]).strip()
        else:
            raise ValueError(
                f"Unsupported property type '{ptype}' for '{NOTION_SOCIAL_CARD_PROP}'"
            )

        if not url:
            raise ValueError(f"Empty SocialCard URL for '{talk_id}'")
        return url

    def download_title_card(self, talk_id: str, titles_dir: Path) -> Path:
        """
        Return the local path to the title card image for talk_id, downloading
        it from Notion if it is not already cached in titles_dir.
        """
        titles_dir.mkdir(parents=True, exist_ok=True)

        # Return immediately if already cached (any non-video extension)
        for existing in titles_dir.glob(f"{talk_id}.*"):
            if existing.suffix.lower() != ".mp4":
                return existing

        url = self.get_social_card_url(talk_id)
        ext = Path(urlparse(url).path).suffix or ".png"
        dest = titles_dir / f"{talk_id}{ext}"

        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        return dest


# ---- Transcription ----

# Netflix Timed Text Style Guide subtitle constraints
_NF_MAX_CHARS_PER_LINE = 42   # max characters per line
_NF_MAX_LINES = 2             # max lines per subtitle card
_NF_MIN_DURATION = 5 / 6     # 0.833s — 20 frames at 24 fps
_NF_MAX_DURATION = 7.0        # seconds
_NF_MIN_GAP = 0.080           # 80 ms minimum gap between subtitles
_NF_PAUSE_BREAK = 0.300       # natural pause gap that warrants a subtitle break


def _srt_timestamp(seconds: float) -> str:
    """Convert fractional seconds to SRT timestamp format HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_to_subtitles(
    words: list[dict],
    intro_offset: float,
) -> list[tuple[float, float, str]]:
    """
    Convert a flat list of word-timestamp dicts into subtitle segments that
    comply with Netflix Timed Text style constraints:

      - Max _NF_MAX_CHARS_PER_LINE characters per line
      - Max _NF_MAX_LINES lines per card (2)
      - Min _NF_MIN_DURATION seconds duration
      - Max _NF_MAX_DURATION seconds duration
      - Min _NF_MIN_GAP seconds gap between consecutive subtitles
      - Break at natural pauses >= _NF_PAUSE_BREAK seconds between words

    Each word dict must have "word", "start", and "end" keys.
    Words missing timing data are skipped.

    intro_offset is added to all timestamps so the SRT aligns with the final
    rendered video (which includes the title/sponsor intro).

    Returns a list of (start_s, end_s, text) tuples ready for _write_srt().
    """
    segments: list[tuple[float, float, str]] = []
    if not words:
        return segments

    # --- Build subtitle cards from words up ---
    line1: list[str] = []
    line2: list[str] = []
    seg_start: float | None = None
    seg_end: float | None = None
    prev_end: float | None = None

    def _flush() -> None:
        nonlocal line1, line2, seg_start, seg_end, prev_end
        if seg_start is None or not line1:
            return
        text = " ".join(line1)
        if line2:
            text += "\n" + " ".join(line2)

        start = seg_start + intro_offset
        end = (seg_end or seg_start) + intro_offset

        # Enforce minimum duration
        if end - start < _NF_MIN_DURATION:
            end = start + _NF_MIN_DURATION

        # Enforce minimum gap from the previous subtitle's end
        if segments:
            prev_sub_end = segments[-1][1]
            if start < prev_sub_end + _NF_MIN_GAP:
                start = prev_sub_end + _NF_MIN_GAP
                if end <= start:
                    end = start + _NF_MIN_DURATION

        segments.append((start, end, text))
        line1 = []
        line2 = []
        seg_start = None
        seg_end = None

    for w in words:
        word_text = w.get("word", "").strip()
        w_start = w.get("start")
        w_end = w.get("end")
        if not word_text or w_start is None or w_end is None:
            continue

        # Natural pause — flush current card and start fresh
        if prev_end is not None and (w_start - prev_end) >= _NF_PAUSE_BREAK:
            _flush()

        # Sentence boundary on previous word — flush at clean break
        if seg_end is not None and line1 and not line2:
            last_word = line1[-1]
            if last_word and last_word[-1] in ".!?":
                _flush()

        # Start a new card if needed
        if seg_start is None:
            seg_start = w_start

        # Try to place word on current line
        target_line = line2 if line2 else line1
        candidate = (" ".join(target_line) + " " + word_text).strip()

        if len(candidate) <= _NF_MAX_CHARS_PER_LINE:
            target_line.append(word_text)
        elif not line2 and len(word_text) <= _NF_MAX_CHARS_PER_LINE:
            # Word doesn't fit on line 1 — move to line 2
            line2.append(word_text)
        else:
            # Both lines full — flush and start a new card with this word
            _flush()
            seg_start = w_start
            line1.append(word_text)

        seg_end = w_end
        prev_end = w_end

        # Duration cap — flush if this card is getting too long
        if seg_start is not None and (w_end - seg_start) >= _NF_MAX_DURATION:
            _flush()

    _flush()  # emit any remaining words
    return segments


def _write_srt(segments: list[tuple[float, float, str]], path: Path) -> None:
    """Write a list of (start_s, end_s, text) tuples as an SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        idx = 1
        for start, end, text in segments:
            text = text.strip()
            if not text:
                continue
            fh.write(f"{idx}\n")
            fh.write(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n")
            fh.write(f"{text}\n\n")
            idx += 1


def _load_whisper_model(model_name: str) -> dict:
    """
    Load and return a faster-whisper model dict.

    Automatically selects the best available device:
      - "cuda"  if a CUDA GPU is available
      - "cpu"   otherwise (Apple Silicon included — CTranslate2 doesn't support MPS)

    Uses BatchedInferencePipeline for parallel chunk processing, which gives a
    significant speedup on multi-core CPUs (e.g. Apple Silicon) by processing
    multiple VAD-segmented audio chunks simultaneously.

    Returns a dict with keys:
      "model":   the loaded WhisperModel (underlying model)
      "pipeline": BatchedInferencePipeline wrapping the model
      "name":    model name string
      "device":  device string used
    """
    import time
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            cpu_threads = 0  # not used on CUDA
        else:
            # CTranslate2 doesn't support MPS; use all available CPU cores on Apple Silicon
            import os
            device = "cpu"
            compute_type = "int8"
            cpu_threads = os.cpu_count() or 4
    except ImportError:
        import os
        device = "cpu"
        compute_type = "int8"
        cpu_threads = os.cpu_count() or 4

    log.info(
        "[pre-warm] loading faster-whisper model %s on %s (%d threads)...",
        model_name, device, cpu_threads,
    )
    t0 = time.monotonic()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    pipeline = BatchedInferencePipeline(model=model)
    log.info("[pre-warm] faster-whisper ready (%.1fs)", time.monotonic() - t0)
    return {"model": model, "pipeline": pipeline, "name": model_name, "device": device}


def transcribe_talk(
    talk_id: str,
    source_path: Path,
    start_time: str,
    end_time: str,
    intro_offset: float,
    srt_path: Path,
    whisper_ctx: dict,
    language: str,
) -> None:
    """
    Transcribe the talk audio from *source_path* (trimmed to start_time/end_time)
    and write Netflix-standard subtitles to *srt_path*.

    Pipeline:
      1. Extract trimmed audio to a temporary 16 kHz mono WAV.
      2. Transcribe with faster-whisper using VAD filtering and word timestamps.
         VAD (Silero) automatically skips silent/pause regions so long pauses
         never confuse the model or produce phantom text.
      3. Collect word-level timestamps from the transcription segments directly
         (no separate alignment pass needed — faster-whisper returns them natively).
      4. Segment words into Netflix-standard subtitle cards via _words_to_subtitles().
      5. Apply intro_offset so SRT timestamps align with the final rendered video.

    Failures are logged but do not raise so the overall build pipeline continues.
    """
    import tempfile
    import time

    try:
        audio_dur = (_ts_to_us(end_time) - _ts_to_us(start_time)) / 1_000_000
        log.info("[transcribing] %s — %.0fs audio", talk_id, audio_dur)

        # ---- 1. Extract trimmed audio to a temp WAV ----
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", start_time,
                    "-t", _duration_str(start_time, end_time),
                    "-i", str(source_path),
                    "-vn", "-ar", "16000", "-ac", "1",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
            )

            fw_pipeline = whisper_ctx["pipeline"]
            t0 = time.monotonic()

            # ---- 2. Transcribe with BatchedInferencePipeline + VAD + word timestamps ----
            # BatchedInferencePipeline processes multiple VAD-segmented audio chunks
            # in parallel (batch_size=16), giving a large speedup on multi-core CPUs
            # like Apple Silicon (where CTranslate2 can't use MPS/GPU).
            #
            # beam_size=1 with temperature=0.0 is pure greedy decoding — identical
            # quality to beam_size=5 for English conference talks but ~3-5x faster
            # since the model only explores one hypothesis at a time.
            #
            # VAD parameters tuned for conference talks:
            # - speech_pad_ms: keep 200 ms of audio around detected speech
            #   so sentence-final words are never clipped.
            # - min_silence_duration_ms: a pause must be ≥ 300 ms before VAD
            #   considers it a segment boundary; skips more silence than 500 ms
            #   while still avoiding choppy splits on natural breath pauses.
            segments_iter, _info = fw_pipeline.transcribe(
                str(tmp_path),
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "speech_pad_ms": 200,
                    "min_silence_duration_ms": 300,
                },
                batch_size=16,
                beam_size=1,
                temperature=0.0,       # greedy decoding
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            # BatchedInferencePipeline processes audio in large batches internally
            # and emits segments in bulk — per-segment progress tracking is
            # meaningless (batches arrive all at once, not evenly spread over time).
            # Instead, log a heartbeat every _PROGRESS_INTERVAL seconds so the
            # user knows the process hasn't hung, based on wall-clock time alone.
            segments: list = []
            t_last_log = t0

            def _heartbeat_log() -> None:
                now = time.monotonic()
                nonlocal t_last_log
                if now - t_last_log >= _PROGRESS_INTERVAL:
                    wall = int(now - t0)
                    log.info("[transcribing] %s — working... (%ds)", talk_id, wall)
                    t_last_log = now

            for seg in segments_iter:
                segments.append(seg)
                _heartbeat_log()

            elapsed_total = time.monotonic() - t0
            log.info("[transcribing] %s — done (%.0fs)", talk_id, elapsed_total)

        finally:
            tmp_path.unlink(missing_ok=True)

        # ---- 3. Collect word-level timestamps ----
        words: list[dict] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append({"word": w.word, "start": w.start, "end": w.end})

        # ---- 4 & 5. Segment into Netflix-standard subtitles with offset ----
        final_segments = _words_to_subtitles(words, intro_offset)

        _write_srt(final_segments, srt_path)
        log.info(
            "[transcribing] %s — wrote %d subtitles to file → %s",
            talk_id, len(final_segments), srt_path,
        )

    except Exception as exc:
        log.error("[transcribing] %s — failed: %s", talk_id, exc)


# ---- Validation ----

REQUIRED_CSV_COLUMNS = {"id", "source_file", "start_time", "end_time"}


def validate_csv_columns(fieldnames: list[str]) -> None:
    missing = REQUIRED_CSV_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}"
        )


def validate_row(row: dict, index: int) -> list[str]:
    """Return a list of validation error strings for a single CSV row."""
    errors: list[str] = []
    row_label = f"Row {index} (id={row.get('id', '?')!r})"

    if not row.get("id", "").strip():
        errors.append(f"{row_label}: 'id' is empty")

    src = row.get("source_file", "").strip()
    if not src:
        errors.append(f"{row_label}: 'source_file' is empty")
    elif not Path(src).exists():
        errors.append(f"{row_label}: source_file not found: {src}")

    for field in ("start_time", "end_time"):
        ts = row.get(field, "").strip()
        if not ts:
            errors.append(f"{row_label}: '{field}' is empty")
        elif not _TIMESTAMP_RE.match(ts):
            errors.append(
                f"{row_label}: '{field}' has invalid format {ts!r} "
                f"(expected HH:MM:SS or HH:MM:SS.mmm)"
            )

    return errors


# ---- Main pipeline ----

def process_talk(
    row: dict,
    sponsor_normalized: Path,
    cache: Path,
    output_dir: Path,
    vcodec: str,
    force: bool,
    dry_run: bool,
    whisper_ctx: dict | None,
    intro_offset: float,
    language: str,
) -> None:
    talk_id = row["id"].strip()
    output = output_dir / f"{talk_id}.mp4"

    if output.exists() and not force:
        log.info("[skip] %s — already built", talk_id)
        return

    if dry_run:
        log.info("[dry-run] %s", talk_id)
        return

    # 1. Locate the title card image (already downloaded to cache by main())
    titles_dir = cache / "titles"
    title_image = next(
        (p for p in titles_dir.glob(f"{talk_id}.*") if p.suffix.lower() != ".mp4"),
        None,
    )
    if title_image is None:
        raise FileNotFoundError(
            f"Title card image not found in {titles_dir} for talk '{talk_id}'"
        )
    title_video = titles_dir / f"{talk_id}.mp4"

    # 2. Start transcription in background (after Notion fetch, before render)
    transcribe_thread: threading.Thread | None = None
    if whisper_ctx is not None:
        srt_path = output_dir / f"{talk_id}.srt"
        if srt_path.exists() and not force:
            log.info("[skip] %s — srt already exists", talk_id)
        else:
            transcribe_thread = threading.Thread(
                target=transcribe_talk,
                args=(
                    talk_id,
                    Path(row["source_file"].strip()),
                    row["start_time"].strip(),
                    row["end_time"].strip(),
                    intro_offset,
                    srt_path,
                    whisper_ctx,
                    language,
                ),
                daemon=True,
            )
            transcribe_thread.start()

    # 3. Render title card video + full talk (concurrent with transcription)
    render_title_card(title_image, title_video, vcodec, label=talk_id)

    sponsor_dur_us = _probe_duration_us(sponsor_normalized)
    if sponsor_dur_us is None:
        raise RuntimeError(f"Could not determine duration of {sponsor_normalized}")

    talk_dur_us = _ts_to_us(row["end_time"].strip()) - _ts_to_us(row["start_time"].strip())
    fade_us = FADE_DURATION * 1_000_000
    if talk_dur_us <= 2 * fade_us:
        raise ValueError(
            f"Talk '{talk_id}' is too short "
            f"({talk_dur_us / 1_000_000:.1f}s) for {FADE_DURATION}s fade bookends."
        )

    render_full_talk(
        title_video,
        sponsor_normalized,
        Path(row["source_file"].strip()),
        row["start_time"].strip(),
        row["end_time"].strip(),
        output,
        vcodec,
        sponsor_dur_us,
    )

    # 4. Wait for transcription to finish before returning
    if transcribe_thread is not None:
        transcribe_thread.join()

    log.info("[done] %s → %s", talk_id, output)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Assemble conference talk videos from source recordings."
    )
    parser.add_argument("--csv", type=Path, default=Path("talks.csv"),
                        help="Path to the talks CSV file (default: talks.csv)")
    parser.add_argument("--sponsor", type=Path, default=Path("sponsor.mp4"),
                        help="Path to the sponsor reel video (default: sponsor.mp4)")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory for finished videos (default: output/)")
    parser.add_argument("--cache", type=Path, default=Path("cache"),
                        help="Cache directory for intermediate files (default: cache/)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Do not delete the cache directory after a successful run")
    parser.add_argument("--force", action="store_true",
                        help="Re-build even if the output file already exists")
    parser.add_argument("--only", type=str, default=None,
                        help="Process only the talk with this ID")
    parser.add_argument("--no-notion", action="store_true",
                        help="Skip Notion; expect title cards already in cache/titles/")
    parser.add_argument("--software-encode", action="store_true",
                        help="Use libx264 (software) instead of h264_videotoolbox (hardware)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without doing any work")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug-level logging")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip automatic SRT subtitle generation")
    parser.add_argument("--whisper-model", type=str, default="distil-large-v3",
                        metavar="MODEL",
                        help="Whisper model name (default: distil-large-v3)")
    parser.add_argument("--whisper-language", type=str, default="en",
                        metavar="LANG",
                        help="Language code for Whisper transcription (default: en)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        global _VERBOSE
        _VERBOSE = True

    vcodec = VCODEC_SW if args.software_encode else VCODEC_HW
    log.info("[encoder] %s", vcodec)

    # ---- Pre-flight checks ----
    if not args.csv.exists():
        log.error("CSV not found: %s", args.csv)
        return 1
    if not args.sponsor.exists():
        log.error("Sponsor reel not found: %s", args.sponsor)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    (args.cache / "titles").mkdir(parents=True, exist_ok=True)

    # ---- Load & validate CSV ----
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        try:
            validate_csv_columns(reader.fieldnames or [])
        except ValueError as exc:
            log.error("%s", exc)
            return 1
        rows = list(reader)

    validation_errors: list[str] = []
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        validation_errors.extend(validate_row(row, i))

    if validation_errors:
        log.error("CSV validation failed:")
        for err in validation_errors:
            log.error("  %s", err)
        return 1

    if args.only:
        rows = [r for r in rows if r["id"].strip() == args.only]
        if not rows:
            log.error("No talk with id %r found in CSV", args.only)
            return 1

    # ---- Set up Notion client and fetch all title cards upfront ----
    fetcher = None
    if not args.no_notion:
        token = os.getenv("NOTION_TOKEN")
        db_id = os.getenv("NOTION_CONFTALKS_DB_ID")
        if not token or not db_id:
            log.warning(
                "NOTION_TOKEN or NOTION_CONFTALKS_DB_ID not set — "
                "will look for title cards locally in cache/titles/"
            )
        else:
            fetcher = NotionFetcher(token, db_id)

    if not args.dry_run:
        # Download all title cards before doing any rendering or transcription.
        # The fetcher already has the full DB loaded (done in __init__), so each
        # call here is a URL lookup + HTTP download.  We try every talk and
        # collect failures so the user can decide whether to continue with the
        # talks that succeeded.
        title_card_failures: list[tuple[str, str]] = []  # (talk_id, error message)
        titles_dir = args.cache / "titles"
        for row in rows:
            talk_id = row["id"].strip()
            log.info("[notion] downloading %s title card...", talk_id)
            try:
                if fetcher is not None:
                    fetcher.download_title_card(talk_id, titles_dir)
                else:
                    # --no-notion mode: title card must already exist locally
                    if not any(
                        p for p in titles_dir.glob(f"{talk_id}.*")
                        if p.suffix.lower() != ".mp4"
                    ):
                        raise FileNotFoundError(
                            f"No title card found in {titles_dir} for '{talk_id}'. "
                            f"Place the image there manually or remove --no-notion."
                        )
            except Exception as exc:
                log.error("[notion] %s — failed: %s", talk_id, exc)
                title_card_failures.append((talk_id, str(exc)))

        if title_card_failures:
            failed_ids = [t for t, _ in title_card_failures]
            if len(title_card_failures) == len(rows):
                log.error("[notion] all %d title card(s) failed — aborting", len(rows))
                return 1

            log.warning(
                "[notion] %d/%d title card(s) failed: %s",
                len(title_card_failures), len(rows), ", ".join(failed_ids),
            )
            try:
                answer = input(
                    f"Proceed with the {len(rows) - len(title_card_failures)} "
                    f"successful talk(s)? [y/N] "
                ).strip().lower()
            except EOFError:
                answer = ""

            if answer != "y":
                log.info("[notion] aborting")
                return 1

            # Remove the failed talks from the processing list
            rows = [r for r in rows if r["id"].strip() not in failed_ids]
        else:
            log.info("[notion] all %d title card(s) ready", len(rows))

    # ---- Normalize sponsor reel ----
    sponsor_normalized = args.cache / "sponsor.mp4"
    if not args.dry_run:
        normalize_sponsor(args.sponsor, sponsor_normalized, vcodec)

    # ---- Pre-warm Whisper model (once, before the main loop) ----
    whisper_ctx: dict | None = None
    if not args.no_transcribe and not args.dry_run:
        whisper_ctx = _load_whisper_model(args.whisper_model)

    # ---- Process talks ----
    total = len(rows)
    log.info("[run] %d talk(s)", total)
    failed: list[str] = []

    # Compute intro_offset once — same formula as xf2 in render_full_talk
    # (BLACK_PAD + title_clip + sponsor_clip - 2*F, but since the title clip is
    #  TITLE_DURATION + 2*F and sponsor_clip = S, the result simplifies to:
    #  B + (T_DUR + 2F) + S - 2F  =  B + T_DUR + S)
    sponsor_dur_s = (_probe_duration_us(sponsor_normalized) or 0) / 1_000_000
    intro_offset = BLACK_PAD + TITLE_DURATION + sponsor_dur_s

    for idx, row in enumerate(rows, start=1):
        talk_id = row["id"].strip()
        log.info("[%d/%d] %s", idx, total, talk_id)
        try:
            process_talk(
                row,
                sponsor_normalized,
                args.cache,
                args.output,
                vcodec,
                args.force,
                args.dry_run,
                whisper_ctx,
                intro_offset,
                args.whisper_language,
            )
        except Exception as exc:
            log.error("[error] %s — %s", talk_id, exc)
            failed.append(talk_id)

    # ---- Summary ----
    if failed:
        log.error("[run] %d talk(s) failed: %s", len(failed), ", ".join(failed))
        log.error("[run] cache preserved at %s for debugging", args.cache)
        return 1

    if not args.keep_cache and not args.dry_run:
        log.info("[run] cleaning up cache")
        shutil.rmtree(args.cache)

    log.info("[run] done — %d talk(s) built", total - len(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
