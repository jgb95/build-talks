"""
Encoding constants and the Config dataclass that flows through the pipeline.

Constants define the canonical output format and timing parameters.
Config carries per-run settings (paths, flags) so functions don't need
long parameter lists and there's no global mutable state.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# ---- Output format ----
WIDTH = 1920
HEIGHT = 1080
FPS = 30
ACODEC = "aac"
SAMPLE_RATE = 48000

# ---- Timing (seconds) ----
# How long the title card is *visibly* held.
# The rendered title clip is TITLE_DURATION + 2 * FADE_DURATION to give
# the xfade filters FADE_DURATION seconds of headroom on each side.
TITLE_DURATION = 3
FADE_DURATION = 1   # duration of every crossfade / fade-in / fade-out
BLACK_PAD = 0.5     # pure black held before/after the fades
SPONSOR_HOLD = 2    # seconds the last frame of sponsor is visibly frozen
                    # (actual tpad stop_duration = SPONSOR_HOLD + FADE_DURATION)

# ---- Hardware encoder (Apple Silicon) vs software fallback ----
VCODEC_HW = "h264_videotoolbox"
VCODEC_SW = "libx264"

# ---- Notion property names (override via environment variables) ----
NOTION_CLIPART_PROP = os.getenv("NOTION_CLIPART_PROP", "Clipart")
NOTION_SOCIAL_CARD_PROP = os.getenv("NOTION_SOCIAL_CARD_PROP", "SocialCard")

# ---- Digital Ocean Spaces (upload + URL resolution) ----
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "btcpp")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "nyc3")
DO_SPACES_BASE_URL = (
    f"https://{DO_SPACES_BUCKET}.{DO_SPACES_REGION}.digitaloceanspaces.com"
)
DO_SPACES_KEY    = os.getenv("DO_SPACES_KEY", "")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET", "")

# ---- Validation ----
# Valid timestamp: HH:MM:SS or HH:MM:SS.mmm
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")


@dataclass
class Config:
    """
    All per-run settings collected in one place.

    Populated from parsed CLI arguments in cli.main() and passed
    through the pipeline so functions receive context without needing
    a long parameter list or global mutable state.
    """
    csv: Path
    sponsor: Path
    output: Path
    cache: Path
    keep_cache: bool
    force: bool
    only: str | None
    no_notion: bool
    vcodec: str
    dry_run: bool
    verbose: bool
    no_transcribe: bool
    no_subtitles: bool
    whisper_model: str
    whisper_language: str
    upload: bool
