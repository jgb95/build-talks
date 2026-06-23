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
# Every transition uses the same 1s crossfade.
# Non-raw segments are internally padded only to provide that fade headroom;
# we do not intentionally hold content beyond the crossfade window.
TITLE_DURATION = 3
FADE_DURATION = 1   # duration of every crossfade / fade-in / fade-out
BLACK_PAD = 0.5     # held-black duration used by automatic head/tail bookends

# ---- Hardware encoder (Apple Silicon) vs software fallback ----
VCODEC_HW = "h264_videotoolbox"
VCODEC_SW = "libx264"

# ---- Notion property names (override via environment variables) ----
NOTION_CLIPART_PROP = os.getenv("NOTION_CLIPART_PROP", "Clipart")
NOTION_SOCIAL_CARD_PROP = os.getenv("NOTION_SOCIAL_CARD_PROP", "SocialCard")

# ---- Asset URL base (for relative Notion SocialCard paths) ----
# Prefer explicit ASSET_BASE_URL, but keep compatibility with the previous
# DO_SPACES_BUCKET/DO_SPACES_REGION defaults when ASSET_BASE_URL is not set.
ASSET_BASE_URL = os.getenv("ASSET_BASE_URL") or (
    f"https://{os.getenv('DO_SPACES_BUCKET', 'btcpp')}"
    f".{os.getenv('DO_SPACES_REGION', 'nyc3')}.digitaloceanspaces.com"
)

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
    recordings: Path
    talks: Path
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
