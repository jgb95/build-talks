"""
Whisper-based transcription and SRT subtitle generation.

Whisper is loaded once per run and reused across all talks.
Subtitles follow Netflix subtitle standards:
  - Max 2 lines per card
  - Max 42 characters per line
  - Min 1.5s / Max 7s display duration
  - Min 2-frame gap between captions
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# ---- Netflix subtitle standards ----
MAX_CHARS = 42
MAX_LINES = 2
MIN_DURATION = 1.5   # seconds
MAX_DURATION = 7.0   # seconds
MIN_GAP = 2 / 30     # 2 frames at 30 fps


def load_model(model_name: str) -> dict:
    """
    Load the Whisper model and return it in a context dict.

    Loading is deferred until first call (not at import time) so that
    --no-transcribe runs never pay the model-load cost.
    """
    import os
    from faster_whisper import WhisperModel

    log.info("[whisper] loading model '%s'...", model_name)
    device = "cpu"
    # Use Metal (GPU) on Apple Silicon when Core ML support is available.
    if os.uname().sysname == "Darwin":
        try:
            import coremltools  # noqa: F401
            device = "auto"
        except ImportError:
            pass

    model = WhisperModel(model_name, device=device, compute_type="int8")
    log.info("[whisper] model ready")
    return {"model": model, "model_name": model_name}


def transcribe_talk(
    source: Path,
    start: str,
    intro_offset: float,
    srt_path: Path,
    whisper_ctx: dict,
    language: str,
) -> None:
    """
    Transcribe the talk audio from *source* and write an SRT file.

    Timestamps in the SRT are offset by *intro_offset* seconds so that
    captions are aligned to the final assembled video, not to the raw source.

    Args:
        source:        Path to the source recording.
        start:         HH:MM:SS[.mmm] timestamp of the talk start in source.
        intro_offset:  Seconds to add to every subtitle timestamp
                       (accounts for BLACK_PAD + title clip + sponsor clip).
        srt_path:      Destination path for the generated .srt file.
        whisper_ctx:   Context dict returned by load_model().
        language:      BCP-47 language code (e.g. "en").
    """
    from faster_whisper import WhisperModel

    if srt_path.exists():
        log.debug("[whisper] %s — SRT already exists, skipping", srt_path.name)
        return

    model: WhisperModel = whisper_ctx["model"]

    # Convert HH:MM:SS.mmm → seconds for faster-whisper's offset argument.
    h, m, s_str = start.split(":")
    start_s = int(h) * 3600 + int(m) * 60 + float(s_str)

    log.info("[whisper] transcribing %s...", source.name)
    segments_iter, _ = model.transcribe(
        str(source),
        language=language,
        word_timestamps=True,
        beam_size=1,          # greedy; matches distil-model training regime
        vad_filter=True,
        initial_prompt=(
            "Conference talk. Technical content. "
            "Use correct punctuation and capitalisation."
        ),
        clip_timestamps=[start_s],
    )
    segments = list(segments_iter)
    log.info("[whisper] transcription done — %d segment(s)", len(segments))

    words = _collect_words(segments, start_s, intro_offset)
    subtitles = _words_to_subtitles(words)
    _write_srt(subtitles, srt_path)
    log.info("[whisper] wrote %s (%d cue(s))", srt_path.name, len(subtitles))


def _collect_words(
    segments: list,
    start_s: float,
    intro_offset: float,
) -> list[dict]:
    """
    Flatten segment words into a list, trimming to the talk window and
    shifting timestamps so they're relative to the final video.
    """
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start < start_s:
                continue
            words.append(
                {
                    "word": w.word,
                    "start": w.start - start_s + intro_offset,
                    "end": w.end - start_s + intro_offset,
                }
            )
    return words


def _words_to_subtitles(words: list[dict]) -> list[dict]:
    """
    Pack words into subtitle cues following Netflix standards.

    Each cue contains at most MAX_LINES lines of MAX_CHARS characters.
    Cue duration is clamped to [MIN_DURATION, MAX_DURATION].
    A MIN_GAP gap is enforced between consecutive cues.
    """
    if not words:
        return []

    cues: list[dict] = []
    line: list[str] = []
    lines: list[str] = []
    cue_start: float = words[0]["start"]
    last_end = 0.0

    def flush(end: float) -> None:
        nonlocal line, lines, cue_start, last_end
        if lines or line:
            if line:
                lines.append(" ".join(line))
            text = "\n".join(lines)
            start = max(cue_start, last_end + MIN_GAP)
            dur = min(max(end - start, MIN_DURATION), MAX_DURATION)
            cues.append({"start": start, "end": start + dur, "text": text})
            last_end = start + dur
        line = []
        lines = []

    for w in words:
        token = w["word"].strip()
        if not token:
            continue

        candidate = " ".join(line + [token]) if line else token
        if len(candidate) > MAX_CHARS:
            if len(lines) >= MAX_LINES - 1:
                # Current cue is full — flush and start a new one.
                flush(w["start"])
                cue_start = w["start"]
            else:
                # Move to the next line within this cue.
                lines.append(" ".join(line))
            line = [token]
        else:
            line.append(token)

    # Flush the final cue using the last word's end time.
    if words:
        flush(words[-1]["end"])

    return cues


def _srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT HH:MM:SS,mmm format."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(cues: list[dict], path: Path) -> None:
    """Write a list of cue dicts to an SRT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, cue in enumerate(cues, start=1):
            fh.write(
                f"{i}\n"
                f"{_srt_timestamp(cue['start'])} --> {_srt_timestamp(cue['end'])}\n"
                f"{cue['text']}\n\n"
            )
