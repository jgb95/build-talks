"""
Command-line entry point.

Handles argument parsing, logging setup, CSV validation, pre-flight checks,
title card downloads, and the main per-talk processing loop.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from build_talks.config import (
    TIMESTAMP_RE,
    VCODEC_HW,
    VCODEC_SW,
    BLACK_PAD,
    TITLE_DURATION,
    FADE_DURATION,
    SPONSOR_HOLD,
    Config,
)
from build_talks import ffmpeg
from build_talks.ffmpeg import ts_to_us
from build_talks.notion import NotionFetcher
from build_talks.render import assemble
from build_talks.segment import BLACK, Segment, talk_offset
from build_talks.transcribe import (
    load_model, transcribe_talk,
    _parse_word_srt, _words_to_subtitles, _write_srt,
)

log = logging.getLogger(__name__)

# Required columns in the talks CSV.
_REQUIRED_COLUMNS = {"id", "source_file", "start_time", "end_time"}


# ---- CSV validation ----

def validate_csv_columns(fieldnames: list[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = _REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")


def validate_row(row: dict, row_num: int) -> list[str]:
    """
    Return a list of error strings for a single CSV row.
    An empty list means the row is valid.
    """
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"Row {row_num}: {msg}")

    talk_id = row.get("id", "").strip()
    if not talk_id:
        err("'id' is empty")

    source = row.get("source_file", "").strip()
    if not source:
        err("'source_file' is empty")
    elif not Path(source).exists():
        err(f"source_file not found: {source!r}")

    for field in ("start_time", "end_time"):
        val = row.get(field, "").strip()
        if not val:
            err(f"'{field}' is empty")
        elif not TIMESTAMP_RE.match(val):
            err(f"'{field}' has invalid format: {val!r} (expected HH:MM:SS[.mmm])")

    return errors


# ---- Per-talk pipeline ----

def process_talk(row: dict, cfg: Config, whisper_ctx: dict | None) -> None:
    """
    Build one talk video end-to-end, filling in only the outputs that are missing.

    Outputs (both produced by default when transcription is enabled):
      - <id>.mp4        — the rendered video
      - <id>.words.srt  — word-level SRT (raw Whisper artifact; the cache key)
      - <id>.subs.srt   — Netflix-standard subtitle SRT (derived from words)

    Cache strategy:
      Each output is checked independently.  If .words.srt already exists,
      Whisper is skipped entirely and .subs.srt is derived by reading the cached
      words file — making re-runs with subtitle tweaks extremely fast.

    Steps:
      1. Determine which outputs are needed.
      2. Skip entirely if all outputs are already present.
      3. Validate talk duration and find the title card image.
      4. Build the segment recipe.
      5. If words are needed, compute audio offset and run transcription in a thread.
      6. If subs are needed but words are cached, derive them from the words file.
      7. Render the video if needed.
      8. Join the transcription thread.
    """
    talk_id = row["id"].strip()
    source = Path(row["source_file"].strip())
    start = row["start_time"].strip()
    end = row["end_time"].strip()

    output    = cfg.output / f"{talk_id}.mp4"
    words_path = cfg.output / f"{talk_id}.words.srt"
    subs_path  = cfg.output / f"{talk_id}.subs.srt"
    titles_dir = cfg.cache / "titles"

    transcribe = whisper_ctx is not None

    # --- Determine what still needs to be built ---
    need_video = not output.exists() or cfg.force
    need_words = transcribe and (not words_path.exists() or cfg.force)
    need_subs  = transcribe and not cfg.no_subtitles and (not subs_path.exists() or cfg.force)

    if not need_video and not need_words and not need_subs:
        log.info("[skip] %s — all outputs exist", talk_id)
        return

    if cfg.dry_run:
        missing = []
        if need_video:
            missing.append(output.name)
        if need_words:
            missing.append(words_path.name)
        if need_subs:
            missing.append(subs_path.name)
        log.info("[dry-run] %s — would generate: %s", talk_id, ", ".join(missing))
        return

    # --- Validate talk duration before doing any expensive work ---
    talk_dur_us = ts_to_us(end) - ts_to_us(start)
    fade_us = FADE_DURATION * 1_000_000
    if talk_dur_us <= 2 * fade_us:
        raise ValueError(
            f"Talk '{talk_id}' is too short "
            f"({talk_dur_us / 1_000_000:.1f}s) for {FADE_DURATION}s fade bookends."
        )

    # --- Find the title card image (already downloaded by the pre-flight loop) ---
    title_images = [
        p for p in titles_dir.glob(f"{talk_id}.*")
        if p.suffix.lower() != ".mp4"
    ]
    if not title_images:
        raise FileNotFoundError(f"No title card image found in {titles_dir} for '{talk_id}'")
    title_image = title_images[0]

    # --- Build the segment recipe ---
    # The title card and sponsor use pad_start/pad_end to provide xfade headroom.
    # The talk is raw — it is normalised inside the final filtergraph.
    recipe = [
        Segment(source=BLACK, pad_start=BLACK_PAD + FADE_DURATION, audio="silence"),
        Segment(source=title_image, pad_start=TITLE_DURATION + 2 * FADE_DURATION, audio="silence"),
        Segment(source=cfg.sponsor, pad_start=FADE_DURATION, pad_end=SPONSOR_HOLD + FADE_DURATION, audio="silence"),
        Segment(source=source, trim_start=start, trim_end=end, audio="source", raw=True),
        Segment(source=cfg.sponsor, pad_start=FADE_DURATION, pad_end=SPONSOR_HOLD + FADE_DURATION, audio="silence"),
        Segment(source=BLACK, pad_end=BLACK_PAD + FADE_DURATION, audio="silence"),
    ]

    # --- Transcription thread (if words need to be generated) ---
    # Runs in parallel with rendering; works directly from the source file.
    transcribe_thread: threading.Thread | None = None
    if need_words:
        offset = talk_offset(recipe, cfg.cache, cfg.vcodec)
        transcribe_thread = threading.Thread(
            target=transcribe_talk,
            args=(source, start, end, offset, words_path, whisper_ctx,
                  cfg.whisper_language, cfg.force),
        )
        transcribe_thread.start()

    # --- Render the video (only if needed) ---
    if need_video:
        assemble(recipe, output, cfg.vcodec, cfg.cache)

    # --- Wait for transcription to finish ---
    if transcribe_thread is not None:
        transcribe_thread.join()

    # --- Derive subtitle SRT from word-level data (if needed) ---
    # If need_words was True, the thread wrote the words file and we can use it.
    # If need_words was False but need_subs is True, the words file was already
    # cached — parse it and reformat without touching Whisper.
    if need_subs:
        words = _parse_word_srt(words_path)
        if words:
            subs = _words_to_subtitles(words)
            _write_srt(subs, subs_path)
            log.info(
                "[transcribing] %s — wrote %d subtitles → %s",
                talk_id, len(subs), subs_path,
            )
        else:
            log.warning("[transcribing] %s — no words available, skipping subs", talk_id)

    log.info("[done] %s", talk_id)


# ---- Entry point ----

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Assemble conference talk videos from source recordings."
    )
    parser.add_argument("--csv", type=Path, default=Path("talks.csv"),
                        help="Path to the talks CSV (default: talks.csv)")
    parser.add_argument("--sponsor", type=Path, default=Path("sponsor.mp4"),
                        help="Path to the sponsor reel (default: sponsor.mp4)")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory (default: output/)")
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
                        help="Skip all transcription (no .words.srt or .subs.srt)")
    parser.add_argument("--no-subtitles", action="store_true",
                        help="Save word-level SRT (.words.srt) but skip Netflix subtitle SRT (.subs.srt)")
    parser.add_argument("--whisper-model", type=str, default="distil-large-v3",
                        metavar="MODEL",
                        help="Whisper model name (default: distil-large-v3)")
    parser.add_argument("--whisper-language", type=str, default="en",
                        metavar="LANG",
                        help="BCP-47 language code for Whisper (default: en)")
    args = parser.parse_args()

    # ---- Logging ----
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    ffmpeg.verbose = args.verbose

    vcodec = VCODEC_SW if args.software_encode else VCODEC_HW
    log.info("[encoder] %s", vcodec)

    cfg = Config(
        csv=args.csv,
        sponsor=args.sponsor,
        output=args.output,
        cache=args.cache,
        keep_cache=args.keep_cache,
        force=args.force,
        only=args.only,
        no_notion=args.no_notion,
        vcodec=vcodec,
        dry_run=args.dry_run,
        verbose=args.verbose,
        no_transcribe=args.no_transcribe,
        no_subtitles=args.no_subtitles,
        whisper_model=args.whisper_model,
        whisper_language=args.whisper_language,
    )

    # ---- Pre-flight checks ----
    if not cfg.csv.exists():
        log.error("CSV not found: %s", cfg.csv)
        return 1
    if not cfg.sponsor.exists():
        log.error("Sponsor reel not found: %s", cfg.sponsor)
        return 1

    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.cache / "titles").mkdir(parents=True, exist_ok=True)

    # ---- Load and validate CSV ----
    with cfg.csv.open() as f:
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

    if cfg.only:
        rows = [r for r in rows if r["id"].strip() == cfg.only]
        if not rows:
            log.error("No talk with id %r found in CSV", cfg.only)
            return 1

    # ---- Set up Notion client ----
    fetcher: NotionFetcher | None = None
    if not cfg.no_notion:
        token = os.getenv("NOTION_TOKEN")
        db_id = os.getenv("NOTION_CONFTALKS_DB_ID")
        if not token or not db_id:
            log.warning(
                "NOTION_TOKEN or NOTION_CONFTALKS_DB_ID not set — "
                "will look for title cards locally in cache/titles/"
            )
        else:
            fetcher = NotionFetcher(token, db_id)

    # ---- Download all title cards before starting any rendering ----
    if not cfg.dry_run:
        titles_dir = cfg.cache / "titles"
        title_card_failures: list[tuple[str, str]] = []

        for row in rows:
            talk_id = row["id"].strip()
            log.info("[notion] downloading %s title card...", talk_id)
            try:
                if fetcher is not None:
                    fetcher.download_title_card(talk_id, titles_dir)
                else:
                    # --no-notion mode: title card must already exist locally.
                    if not any(
                        p for p in titles_dir.glob(f"{talk_id}.*")
                        if p.suffix.lower() != ".mp4"
                    ):
                        raise FileNotFoundError(
                            f"No title card found in {titles_dir} for '{talk_id}'. "
                            "Place the image there manually or remove --no-notion."
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
                    "successful talk(s)? [y/N] "
                ).strip().lower()
            except EOFError:
                answer = ""

            if answer != "y":
                log.info("[notion] aborting")
                return 1

            rows = [r for r in rows if r["id"].strip() not in failed_ids]
        else:
            log.info("[notion] all %d title card(s) ready", len(rows))

    # ---- Pre-warm Whisper model once before the main loop ----
    whisper_ctx: dict | None = None
    if not cfg.no_transcribe and not cfg.dry_run:
        whisper_ctx = load_model(cfg.whisper_model)

    # ---- Main processing loop ----
    total = len(rows)
    log.info("[run] %d talk(s)", total)
    failed: list[str] = []

    for idx, row in enumerate(rows, start=1):
        talk_id = row["id"].strip()
        log.info("[%d/%d] %s", idx, total, talk_id)
        try:
            process_talk(row, cfg, whisper_ctx)
        except Exception as exc:
            log.error("[error] %s — %s", talk_id, exc)
            failed.append(talk_id)

    # ---- Summary ----
    if failed:
        log.error("[run] %d talk(s) failed: %s", len(failed), ", ".join(failed))
        log.error("[run] cache preserved at %s for debugging", cfg.cache)
        return 1

    if not cfg.keep_cache and not cfg.dry_run:
        log.info("[run] cleaning up cache")
        shutil.rmtree(cfg.cache)

    log.info("[run] done — %d talk(s) built", total - len(failed))
    return 0
