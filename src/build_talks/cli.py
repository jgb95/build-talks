"""
Command-line entry point (manifest + recipe driven).

This CLI reads a recordings manifest (JSON), validates it, builds a recipe per
job, renders output video(s), optionally transcribes to word-level + Netflix
subtitle SRTs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from build_talks import ffmpeg
from build_talks.config import (
    VCODEC_HW,
    VCODEC_SW,
    Config,
)
from build_talks.notion import NotionFetcher
from build_talks.recipes import build as build_recipe
from build_talks.render import assemble
from build_talks.segment import raw_offsets
from build_talks.talks import TalkWindow, load_talks
from build_talks.transcribe import (
    _parse_word_srt,
    _words_to_subtitles,
    _write_srt,
    _write_word_srt,
    load_model,
    transcribe_talk,
)

log = logging.getLogger(__name__)


def _validate_recipe_step(step: Any, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(step, dict):
        return [f"{where}: recipe step must be an object"]
    if len(step) != 1:
        return [f"{where}: recipe step must have exactly one key"]

    kind, raw_params = next(iter(step.items()))
    kind = str(kind).strip().lower()
    params = raw_params if isinstance(raw_params, dict) else {}

    if raw_params is not None and not isinstance(raw_params, dict):
        errors.append(f"{where}: step params must be an object")

    if kind not in {"photo", "video", "talk"}:
        errors.append(f"{where}: unsupported step type {kind!r}")
        return errors

    each = params.get("each")
    if each is not None and not isinstance(each, bool):
        errors.append(f"{where}: 'each' must be boolean when provided")

    allowed_keys: set[str]
    if kind == "talk":
        allowed_keys = {"each"}
    elif kind == "video":
        allowed_keys = {"src", "each"}
    else:  # photo
        allowed_keys = {"src", "from", "duration", "each"}

    unknown = sorted(k for k in params.keys() if k not in allowed_keys)
    if unknown:
        errors.append(
            f"{where}: unsupported key(s) for {kind} step: {', '.join(repr(k) for k in unknown)}"
        )

    if kind == "video":
        src = str(params.get("src", "")).strip()
        if not src:
            errors.append(f"{where}: video step requires non-empty 'src'")
        elif not Path(src).exists():
            errors.append(f"{where}: video src not found: {src!r}")

    if kind == "photo":
        src = str(params.get("src", "")).strip()
        from_ref = str(params.get("from", "")).strip().lower()
        if not src and from_ref != "talk":
            errors.append(f"{where}: photo step requires 'src' or 'from: talk'")
        if src and not Path(src).exists():
            errors.append(f"{where}: photo src not found: {src!r}")
        duration = params.get("duration")
        if duration is not None and not isinstance(duration, (int, float)):
            errors.append(f"{where}: photo duration must be numeric when provided")

    return errors


def _validate_recording(recording: dict[str, Any], where: str, seen_ids: set[str]) -> list[str]:
    errors: list[str] = []

    rec_id = str(recording.get("id", "")).strip()
    if not rec_id:
        return [f"{where}: missing/empty 'id'"]
    if rec_id in seen_ids:
        return [f"{where}: duplicate id {rec_id!r}"]
    seen_ids.add(rec_id)

    rec_type = str(recording.get("type", "")).strip().lower()
    if rec_type not in {"individual", "playlist"}:
        errors.append(f"{where} ({rec_id}): unsupported type {rec_type!r}")

    talks = recording.get("talks")
    if not isinstance(talks, list) or not talks:
        errors.append(f"{where} ({rec_id}): 'talks' must be a non-empty array of talk ids")
    else:
        for ti, talk_id in enumerate(talks, start=1):
            if isinstance(talk_id, (dict, list)):
                errors.append(
                    f"{where} ({rec_id}) talks[{ti}]: must be a talk id string, not object"
                )
                continue
            tid = str(talk_id).strip()
            if not tid:
                errors.append(f"{where} ({rec_id}) talks[{ti}]: must be a non-empty talk id")

    recipe = recording.get("recipe")
    if not isinstance(recipe, list) or not recipe:
        errors.append(f"{where} ({rec_id}): 'recipe' is required and must be a non-empty array")
    else:
        for si, step in enumerate(recipe, start=1):
            errors.extend(_validate_recipe_step(step, f"{where} ({rec_id}) recipe step #{si}"))

    return errors


def load_recordings(path: Path) -> list[dict[str, Any]]:
    """
    Load and validate the recordings manifest.

    Supported recording types:
      - individual: {type,id,talks:[...],recipe:[...]}
      - playlist: {type,id,talks:[...],recipe:[...]}
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}")

    if not isinstance(payload, list):
        raise ValueError("recordings manifest must be a JSON array of jobs")

    recordings: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, raw in enumerate(payload, start=1):
        where = f"job #{i}"
        if not isinstance(raw, dict):
            errors.append(f"{where}: must be an object")
            continue

        recording = dict(raw)
        errors.extend(_validate_recording(recording, where, seen_ids))
        recordings.append(recording)

    if errors:
        raise ValueError("Recordings validation failed:\n- " + "\n- ".join(errors))

    return recordings


def _validate_talk_references(
    recordings: list[dict[str, Any]],
    talks_by_id: dict[str, TalkWindow],
) -> list[str]:
    errors: list[str] = []
    for i, rec in enumerate(recordings, start=1):
        rec_id = str(rec.get("id", "")).strip()
        for ti, talk_id in enumerate(rec.get("talks") or [], start=1):
            tid = str(talk_id).strip()
            if tid and tid not in talks_by_id:
                errors.append(
                    f"job #{i} ({rec_id}) talks[{ti}] unknown talk id: {tid!r}"
                )
    return errors


def _recording_uses_talk_photo(recording: dict[str, Any]) -> bool:
    recipe = recording.get("recipe") or []
    for step in recipe:
        if not isinstance(step, dict) or len(step) != 1:
            continue
        kind, raw_params = next(iter(step.items()))
        if str(kind).strip().lower() != "photo":
            continue
        params = raw_params if isinstance(raw_params, dict) else {}
        if str(params.get("from", "")).strip().lower() == "talk":
            return True
    return False


def _recording_talk_ids(recording: dict[str, Any]) -> list[str]:
    return [
        str(talk_id).strip()
        for talk_id in recording.get("talks") or []
        if str(talk_id).strip()
    ]


def _title_card_path(titles_dir: Path, talk_id: str) -> Path:
    matches = [p for p in titles_dir.glob(f"{talk_id}.*") if p.suffix.lower() != ".mp4"]
    if not matches:
        raise FileNotFoundError(f"No title card image found in {titles_dir} for '{talk_id}'")
    return matches[0]


def _transcribe_recipe(
    *,
    output_stem: str,
    talks: list[TalkWindow],
    offsets: list[float],
    words_path: Path,
    subs_path: Path,
    whisper_ctx: dict | None,
    cfg: Config,
    need_words: bool,
    need_subs: bool,
) -> None:
    if not need_words and not need_subs:
        return

    if whisper_ctx is None and need_words:
        raise RuntimeError("Transcription requested but Whisper model is not loaded")

    if need_words:
        threads: list[threading.Thread] = []
        results: list[list[dict] | None] = [None] * len(talks)

        def worker(idx: int, talk: TalkWindow, offset: float) -> None:
            # Write each talk to a temporary per-talk words file then merge.
            tmp_words = cfg.cache / "words" / f"{output_stem}.{idx:02d}.{talk.talk_id}.words.srt"
            tmp_words.parent.mkdir(parents=True, exist_ok=True)
            words = transcribe_talk(
                talk.source_file,
                talk.start_time,
                talk.end_time,
                offset,
                tmp_words,
                whisper_ctx,  # type: ignore[arg-type]
                cfg.whisper_language,
                cfg.force,
            )
            if words is None:
                words = _parse_word_srt(tmp_words)
            results[idx] = words or []

        for idx, (talk, offset) in enumerate(zip(talks, offsets, strict=True)):
            t = threading.Thread(target=worker, args=(idx, talk, offset))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        merged_words: list[dict] = []
        for chunk in results:
            if chunk:
                merged_words.extend(chunk)
        merged_words.sort(key=lambda w: float(w.get("start", 0.0)))

        _write_word_srt(merged_words, words_path)
        log.info("[transcribing] %s — wrote %d merged words → %s", output_stem, len(merged_words), words_path)

    if need_subs:
        words = _parse_word_srt(words_path)
        if words:
            subs = _words_to_subtitles(words)
            _write_srt(subs, subs_path)
            log.info("[transcribing] %s — wrote %d subtitles → %s", output_stem, len(subs), subs_path)
        else:
            log.warning("[transcribing] %s — no words available, skipping subs", output_stem)


def process_job(
    job: dict[str, Any],
    cfg: Config,
    whisper_ctx: dict | None,
    talks_by_id: dict[str, TalkWindow],
) -> list[str]:
    """Process one manifest job end-to-end, returning all output ids."""
    job_id = str(job["id"]).strip()

    def _title_image_for_talk(talk_id: str) -> Path:
        if cfg.dry_run:
            # In dry-run we only need a stable placeholder path for planning;
            # existence is enforced in non-dry runs via Notion/local prefetch.
            return cfg.cache / "titles" / f"{talk_id}.dryrun.png"
        return _title_card_path(cfg.cache / "titles", talk_id)

    specs = build_recipe(
        job,
        talks_by_id=talks_by_id,
        title_image_for_talk=_title_image_for_talk,
    )

    output_ids: list[str] = []
    for spec in specs:
        output_id = spec.output_id
        output_ids.append(output_id)

        output = cfg.output / f"{output_id}.mp4"
        words_path = cfg.output / f"{output_id}.words.srt"
        subs_path = cfg.output / f"{output_id}.subs.srt"

        transcribe_enabled = not cfg.no_transcribe
        need_video = not output.exists() or cfg.force
        need_words = transcribe_enabled and (not words_path.exists() or cfg.force)
        need_subs = transcribe_enabled and not cfg.no_subtitles and (not subs_path.exists() or cfg.force)

        if not need_video and not need_words and not need_subs:
            log.info("[skip] %s — all outputs exist", output_id)
            continue

        if cfg.dry_run:
            missing = []
            if need_video:
                missing.append(output.name)
            if need_words:
                missing.append(words_path.name)
            if need_subs:
                missing.append(subs_path.name)
            log.info("[dry-run] %s — would generate: %s", output_id, ", ".join(missing))
            continue

        offsets: list[float] = []
        if need_words:
            offsets = raw_offsets(spec.segments, cfg.cache, cfg.vcodec)
            if len(offsets) != len(spec.transcribe_talks):
                raise RuntimeError(
                    f"Internal error: offsets ({len(offsets)}) != talks ({len(spec.transcribe_talks)}) for {output_id}"
                )

        render_thread: threading.Thread | None = None
        render_err: list[Exception] = []

        def _render() -> None:
            try:
                if need_video:
                    assemble(spec.segments, output, cfg.vcodec, cfg.cache)
            except Exception as exc:  # pragma: no cover - threading path
                render_err.append(exc)

        if need_video:
            render_thread = threading.Thread(target=_render)
            render_thread.start()

        _transcribe_recipe(
            output_stem=output_id,
            talks=spec.transcribe_talks,
            offsets=offsets,
            words_path=words_path,
            subs_path=subs_path,
            whisper_ctx=whisper_ctx,
            cfg=cfg,
            need_words=need_words,
            need_subs=need_subs,
        )

        if render_thread is not None:
            render_thread.join()
            if render_err:
                raise render_err[0]

        log.info("[done] %s", output_id)

    log.info("[done] %s — %d output(s)", job_id, len(output_ids))
    return output_ids


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Assemble videos from a recordings manifest using recipe types."
    )
    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("recordings.json"),
        help="Path to the recordings manifest JSON (default: recordings.json)",
    )
    parser.add_argument(
        "--talks",
        type=Path,
        default=Path("talks.csv"),
        help="Path to talks CSV registry (default: talks.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache"),
        help="Cache directory for intermediate files (default: cache/)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not delete the cache directory after a successful run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-build even if outputs already exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Process only the job with this id",
    )
    parser.add_argument(
        "--no-notion",
        action="store_true",
        help="Skip Notion; expect individual title cards in cache/titles/",
    )
    parser.add_argument(
        "--software-encode",
        action="store_true",
        help="Use libx264 (software) instead of h264_videotoolbox (hardware)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing any work",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip all transcription (no .words.srt or .subs.srt)",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Save word-level SRT but skip Netflix subtitle SRT",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="distil-large-v3",
        metavar="MODEL",
        help="Whisper model name (default: distil-large-v3)",
    )
    parser.add_argument(
        "--whisper-language",
        type=str,
        default="en",
        metavar="LANG",
        help="BCP-47 language code (default: en)",
    )
    args = parser.parse_args()

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
        recordings=args.recordings,
        talks=args.talks,
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

    if not cfg.recordings.exists():
        log.error("Recordings manifest not found: %s", cfg.recordings)
        return 1

    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.cache / "titles").mkdir(parents=True, exist_ok=True)

    try:
        jobs = load_recordings(cfg.recordings)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    if cfg.only:
        jobs = [j for j in jobs if str(j.get("id", "")).strip() == cfg.only]
        if not jobs:
            log.error("No job with id %r found in manifest", cfg.only)
            return 1

    try:
        talks_by_id = load_talks(cfg.talks)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    ref_errors = _validate_talk_references(jobs, talks_by_id)
    if ref_errors:
        log.error("Talk reference validation failed:\n- %s", "\n- ".join(ref_errors))
        return 1

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

    if not cfg.dry_run:
        titles_dir = cfg.cache / "titles"
        title_card_failures: list[tuple[str, str]] = []

        photo_talk_ids = sorted(
            {
                talk_id
                for job in jobs
                if _recording_uses_talk_photo(job)
                for talk_id in _recording_talk_ids(job)
            }
        )

        for talk_id in photo_talk_ids:
            log.info("[notion] downloading %s title card...", talk_id)
            try:
                if fetcher is not None:
                    fetcher.download_title_card(talk_id, titles_dir)
                else:
                    _title_card_path(titles_dir, talk_id)
            except Exception as exc:
                log.error("[notion] %s — failed: %s", talk_id, exc)
                title_card_failures.append((talk_id, str(exc)))

        if title_card_failures:
            failed_ids = {t for t, _ in title_card_failures}
            all_required_failed = bool(photo_talk_ids) and len(failed_ids) == len(photo_talk_ids)
            if all_required_failed:
                log.error("[notion] all %d required title card(s) failed — aborting", len(photo_talk_ids))
                return 1

            if failed_ids:
                log.warning(
                    "[notion] %d required title card(s) failed: %s",
                    len(failed_ids),
                    ", ".join(sorted(failed_ids)),
                )
                jobs = [
                    j for j in jobs
                    if not (
                        _recording_uses_talk_photo(j)
                        and any(talk_id in failed_ids for talk_id in _recording_talk_ids(j))
                    )
                ]

    whisper_ctx: dict | None = None
    if not cfg.no_transcribe and not cfg.dry_run:
        whisper_ctx = load_model(cfg.whisper_model)

    total = len(jobs)
    log.info("[run] %d job(s)", total)
    failed: list[str] = []

    for idx, job in enumerate(jobs, start=1):
        job_id = str(job["id"]).strip()
        log.info("[%d/%d] %s", idx, total, job_id)
        try:
            process_job(job, cfg, whisper_ctx, talks_by_id)
        except Exception as exc:
            log.error("[error] %s — %s", job_id, exc)
            failed.append(job_id)

    if failed:
        log.error("[run] %d job(s) failed: %s", len(failed), ", ".join(failed))
        log.error("[run] cache preserved at %s for debugging", cfg.cache)
        return 1

    if not cfg.keep_cache and not cfg.dry_run:
        log.info("[run] cleaning up cache")
        shutil.rmtree(cfg.cache)

    log.info("[run] done — %d job(s) built", total - len(failed))
    return 0
