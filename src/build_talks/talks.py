"""
Talk registry loader (source of truth for timing/source media windows).

This module loads talk windows from a CSV file keyed by talk id.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass
class TalkWindow:
    """A single talk window inside a source recording."""

    talk_id: str
    source_file: Path
    source_chunks: list[Path] | None
    start_time: str
    end_time: str

    @property
    def is_chunked(self) -> bool:
        return bool(self.source_chunks)


def _resolve_source(path: Path, where: str) -> tuple[Path, list[Path] | None, list[str]]:
    """
    Resolve a talk source path.

    Returns:
      - source_file_path to store on TalkWindow
      - optional chunk list (ordered, contiguous run)
      - warnings (non-fatal)

    Resolution rules:
      1) If exact path exists as a file, use it directly.
      2) Otherwise, treat path as a chunk prefix and match
         <prefix>-<digits>.<ext> in the same directory.
      3) Chunks are sorted by numeric index. If a gap is found,
         keep only the contiguous run and warn.
    """
    if path.is_file():
        return path, None, []

    parent = path.parent if str(path.parent) else Path(".")
    prefix = path.name
    chunk_re = re.compile(rf"^{re.escape(prefix)}-(\d+)$")

    matches: list[tuple[int, Path, str]] = []
    for candidate in parent.glob(f"{prefix}-*"):
        if not candidate.is_file():
            continue
        m = chunk_re.match(candidate.stem)
        if not m:
            continue
        matches.append((int(m.group(1)), candidate, candidate.suffix.lower()))

    if not matches:
        return path, None, []

    matches.sort(key=lambda item: item[0])
    chosen_ext = matches[0][2]
    ext_filtered = [(idx, p) for idx, p, ext in matches if ext == chosen_ext]

    warnings: list[str] = []
    if len(ext_filtered) != len(matches):
        warnings.append(
            f"{where}: mixed chunk extensions for {str(path)!r}; using only *{chosen_ext} chunks"
        )

    contiguous: list[Path] = []
    if ext_filtered:
        prev_idx, first_path = ext_filtered[0]
        contiguous.append(first_path)
        for idx, candidate in ext_filtered[1:]:
            if idx != prev_idx + 1:
                warnings.append(
                    f"{where}: chunk index gap for {str(path)!r} "
                    f"(expected {prev_idx + 1}, found {idx}); "
                    "using contiguous prefix only"
                )
                break
            contiguous.append(candidate)
            prev_idx = idx

    return path, contiguous, warnings


def load_talks(path: Path) -> tuple[dict[str, TalkWindow], set[str]]:
    """
    Load talk windows from CSV.

    Required headers:
      id,source_file,start_time,end_time

    source_file accepts either:
      - a concrete file path (single recording)
      - a chunk prefix path; matching files in the same directory using
        <prefix>-<digits>.<ext> are treated as one virtual contiguous source.
    """
    if not path.exists():
        raise ValueError(f"Talks file not found: {path}")

    required = {"id", "source_file", "start_time", "end_time"}
    talks: dict[str, TalkWindow] = {}
    fatal_errors: list[str] = []
    skipped_warnings: list[str] = []
    skipped_talk_ids: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = set(reader.fieldnames or [])
        missing = required - headers
        if missing:
            raise ValueError(
                f"Talks CSV missing required columns: {', '.join(sorted(missing))}"
            )

        for i, row in enumerate(reader, start=2):
            where = f"{path}:{i}"
            talk_id = str((row.get("id") or "")).strip()
            source_file = str((row.get("source_file") or "")).strip()
            start_time = str((row.get("start_time") or "")).strip()
            end_time = str((row.get("end_time") or "")).strip()

            if not talk_id:
                fatal_errors.append(f"{where}: missing id")
                continue
            if talk_id in talks:
                fatal_errors.append(f"{where}: duplicate id {talk_id!r}")
                continue
            if not source_file:
                skipped_warnings.append(f"{where}: missing source_file")
                skipped_talk_ids.add(talk_id)
                continue
            if not start_time:
                skipped_warnings.append(f"{where}: missing start_time")
                skipped_talk_ids.add(talk_id)
                continue
            if not end_time:
                skipped_warnings.append(f"{where}: missing end_time")
                skipped_talk_ids.add(talk_id)
                continue

            source_path = Path(source_file)
            resolved_source, source_chunks, resolve_warnings = _resolve_source(source_path, where)
            for warning in resolve_warnings:
                log.warning("[warn] %s", warning)

            if source_chunks is None and not resolved_source.is_file():
                skipped_warnings.append(f"{where}: source_file not found: {source_file!r}")
                skipped_talk_ids.add(talk_id)
                continue

            talks[talk_id] = TalkWindow(
                talk_id=talk_id,
                source_file=resolved_source,
                source_chunks=source_chunks,
                start_time=start_time,
                end_time=end_time,
            )

    for warning in skipped_warnings:
        log.warning("[skip] %s", warning)

    if fatal_errors:
        raise ValueError("Talks CSV validation failed:\n- " + "\n- ".join(fatal_errors))

    return talks, skipped_talk_ids
