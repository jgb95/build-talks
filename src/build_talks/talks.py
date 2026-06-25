"""
Talk registry loader (source of truth for timing/source media windows).

This module loads talk windows from a CSV file keyed by talk id.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass
class TalkWindow:
    """A single talk window inside a source recording."""

    talk_id: str
    source_file: Path
    start_time: str
    end_time: str


def load_talks(path: Path) -> tuple[dict[str, TalkWindow], set[str]]:
    """
    Load talk windows from CSV.

    Required headers:
      id,source_file,start_time,end_time
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
            if not source_path.exists():
                skipped_warnings.append(f"{where}: source_file not found: {source_file!r}")
                skipped_talk_ids.add(talk_id)
                continue

            talks[talk_id] = TalkWindow(
                talk_id=talk_id,
                source_file=source_path,
                start_time=start_time,
                end_time=end_time,
            )

    for warning in skipped_warnings:
        log.warning("[skip] %s", warning)

    if fatal_errors:
        raise ValueError("Talks CSV validation failed:\n- " + "\n- ".join(fatal_errors))

    return talks, skipped_talk_ids
