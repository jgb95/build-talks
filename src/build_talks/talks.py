"""
Talk registry loader (source of truth for timing/source media windows).

This module loads talk windows from a CSV file keyed by talk id.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TalkWindow:
    """A single talk window inside a source recording."""

    talk_id: str
    source_file: Path
    start_time: str
    end_time: str


def load_talks(path: Path) -> dict[str, TalkWindow]:
    """
    Load talk windows from CSV.

    Required headers:
      id,source_file,start_time,end_time
    """
    if not path.exists():
        raise ValueError(f"Talks file not found: {path}")

    required = {"id", "source_file", "start_time", "end_time"}
    talks: dict[str, TalkWindow] = {}
    errors: list[str] = []

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
                errors.append(f"{where}: missing id")
                continue
            if talk_id in talks:
                errors.append(f"{where}: duplicate id {talk_id!r}")
                continue
            if not source_file:
                errors.append(f"{where}: missing source_file")
                continue
            if not start_time:
                errors.append(f"{where}: missing start_time")
                continue
            if not end_time:
                errors.append(f"{where}: missing end_time")
                continue

            source_path = Path(source_file)
            if not source_path.exists():
                errors.append(f"{where}: source_file not found: {source_file!r}")
                continue

            talks[talk_id] = TalkWindow(
                talk_id=talk_id,
                source_file=source_path,
                start_time=start_time,
                end_time=end_time,
            )

    if errors:
        raise ValueError("Talks CSV validation failed:\n- " + "\n- ".join(errors))

    return talks
