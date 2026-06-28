"""
Recording recipe interpreter.

This module translates recording manifest recipe JSON into timeline Segment
objects and the ordered talk windows that should be transcribed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from build_talks.config import BLACK_PAD, FADE_DURATION, TITLE_DURATION
from build_talks.segment import BLACK, Segment
from build_talks.talks import TalkWindow


@dataclass
class RecipeSpec:
    """A fully-built render recipe and metadata for downstream steps."""

    output_id: str
    segments: list[Segment]
    transcribe_talks: list[TalkWindow]


def _as_bool(value: Any, *, default: bool = False) -> bool:
    return bool(value) if value is not None else default


def _num(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected number, got {value!r}")


def _resolve_photo(
    params: dict[str, Any],
    *,
    current_talk: TalkWindow | None,
    title_image_for_talk: Callable[[str], Path],
) -> Path:
    src = params.get("src")
    from_ref = str(params.get("from", "")).strip().lower()
    if src:
        return Path(str(src))
    if from_ref == "talk":
        if current_talk is None:
            raise ValueError("photo step with from=talk requires talk context")
        return title_image_for_talk(current_talk.talk_id)
    raise ValueError("photo step requires either 'src' or 'from: talk'")


def _segment_from_step(
    step: dict[str, Any],
    *,
    current_talk: TalkWindow | None,
    title_image_for_talk: Callable[[str], Path],
) -> tuple[Segment, TalkWindow | None]:
    if len(step) != 1:
        raise ValueError(f"Each recipe step must have exactly one key: {step!r}")

    kind, raw_params = next(iter(step.items()))
    kind = str(kind).strip().lower()
    params = raw_params if isinstance(raw_params, dict) else {}

    if kind == "video":
        src = params.get("src")
        if not src:
            raise ValueError("video step requires 'src'")
        pad_start = FADE_DURATION
        pad_end = FADE_DURATION
        return (
            Segment(
                source=Path(str(src)),
                pad_start=pad_start,
                pad_end=pad_end,
                audio="silence",
            ),
            None,
        )

    if kind == "photo":
        src = _resolve_photo(
            params,
            current_talk=current_talk,
            title_image_for_talk=title_image_for_talk,
        )
        duration = _num(params.get("duration"), TITLE_DURATION)
        pad_start = duration + 2 * FADE_DURATION
        return Segment(source=src, pad_start=pad_start, audio="silence"), None

    if kind == "talk":
        if current_talk is None:
            raise ValueError("talk step requires talk context")
        seg = Segment(
            source=current_talk.source_file,
            source_chunks=current_talk.source_chunks,
            trim_start=current_talk.start_time,
            trim_end=current_talk.end_time,
            audio="source",
            raw=True,
        )
        return seg, current_talk

    raise ValueError(f"Unsupported recipe step type: {kind!r}")


def _auto_black_segment() -> Segment:
    # 0.5s held black + 1.0s fade headroom (with FADE_DURATION=1).
    # This reproduces the old explicit `{ "black": {} }` recipe bookends,
    # but now it is injected automatically in code.
    return Segment(
        source=BLACK,
        pad_start=BLACK_PAD + FADE_DURATION,
        pad_end=0.0,
        audio="silence",
    )


def _with_auto_bookends(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments
    black = _auto_black_segment()
    return [black, *segments, black]


def _is_each_step(step: dict[str, Any]) -> bool:
    if len(step) != 1:
        return False
    _, raw_params = next(iter(step.items()))
    params = raw_params if isinstance(raw_params, dict) else {}
    return _as_bool(params.get("each"), default=False)


def build(
    recording: dict[str, Any],
    *,
    talks_by_id: dict[str, TalkWindow],
    title_image_for_talk: Callable[[str], Path],
) -> list[RecipeSpec]:
    """
    Build one or more recipe specs from a recording object.

    Rules:
      - type=individual: produce one output per talk id in `talks` using output
        id format `<recording.id>_<talk_id>`.
      - type=playlist: produce one combined output using recording.id.
      - recipe steps with each:true are expanded as contiguous repeating blocks
        over talks in order.
    """
    recording_id = str(recording.get("id", "")).strip()
    recording_type = str(recording.get("type", "")).strip().lower()
    talk_ids = recording.get("talks") or []
    recipe = recording.get("recipe") or []

    talks = [talks_by_id[str(tid)] for tid in talk_ids]

    if recording_type == "individual":
        specs: list[RecipeSpec] = []
        for talk in talks:
            segments: list[Segment] = []
            transcribe: list[TalkWindow] = []
            for step in recipe:
                seg, maybe_talk = _segment_from_step(
                    step,
                    current_talk=talk,
                    title_image_for_talk=title_image_for_talk,
                )
                segments.append(seg)
                if maybe_talk is not None:
                    transcribe.append(maybe_talk)
            out_id = f"{recording_id}_{talk.talk_id}"
            specs.append(
                RecipeSpec(
                    output_id=out_id,
                    segments=_with_auto_bookends(segments),
                    transcribe_talks=transcribe,
                )
            )
        return specs

    if recording_type != "playlist":
        raise ValueError(f"Unsupported recording type: {recording_type!r}")

    segments: list[Segment] = []
    transcribe: list[TalkWindow] = []

    i = 0
    while i < len(recipe):
        step = recipe[i]
        if _is_each_step(step):
            j = i
            while j < len(recipe) and _is_each_step(recipe[j]):
                j += 1
            each_block = recipe[i:j]
            for talk in talks:
                for each_step in each_block:
                    seg, maybe_talk = _segment_from_step(
                        each_step,
                        current_talk=talk,
                        title_image_for_talk=title_image_for_talk,
                    )
                    segments.append(seg)
                    if maybe_talk is not None:
                        transcribe.append(maybe_talk)
            i = j
            continue

        seg, maybe_talk = _segment_from_step(
            step,
            current_talk=talks[0] if talks else None,
            title_image_for_talk=title_image_for_talk,
        )
        segments.append(seg)
        if maybe_talk is not None:
            transcribe.append(maybe_talk)
        i += 1

    return [
        RecipeSpec(
            output_id=recording_id,
            segments=_with_auto_bookends(segments),
            transcribe_talks=transcribe,
        )
    ]
