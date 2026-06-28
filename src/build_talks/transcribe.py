"""
Whisper-based transcription and SRT subtitle generation.

Whisper is loaded once per run and reused across all talks.
Subtitles follow Netflix subtitle standards:
  - Max 2 lines per card
  - Max 42 characters per line
  - Min 0.833s / Max 7s display duration
  - Min 80 ms gap between captions
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from build_talks.ffmpeg import concat_list_file

log = logging.getLogger(__name__)

# ---- Netflix subtitle standards ----
MAX_CHARS = 42
MAX_LINES = 2
MIN_DURATION = 5 / 6     # 0.833s — 20 frames at 24 fps
MAX_DURATION = 7.0       # seconds
MIN_GAP = 0.080          # 80 ms minimum gap between subtitles
PAUSE_BREAK = 0.300      # natural pause gap that warrants a subtitle break

# Seconds between heartbeat log lines during transcription
_PROGRESS_INTERVAL = 30


def load_model(model_name: str) -> dict:
    """
    Load the Whisper model and return it in a context dict.

    Automatically selects the best available device:
      - "cuda"  if a CUDA GPU is available
      - "cpu"   otherwise (Apple Silicon included — CTranslate2 doesn't support MPS)

    Uses BatchedInferencePipeline for parallel chunk processing, which gives a
    significant speedup on multi-core CPUs (e.g. Apple Silicon) by processing
    multiple VAD-segmented audio chunks simultaneously.

    Returns a dict with keys:
      "model":    the loaded WhisperModel (underlying model)
      "pipeline": BatchedInferencePipeline wrapping the model
      "name":     model name string
      "device":   device string used
    """
    import os
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            cpu_threads = 0  # not used on CUDA
        else:
            # CTranslate2 doesn't support MPS; use all available CPU cores on Apple Silicon
            device = "cpu"
            compute_type = "int8"
            cpu_threads = os.cpu_count() or 4
    except ImportError:
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
    source: Path,
    start: str,
    end: str,
    intro_offset: float,
    words_path: Path,
    whisper_ctx: dict,
    language: str,
    force: bool = False,
    source_chunks: list[Path] | None = None,
    cache_dir: Path | None = None,
) -> list[dict] | None:
    """
    Transcribe the talk audio from *source* (trimmed to start/end) and write
    a word-level SRT to *words_path*.

    The word-level SRT is the primary Whisper artifact — one subtitle entry per
    word, with raw timestamps offset by *intro_offset* to align with the final
    rendered video.  The formatted (Netflix-standard) subtitle file is derived
    from this data by the caller via _words_to_subtitles() + _write_srt(); it
    does not need Whisper at all if *words_path* already exists on disk.

    Pipeline:
      1. Extract trimmed audio to a temporary 16 kHz mono WAV.
      2. Transcribe with faster-whisper using VAD filtering and word timestamps.
      3. Collect word-level timestamps and apply intro_offset.
      4. Write word-level SRT to *words_path*.

    Returns the list of word dicts on success, or None on failure.
    Failures are logged but do not raise so the overall build pipeline continues.

    Args:
        source:       Path to the source recording.
        start:        HH:MM:SS[.mmm] start timestamp of the talk in the source file.
        end:          HH:MM:SS[.mmm] end timestamp of the talk in the source file.
        intro_offset: Seconds to add to every timestamp (accounts for title/sponsor clips).
        words_path:   Destination path for the word-level .srt file.
        whisper_ctx:  Context dict returned by load_model().
        language:     BCP-47 language code (e.g. "en").
        force:        If True, overwrite an existing file.
        source_chunks: Optional ordered chunk files to treat as one virtual input.
        cache_dir:    Required when source_chunks is provided (for concat list file).
    """
    if words_path.exists() and not force:
        log.debug("[transcribing] %s — words SRT already exists, skipping Whisper", words_path.stem)
        return None  # signal: use cached file instead

    try:
        # Compute talk duration for logging
        h, m, s_str = start.split(":")
        start_s = int(h) * 3600 + int(m) * 60 + float(s_str)
        h2, m2, s2_str = end.split(":")
        end_s = int(h2) * 3600 + int(m2) * 60 + float(s2_str)
        audio_dur = end_s - start_s

        log.info("[transcribing] %s — %.0fs audio", words_path.stem, audio_dur)

        # ---- 1. Extract trimmed audio to a temp WAV ----
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Compute duration string HH:MM:SS.mmm for ffmpeg -t
            total_s = audio_dur
            dur_h = int(total_s // 3600)
            dur_m = int((total_s % 3600) // 60)
            dur_s = total_s % 60
            duration_str = f"{dur_h}:{dur_m:02d}:{dur_s:06.3f}"

            input_args: list[str]
            if source_chunks:
                if cache_dir is None:
                    raise ValueError("cache_dir is required when source_chunks is provided")
                concat_list = concat_list_file(source_chunks, cache_dir)
                input_args = ["-f", "concat", "-safe", "0", "-i", str(concat_list)]
            else:
                input_args = ["-i", str(source)]

            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", start,
                    "-t", duration_str,
                    *input_args,
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
            #   considers it a segment boundary.
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
                temperature=0.0,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            # BatchedInferencePipeline processes audio in large batches internally
            # and emits segments in bulk — per-segment progress tracking is
            # meaningless. Log a heartbeat every _PROGRESS_INTERVAL seconds so the
            # user knows the process hasn't hung.
            segments: list = []
            t_last_log = t0

            def _heartbeat_log() -> None:
                now = time.monotonic()
                nonlocal t_last_log
                if now - t_last_log >= _PROGRESS_INTERVAL:
                    wall = int(now - t0)
                    log.info("[transcribing] %s — working... (%ds)", words_path.stem, wall)
                    t_last_log = now

            for seg in segments_iter:
                segments.append(seg)
                _heartbeat_log()

            elapsed_total = time.monotonic() - t0
            log.info("[transcribing] %s — done (%.0fs)", words_path.stem, elapsed_total)

        finally:
            tmp_path.unlink(missing_ok=True)

        # ---- 3. Collect word-level timestamps ----
        # Words are relative to the start of the trimmed WAV (i.e. start at 0),
        # so we only need to add intro_offset to align with the final video.
        words: list[dict] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": w.start + intro_offset,
                    "end": w.end + intro_offset,
                })

        # ---- 4. Write word-level SRT ----
        _write_word_srt(words, words_path)
        log.info(
            "[transcribing] %s — wrote %d words → %s",
            words_path.stem, len(words), words_path,
        )
        return words

    except Exception as exc:
        log.error("[transcribing] %s — failed: %s", words_path.stem, exc)
        return None


def _words_to_subtitles(words: list[dict]) -> list[tuple[float, float, str]]:
    """
    Convert a flat list of word-timestamp dicts into subtitle segments that
    comply with Netflix Timed Text style constraints:

      - Max MAX_CHARS characters per line
      - Max MAX_LINES lines per card (2)
      - Min MIN_DURATION seconds duration
      - Max MAX_DURATION seconds duration
      - Min MIN_GAP seconds gap between consecutive subtitles
      - Break at natural pauses >= PAUSE_BREAK seconds between words

    Each word dict must have "word", "start", and "end" keys.
    Words missing timing data are skipped.

    Returns a list of (start_s, end_s, text) tuples ready for _write_srt().
    """
    segments: list[tuple[float, float, str]] = []
    if not words:
        return segments

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

        start = seg_start
        end = seg_end or seg_start

        # Enforce minimum duration
        if end - start < MIN_DURATION:
            end = start + MIN_DURATION

        # Enforce minimum gap from the previous subtitle's end
        if segments:
            prev_sub_end = segments[-1][1]
            if start < prev_sub_end + MIN_GAP:
                start = prev_sub_end + MIN_GAP
                if end <= start:
                    end = start + MIN_DURATION

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
        if prev_end is not None and (w_start - prev_end) >= PAUSE_BREAK:
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

        if len(candidate) <= MAX_CHARS:
            target_line.append(word_text)
        elif not line2 and len(word_text) <= MAX_CHARS:
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
        if seg_start is not None and (w_end - seg_start) >= MAX_DURATION:
            _flush()

    _flush()  # emit any remaining words
    return segments


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


def _write_word_srt(words: list[dict], path: Path) -> None:
    """
    Write a word-level SRT file — one subtitle entry per word.

    Each word uses its raw Whisper start/end timestamps (already offset-adjusted).
    Words missing text or timing are skipped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        idx = 1
        for w in words:
            text = w.get("word", "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            if not text or w_start is None or w_end is None:
                continue
            fh.write(f"{idx}\n")
            fh.write(f"{_srt_timestamp(w_start)} --> {_srt_timestamp(w_end)}\n")
            fh.write(f"{text}\n\n")
            idx += 1


def _parse_word_srt(path: Path) -> list[dict]:
    """
    Read a word-level SRT file (written by _write_word_srt) back into the same
    ``[{"word": str, "start": float, "end": float}]`` format that transcribe_talk()
    returns.  This lets the caller derive the Netflix-standard subs from a cached
    .words.srt without re-running Whisper.

    SRT timestamp format: HH:MM:SS,mmm
    Malformed entries are silently skipped.
    """
    words: list[dict] = []
    if not path.exists():
        return words

    def _ts_to_sec(ts: str) -> float:
        """Parse HH:MM:SS,mmm → float seconds."""
        ts = ts.replace(",", ".")
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    text = path.read_text(encoding="utf-8")
    # SRT blocks are separated by blank lines; each block is: index\ntimecodes\ntext
    for block in text.strip().split("\n\n"):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        if len(lines) < 3:
            continue
        try:
            # lines[0] = index (ignored), lines[1] = timecodes, lines[2] = word
            start_str, end_str = lines[1].split(" --> ")
            words.append({
                "word": lines[2],
                "start": _ts_to_sec(start_str.strip()),
                "end": _ts_to_sec(end_str.strip()),
            })
        except (ValueError, IndexError):
            continue

    return words
