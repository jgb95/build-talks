# build-talks

An internal tool for **Bitcoin++** that assembles polished conference talk
videos from raw source recordings. Each talk gets a title card (pulled from
Notion), a sponsor reel, crossfade transitions, and an auto-generated SRT
subtitle file — all in one command.

---

## Requirements

- **macOS** with Apple Silicon (uses `h264_videotoolbox` by default; pass
  `--software-encode` for `libx264` on any platform)
- **ffmpeg** in `$PATH`
- **[uv](https://docs.astral.sh/uv/)** for dependency and environment management
- A talks CSV and sponsor video (default to `talks.csv` and `sponsor.mp4` in
  the working directory; override with `--csv` and `--sponsor`)
- The raw source video files referenced by the CSV
- A `.env` file (or exported env vars) with your Notion credentials

---

## Installation

```bash
# 1. Clone / navigate to the project directory
cd build-talks

# 2. Create the virtual environment and install all dependencies
uv sync
```

That's it — `uv sync` creates `.venv/`, pins dependencies into `uv.lock`, and
installs the package in editable mode. Run it again any time you pull new
changes to keep the environment in sync.

Before running the tool, activate the environment in your current shell:

```bash
source /path/to/build-talks/.venv/bin/activate
```

`build-talks` will then be available in that shell session for any working
directory.

---

## Configuration

### `.env`

Create a `.env` file in your working directory (or export the variables):

```dotenv
# ---- Notion (required) ----
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxxxxx
NOTION_CONFTALKS_DB_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ---- Notion property name overrides (optional) ----
# NOTION_CLIPART_PROP=Clipart
# NOTION_SOCIAL_CARD_PROP=SocialCard

# ---- Digital Ocean Spaces — required when using --upload ----
DO_SPACES_KEY=xxxxxxxxxxxxxxxxxxxx
DO_SPACES_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ---- Digital Ocean Spaces overrides (optional) ----
# DO_SPACES_BUCKET=btcpp
# DO_SPACES_REGION=nyc3
```

### `talks.csv`

| Column        | Required | Description |
|---------------|----------|-------------|
| `id`          | ✅        | Unique talk identifier (used for output filenames) |
| `source_file` | ✅        | Path to the raw recording (relative to CWD) |
| `start_time`  | ✅        | Trim start — `HH:MM:SS` or `HH:MM:SS.mmm` |
| `end_time`    | ✅        | Trim end — `HH:MM:SS` or `HH:MM:SS.mmm` |

Extra columns are ignored.

---

## Usage

With the venv activated (see Installation), `build-talks` works from any
directory — point it at your video files with the CSV flags:

```bash
# Standard run (reads talks.csv and sponsor.mp4 from CWD)
build-talks

# Process a single talk by ID
build-talks --only my-talk-id

# Dry run — show what would be processed without doing any work
build-talks --dry-run

# Re-build even if output files already exist
build-talks --force

# Skip Notion — title card images must already be in cache/titles/<id>.*
build-talks --no-notion

# Skip subtitle generation
build-talks --no-transcribe

# Use software encoder (libx264) instead of hardware (h264_videotoolbox)
build-talks --software-encode

# Upload outputs to Digital Ocean Spaces after processing
build-talks --upload

# Debug logging
build-talks --verbose
```

### All options

```
--csv PATH            Path to the talks CSV                    (default: talks.csv)
--sponsor PATH        Path to the sponsor reel                 (default: sponsor.mp4)
--output PATH         Output directory                         (default: output/)
--cache PATH          Cache directory                          (default: cache/)
--keep-cache          Don't delete cache after a successful run
--force               Re-build even if output already exists
--only ID             Process only the talk with this ID
--no-notion           Skip Notion; use pre-placed images in cache/titles/
--software-encode     Use libx264 instead of h264_videotoolbox
--dry-run             Show what would run, without doing any work
--verbose, -v         Debug-level logging
--no-transcribe       Skip all transcription (no .words.srt or .subs.srt)
--no-subtitles        Save word-level SRT but skip Netflix subtitle SRT
--whisper-model MODEL Whisper model name                       (default: distil-large-v3)
--whisper-language LANG BCP-47 language code                   (default: en)
--upload              Upload .mp4, .words.srt, .subs.srt to Digital Ocean Spaces
```

---

## Output

```
output/
  <id>.mp4        # Final assembled video
  <id>.words.srt  # Word-level subtitle timings (unless --no-transcribe)
  <id>.subs.srt   # Netflix-style subtitle SRT (unless --no-transcribe or --no-subtitles)
```

The cache directory (default `cache/`) is deleted automatically after a
successful run unless `--keep-cache` is passed. On failure, the cache is
preserved so you can inspect intermediate files.

---

## Project layout

```
build-talks.py              # Original monolithic script (preserved for reference)
pyproject.toml              # Package metadata and dependencies
src/
  build_talks/
    __init__.py             # Package marker + __version__
    __main__.py             # Enables: python -m build_talks
    config.py               # Constants (format, timing, codecs) + Config dataclass
    ffmpeg.py               # ffmpeg subprocess helpers and probe utilities
    render.py               # Title card, sponsor, and full-talk rendering
    notion.py               # Notion title-card image downloader
    transcribe.py           # Whisper model loading + SRT generation
    cli.py                  # Argument parsing, validation, and main loop
```

---

## Development

```bash
# Editable install already covers code changes — no reinstall needed.

# Run via uv (no activation needed):
uv run build-talks --help

# Or via the module entry point:
uv run python -m build_talks --help
```
