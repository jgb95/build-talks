# build-talks

An internal tool for **Bitcoin++** that assembles polished conference videos
from raw source recordings.

The pipeline is now **manifest + recipe driven**:

- `individual` recipe: title card + sponsor + one talk
- `playlist` recipe: sponsor bookends + multiple talks in sequence

This structure is intentionally close to what a backend admin dashboard would
send as job payloads.

---

## Requirements

- **ffmpeg** in `$PATH`
- **[uv](https://docs.astral.sh/uv/)** for dependency and environment management
- A recordings manifest JSON (default: `recordings.json`)
- A talks registry CSV (default: `talks.csv`)
- Any media assets referenced by recipe steps (for example `sponsor.mp4`)
- Raw source video files referenced by the manifest
- Optional: `.env` with Notion credentials (for title card download)

By default the tool uses `h264_videotoolbox` on macOS; use
`--software-encode` to force `libx264`.

---

## Installation

```bash
cd build-talks
uv sync
```

`uv sync` creates `.venv/`, installs dependencies, and registers the
`build-talks` command.

Run commands either with an activated venv or via `uv run`:

```bash
uv run build-talks --help
```

---

## Configuration

### `.env` (optional)

```dotenv
# ---- Notion ----
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxxxxx
NOTION_CONFTALKS_DB_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional property name overrides
# NOTION_CLIPART_PROP=Clipart
# NOTION_SOCIAL_CARD_PROP=SocialCard

# Optional: base URL for relative SocialCard values in Notion
# (defaults to https://btcpp.nyc3.digitaloceanspaces.com for compatibility)
# ASSET_BASE_URL=https://assets.example.com
```

If `--no-notion` is used (or Notion credentials are missing), individual job
title cards must already exist in:

```text
cache/titles/<talk_id>.*
```

### `recordings.json`

The manifest is a JSON array of jobs.

Recipe step schema (trimmed for clarity):

- `video`: `{ "video": { "src": "...", "each"?: boolean } }`
- `photo`: `{ "photo": { "src"?: "...", "from"?: "talk", "duration"?: number, "each"?: boolean } }`
- `talk`: `{ "talk": { "each"?: boolean } }`

Notes:

- `video` no longer accepts custom hold/padding knobs; sponsor/media clips use a plain
  `FADE_DURATION` head/tail pad internally so all transitions stay consistent.
- `photo.duration` is optional and defaults to `TITLE_DURATION` (3s).
- Every transition uses a standard 1-second crossfade.
- Outputs automatically include head/tail black bookends: **0.5s held black + 1s fade**
  at the start, and 1s fade + 0.5s held black at the end.

#### `individual` job

```json
{
  "type": "individual",
  "id": "day1-talks",
  "talks": ["alice-utxo-mgmt", "bob-lightning"],
  "recipe": [
    { "photo": { "from": "talk" } },
    { "talk": {} },
    { "video": { "src": "sponsor.mp4" } }
  ]
}
```

This produces one output per talk using ids in the form:
`<job-id>_<talk-id>`.

#### `playlist` job

```json
{
  "type": "playlist",
  "id": "day1-playlist",
  "talks": ["alice-utxo-mgmt", "bob-lightning"],
  "recipe": [
    { "video": { "src": "sponsor.mp4" } },
    { "photo": { "from": "talk", "duration": 3, "each": true } },
    { "talk": { "each": true } },
    { "video": { "src": "sponsor.mp4" } }
  ]
}
```

Included sample: [`recordings.json`](./recordings.json)

#### `talks.csv`

Talk timing/source windows are defined separately in CSV and referenced by id
from the manifest jobs.

```csv
id,source_file,start_time,end_time
alice-utxo-mgmt,day1.mp4,00:10:00,00:45:30
bob-lightning,day1.mp4,00:50:00,01:25:00
```

Timestamp format: `HH:MM:SS` or `HH:MM:SS.mmm`.

`source_file` supports two modes:

1. **Single file path** (existing behavior)

   ```csv
   id,source_file,start_time,end_time
   alice-utxo-mgmt,recordings/day1.mp4,00:10:00,00:45:30
   ```

2. **Chunk prefix path** (new)

   ```csv
   id,source_file,start_time,end_time
   nairobi-main,recordings/01main1000,00:12:30,00:42:00
   ```

   The tool resolves that prefix to files in the same directory matching:

   - `recordings/01main1000-00.mp4`
   - `recordings/01main1000-01.mp4`
   - `recordings/01main1000-02.mp4`
   - ...

   and treats them as one virtual contiguous source.

   **Important:** `start_time` and `end_time` are interpreted against the virtual
   concatenated timeline (all chunks joined in index order), not against any
   individual chunk file.

---

## Usage

```bash
# Standard run (reads recordings.json and talks.csv)
uv run build-talks

# Process one job by ID
uv run build-talks --only day1-playlist

# Dry-run validation + planning output only
uv run build-talks --dry-run

# Force rebuild outputs
uv run build-talks --force

# Skip Notion (use local cache/titles/<id>.*)
uv run build-talks --no-notion

# Skip all transcription
uv run build-talks --no-transcribe

# Keep word-level subtitles but skip formatted subtitle cards
uv run build-talks --no-subtitles

# Use software encoding
uv run build-talks --software-encode
```

### CLI options

```text
--recordings PATH       Path to recordings manifest JSON        (default: recordings.json)
--talks PATH            Path to talks CSV registry              (default: talks.csv)
--output PATH           Output directory                        (default: output/)
--cache PATH            Cache directory                         (default: cache/)
--keep-cache            Don't delete cache after a successful run
--force                 Re-build even if outputs exist
--only ID               Process only the job with this ID
--no-notion             Skip Notion; use pre-placed images in cache/titles/
--software-encode       Use libx264 instead of h264_videotoolbox
--dry-run               Show what would run, without doing any work
--verbose, -v           Debug-level logging
--no-transcribe         Skip all transcription (no .words.srt or .subs.srt)
--no-subtitles          Save word-level SRT but skip Netflix subtitle SRT
--whisper-model MODEL   Whisper model name                      (default: distil-large-v3)
--whisper-language LANG BCP-47 language code                    (default: en)
```

---

## Output

Each rendered output id generates:

```text
output/
  <id>.mp4
  <id>.words.srt
  <id>.subs.srt
```

For playlist jobs, subtitle files are merged across all talks into one set of
files for the final combined video timeline.

For `individual` jobs, output ids are expanded as `<job-id>_<talk-id>`.

Cache directory (`cache/`) is removed after successful runs unless
`--keep-cache` is passed. On failure, cache is preserved for debugging.

---

## Project layout

```text
pyproject.toml
recordings.json          # Example manifest
talks.csv                # Talk registry (id, source_file, start/end)
src/
  build_talks/
    __init__.py
    __main__.py
    cli.py               # Manifest parsing and orchestration
    config.py            # Constants + Config dataclass
    ffmpeg.py            # ffmpeg helpers/probe/progress plumbing
    notion.py            # Notion title-card fetcher
    recipes.py           # Recipe builders (individual, playlist)
    render.py            # Segment assembly + final filtergraph
    segment.py           # Segment model, caching, timeline offsets
    talks.py             # CSV talk loader and validation
    transcribe.py        # Whisper + SRT generation
```
