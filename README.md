# Drop Zone Intel

Turns BGMI esports **map-stream VODs** into time-stamped team positions, safe zones and alive counts, then serves a local scouting dashboard. Personal / portfolio project: no footage, frames or models are ever committed.

- Phase 1 (requirements): https://claude.ai/artifact/HBRGDw1om3PvW8jrodotWW
- Phase 2 (design): https://claude.ai/artifact/HXoxybRKwUvadKXwRrQb9v

## Status

- PRD build phase 0 (setup), done: catalog (F1), video type detection (F2), processing queue (F4), disk guard (F5).
- PRD build phase 1 (ingest + zone + team table), in progress: scene classifier (F8), match segmentation (F9), safe-zone circle (F10), team table reader (F12), basic data quality (F39). Gate: classifier >= 98% on 200 labelled frames, 6 matches in Day 3 within +-30 s, zone on >= 95% of Match 15 map frames, alive counts match a manual check at 10 timestamps.

## Folder layout on the laptop

```
D:\FullStack\BGMI\
  videos\           read-only: <tournament>\<stage>\<day>\*.mp4
  data\             pipeline output, logs, debug (not in git)
  bgmi-analytics\   this repo
```

## Setup (Windows 11, PowerShell)

1. Install Python 3.14 from python.org (or 3.13 if PyTorch has no CUDA build for 3.14 yet).
2. Clone the repo into `D:\FullStack\BGMI\bgmi-analytics`:
   ```powershell
   cd D:\FullStack\BGMI
   git clone https://github.com/3T3RN4TU5/BGMI.git bgmi-analytics
   cd bgmi-analytics
   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
   ```
3. Check `settings.yaml` (created from `settings.example.yaml`; git-ignored).
4. Run:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python cli.py doctor     # Python, CUDA GPU, ffmpeg, paths, disk
   python cli.py catalog    # scan videos and detect their type
   python cli.py status     # processing queue
   ```

## Commands

| Command | What it does |
| --- | --- |
| `doctor` | Checks Python, PyTorch + CUDA, ffmpeg, paths, free disk, overlay configs |
| `catalog [--redetect] [--no-detect]` | Scans `videos_root`, reads tournament/stage/day from the path, probes each file and detects `map_stream` / `gameplay` / `unknown` |
| `status` | Lists every video and its status (`queued → running → done / failed`) |
| `requeue [VIDEO_ID]` | Puts failed videos back in the queue |
| `diskcheck` | Exits 1 when the data drive is below `min_free_gb` |
| `calibrate IMAGE --tags "..." [--fins "..."] [--match-no N]` | Learns team-tag, digit and match-number templates from one labelled map-view frame. Saved to `data\calib\<overlay>\glyphs.npz` (not in git) |
| `analyze VIDEO_ID [--start 2:05:00 --end 2:40:00] [--every 3]` | Phase 1 run: scenes, matches, zone and team table. Writes `data\out\<video_id>\` (frames/teams Parquet, matches.json, quality.json) and debug images to `data\debug\<video_id>\phase1\` |
| `scenes VIDEO_ID [--n 200] [--score]` | Scene-classifier check: contact sheets + a CSV to correct by hand, then `--score` prints the accuracy |

## How the pieces work

- **Catalog**: stage names are normalized with `stage_aliases` ("Quater Finals" → Quarter Finals, "SF" → Semi Finals, "Grand Finals" → Finals). Day folders like `Day 3`, `day3`, `D3` are accepted. Files at the wrong depth are listed as skipped, never guessed. A video's id comes from its relative path + size, so moving `videos_root` to another drive keeps the ids.
- **Video type**: samples 24 frames across the video. A frame looks like the map view when the left team table is the same graphic as in several other samples and its brightness repeats at the 40 px row pitch. Enough such frames → `map_stream`. The filename is only a hint; disagreements are noted. Thresholds live in `config/overlays/*.yaml` and must be tuned on the real BMSD files.
- **Scene** (map view / match start / other): a few fixed overlay graphics are solid lime only in one layout (the team-table header, the finish-leaders bar, the 16 tag strips of the match-start screen). Probe boxes and colours are in the overlay YAML.
- **Team table**: 16 rows, 40.3 px apart, sorted by standings, so the order changes. Team tags and FIN. digits are matched against templates learned with `calibrate`; each tag is used once per frame. Alive = the 4 STATUS bars: white alive (height = health), red knocked, grey dead. A digit or tag that was never learned is left empty, never guessed.
- **Zone**: the white ring is found with a top-hat filter (thin bright lines) and a RANSAC circle fit, scored by how much of the visible ring is covered. Values are screen pixels; map coordinates come with map registration (PRD build phase 2).
- **Matches**: the right panel shows "MATCH N"; a new match starts at a match-start screen or when that number changes after a cut away from the map. Numbers are read when learned, otherwise counted from `config/tournaments/*.yaml`, which also gives the map.
- **Queue**: SQLite at `data\state\pipeline.db`. A crash leaves nothing half-written; `running` videos go back to `queued` on the next run. Re-cataloguing never resets finished work.
- **Safety**: nothing under `videos_root` is ever written or deleted; deletions happen only inside `data_root`; jobs refuse to start under `min_free_gb`.

## Tests

```powershell
python -m pytest -q
```

GitHub Actions runs the same tests on every push.
