# Drop Zone Intel

Turns BGMI esports **map-stream VODs** into time-stamped team positions, safe zones and alive counts, then serves a local scouting dashboard. Personal / portfolio project: no footage, frames or models are ever committed.

- Phase 1 (requirements): https://claude.ai/artifact/HBRGDw1om3PvW8jrodotWW
- Phase 2 (design): https://claude.ai/artifact/HXoxybRKwUvadKXwRrQb9v

## Status

PRD build phase 0 (setup): catalog (F1), video type detection (F2), processing queue (F4), disk guard (F5). Gate: `python cli.py catalog` lists every video with the correct tournament, stage, day and type.

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

## How the pieces work

- **Catalog**: stage names are normalized with `stage_aliases` ("Quater Finals" → Quarter Finals, "SF" → Semi Finals, "Grand Finals" → Finals). Day folders like `Day 3`, `day3`, `D3` are accepted. Files at the wrong depth are listed as skipped, never guessed. A video's id comes from its relative path + size, so moving `videos_root` to another drive keeps the ids.
- **Video type**: samples 24 frames across the video. A frame looks like the map view when the left team table is the same graphic as in several other samples and its brightness repeats at the 40 px row pitch. Enough such frames → `map_stream`. The filename is only a hint; disagreements are noted. Thresholds live in `config/overlays/*.yaml` and must be tuned on the real BMSD files.
- **Queue**: SQLite at `data\state\pipeline.db`. A crash leaves nothing half-written; `running` videos go back to `queued` on the next run. Re-cataloguing never resets finished work.
- **Safety**: nothing under `videos_root` is ever written or deleted; deletions happen only inside `data_root`; jobs refuse to start under `min_free_gb`.

## Tests

```powershell
python -m pytest -q
```

GitHub Actions runs the same tests on every push.
