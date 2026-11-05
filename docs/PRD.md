# PRD — BGMI Esports Map-Stream Analytics ("Drop Zone Intel")

> **For Claude Code.** This is the product requirements document for a local Windows application. Read it fully before writing code. Build phase by phase, and stop at every **gate** to show me results before continuing. Product name is a working title.

| | |
|---|---|
| Owner | Jeevan (GitHub: 3T3RN4TU5) |

| Platform | Local Windows 11 laptop (pipeline) + local web dashboard |
| Stack | Python latest (CV/ML pipeline) · Node.js + Express  + React (dashboard) |

---

## 1. Summary

BGMI esports broadcasts include a **map stream**: an observer view of the full game map that shows every team's position, the safe zone, and a sidebar with each team's alive count. Nobody publicly turns this into structured data. Today, coaches and analysts get drop spots and rotation patterns by watching VODs by hand.

This product **automatically converts map-stream VODs into a database of team positions, zones and alive counts over time**. On top of that data it builds a **scouting dashboard**: where each team drops, how and when they rotate, whether they play the zone edge or the center, where fights are likely, and how all of this links to placement. It works across many tournaments, stages and days.

**Positioning:** a scouting and pattern tool that helps humans make decisions. It is **not** an AI coach that tells teams what to do.

---

## 2. Research findings (done before writing this PRD)

### 2.1 What already exists

| Product | What it does | Gap we fill |
|---|---|---|
| [Pivaga](https://pivaga.com) | AI coaching platform for BGMI/PUBG Mobile: Map Canvas strategy planner, video analyzer of uploaded clips, stat analyzer, playstyle profile. Targets players, coaches, IGLs and casters. | Its [interactive maps](https://pivaga.com/interactivemaps) are a *manual drawing* tool with no pro-team data. |
| [Pivaga BGIS analysis page](https://www.pivaga.com/BGIS-25) | Drop table (Priority 1–3 drop spots per team) and rotation highlight write-ups for BGIS. Claims 250+ teams. | Doesn't say how the data is collected, and it looks curated by hand. We **automate** it, add **time-series positions**, and cover every match. |
| [AFK Gaming BGMI leaderboards](https://afkgaming.com/bgmi-leaderboards), [Escharts](https://escharts.com/teams/bgmi) | Team rankings, results and viewership stats. | Placement and kill numbers only. No positional or rotation data. |
| [chicken-dinner](https://pypi.org/project/chicken-dinner/0.9.0/) (PUBG PC API) | A Python wrapper for PUBG's official telemetry API, with match playback animation. | PUBG **PC** has official position telemetry. I found **no public telemetry API for BGMI / PUBG Mobile**, so broadcast computer vision is the only route to this data. That is our moat. |
| [LHM.gg](https://lhm.gg/blog/best-tools-for-pubg-mobile-spectating-observing-and-hud-management) | Observer and HUD tools for PUBG Mobile tournament organizers (scoreboard, killfeed, player panel). | For broadcast production, not analytics. Live data goes to organizers only. Shows that the observer client exposes live data, which is a possible future partnership route. |

**Conclusion:** automated, cross-tournament, positional analytics from BGMI broadcasts appears to be **unserved**. Pivaga is the closest competitor and the benchmark for presentation (drop priority tables, rotation write-ups). We beat it on automation, coverage and time-series depth.

### 2.2 Technical precedent (approaches to copy)

- **PandaScore, League of Legends minimap tracking** ([blog](https://pandascore.co/blog/league-of-legends-getting-champion-coordinates-from-the-minimap-using-deep-learning)). Standard YOLO/SSD detectors trained on one dataset **did not generalize** to unseen champions. They switched to a classifier trained on synthetic icons with GAN-made overlay noise, plus a **convolutional LSTM** that uses temporal memory to handle overlapping icons. Result: over 95% of detections within 20 px.
  → *Lessons:* generate **synthetic training data** (paste nameplates and markers onto clean map crops), and use **temporal models or tracking** to handle late-game overlap.
- **SoccerNet Game State Reconstruction 2024 winner** ([arXiv 2504.06357](https://arxiv.org/abs/2504.06357v1)). Pipeline: YOLOv5m detection, then **camera calibration** to project players onto a pitch map, then DeepSORT tracking with re-identification and **jersey number OCR**.
  → *Direct analogy:* camera calibration = our **map registration** (the map stream zooms and pans), and jersey OCR = our **nameplate OCR**.
- **Synthetic-to-real transfer for LoL minimap detection** (AAAI student abstract, [link](https://ojs.aaai.org/index.php/AAAI/article/view/42235)). More support for training on synthetic composites. I couldn't open the full text, so treat this as a pointer only.

### 2.3 Market context

- KRAFTON India reports BGMI esports at **930M+ views in 2025, up from 308.56M in 2024**, and a **BGIS 2026 prize pool of ₹4 crore** ([Tribune / PNN press release, Apr 2026](https://www.tribuneindia.com/news/business/krafton-india-esports-records-930-million-views-establishing-bgmi-esports-as-indias-definitive-mobile-esports-platform/)). These are the company's own claims.
- Target users: team coaches and analysts, IGLs, casters, content creators and esports journalists.

---

## 3. Goals and non-goals

**Goals**
1. Turn any supported map-stream VOD into clean, time-stamped team-level data, without hand-labeling each match.
2. Scale across tournaments, stages and days, using one config per broadcaster overlay.
3. Deliver a fast local dashboard for scouting any team.
4. Be portfolio-grade: clean code, tests, docs and a demo.

**Non-goals (v1)**
- Real-time analysis of live streams (VODs only; live is a v3 idea).
- Telling teams what to do ("AI coach").
- Reliable individual-player tracking in late game (team level is the priority).
- Loot, vehicles, utility and comms analysis.
- Commercial distribution of KRAFTON or broadcaster footage. Keep this a **personal/portfolio project**, and get permission before any commercial use.

---

## 4. Users and key questions

| User | Questions the product must answer |
|---|---|
| Coach / analyst | Where does Team X drop on Erangel, and how often is that contested? When do they start rotating after a zone shift? Edge or center? Which routes? |
| IGL | Which teams are likely near our drop or rotation path? Where are the usual gatekeep and choke spots for this zone? |
| Caster / content creator | Quick stats for a broadcast: "SOUL holds center 70% of late zones on Miramar." Replay of any match on a clean map. |
| Me (developer) | Is the extracted data accurate? Which videos failed, and why? |

---

## 5. Feature list

Priority: **P0** = MVP must-have · **P1** = v1 · **P2** = later

### 5.1 Ingestion and library
| # | Feature | Pri |
|---|---|---|
| F1 | Watch-folder catalog: scan `videos_root\<tournament>\<stage>\<day>\*.mp4` and read tournament, stage and day from the path (normalize "Quater", "SF", "Grand Finals" etc.) | P0 |
| F2 | Auto-detect video type (`map_stream` / `gameplay` / `unknown`) by sampling frames; the filename is only a hint | P0 |
| F3 | Auto-detect broadcaster overlay, and stop with saved sample frames when it's unknown | P0 |
| F4 | Resumable, idempotent processing queue with status per video (`queued → running → done/failed`) | P0 |
| F5 | Disk-space guard; auto-delete extracted frames after each video | P0 |
| F6 | Pair map stream + gameplay for the same day, and align their timelines per match (match start screen + audio cross-correlation) | P1 |
| F7 | `new-overlay` wizard: sample frames, auto-find the map viewport, write a draft YAML | P1 |

### 5.2 Vision pipeline
| # | Feature | Pri |
|---|---|---|
| F8 | Scene classifier: map view / full-map match start / other (desk, cams, replays, ads) | P0 |
| F9 | Match segmentation: start and end time, match number, map name | P0 |
| F10 | Safe-zone detection (white circle center + radius) with temporal smoothing | P0 |
| F11 | Blue-zone / play-area estimate (purple tint boundary) | P1 |
| F12 | Sidebar reader: alive bars, finishes and rank for 16 teams; elimination time per team | P0 |
| F13 | Map registration: frame → reference map homography (ORB/SIFT + RANSAC, overlays masked), with a confidence score | P0 |
| F14 | Plane-path detection on the match start screen (direction line), for drop context | P1 |
| F15 | Marker + nameplate detector (YOLO, trained on real + synthetic composites) | P0 |
| F16 | Team identification: plate color + OCR (EasyOCR/PaddleOCR) + fuzzy match to the tournament roster | P0 |
| F17 | Multi-object tracking (Hungarian / ByteTrack-style) with gap filling and last-known position | P0 |
| F18 | Player-level identity when the nameplate is readable | P1 |
| F19 | Synthetic data generator: composite plates and markers onto clean map crops to scale training | P1 |
| F20 | Gameplay video: kill-feed OCR, giving a fight log (who killed whom, when, weapon) | P2 |

### 5.3 Analytics
| # | Feature | Pri |
|---|---|---|
| F21 | Drop spot detection (positions in the first ~2 min, clustered with DBSCAN, named by nearest compound) + **Priority 1/2/3 drop table per team per map** | P0 |
| F22 | Drop contest rate: how often another team lands within X m | P0 |
| F23 | Rotation timing: delay from zone reveal to first move; arrive early or late | P0 |
| F24 | Rotation speed (km/h) and path; route clustering (which side or bridge they use) | P1 |
| F25 | Edge vs center ratio per zone phase | P0 |
| F26 | Clash-risk / chokepoint heatmaps from team-to-team distance matrices | P1 |
| F27 | Survival curves: alive count by phase; elimination phase per team | P0 |
| F28 | Correlation of drop, rotation style and edge/center with placement (per map) | P1 |
| F29 | Zone tendencies: where final circles land per map, heatmap across all matches | P1 |
| F30 | Team "playstyle card" (aggressive / passive, edge / center, early / late rotator), rule-based and explainable | P1 |
| F31 | Head-to-head: where two teams' paths intersect, and outcomes | P2 |
| F32 | Natural-language query over the data ("Which teams drop Pochinki on Erangel?") via local LLM / Claude API | P2 |

### 5.4 Dashboard (React + Express + MongoDB, local)
| # | Feature | Pri |
|---|---|---|
| F33 | Library page: tournaments → stages → days → matches, with processing status | P0 |
| F34 | **Match replay**: clean reference map, zone circles, team dots and trails, time slider, play/pause, team filter | P0 |
| F35 | **Team scouting page**: drop table, drop heatmap, rotation stats, edge/center, survival, playstyle card, filter by map/tournament/stage | P0 |
| F36 | Map-level heatmaps (all teams): drops, final circles, death locations | P1 |
| F37 | Compare teams side by side | P1 |
| F38 | Export: CSV/Parquet, PNG of any chart or map, PDF scouting report per team | P1 |
| F39 | Data-quality page: per-video confidence, failed frames, debug overlays | P0 |
| F40 | Jump to the video timestamp from any event (opens the local VOD at that time) | P1 |
| F41 | Share-ready "broadcast stat" cards for casters and creators | P2 |

---

## 6. Verified facts about the source (BMSD 2026 Grand Finals Day 3 map stream)

Measured on the real file. Use these as the first overlay config and as test fixtures.

- 1280x720, 59.94 fps, H.264, 6 h 15 min, 4.5 GB.
- Timeline: pre-show about 0:00–0:55 → **6 matches about 0:55–4:50** → awards. The stream regularly cuts to the desk, player cams, replays and ads.
- Day 3 schedule screen: Matches 13–18 = Rondo, Erangel, Erangel, Erangel, Miramar, Miramar.
- **Layout A, in-match map view:** left sidebar x 0–292, a 16-row team table (#, logo, TEAM tag, FIN., STATUS = 4 alive bars), rows about y 54–660, roughly 40 px apart, footer "ALIVE KNOCK ELIM."; map viewport about x 292–987; right sidebar x 987–1280 with logo, "STANDINGS · GRAND FINALS · DAY 3 - MATCH N", and the top 3 finish leaders.
- **Layout B, match-start full map:** whole map, grid letters B–H on top and row letters down the side, team logo tiles on both sidebars, "GRAND FINALS / DAY 3 - MATCH N" at the top left, and a **white plane path** line.
- **Camera zooms and pans to follow the zone**, so screen pixels are not map coordinates.
- **Zone:** a thin white circle. Outside the play area the map has a purple tint. An untuned HoughCircles on a low-saturation, high-value mask already found the circle on 4 of 5 map frames, and gave one false positive on a player-cam frame (hence F8).
- **Nameplates:** team-colored plates, with the prefix glued to the name: `WELTwGokulwnl`, `GodLSAUmay`, `NinzaApollozz`, `iQOOSoulNakul`, `BAxTermixVRI`, `K9xSaumRaj`, `GDRxJUSTIN`. In late game they overlap heavily.
- **Sidebar tags:** GENS, NBE, MYTH, GODL, NINZ, WELT, GDR, V1BA, BLEG, SOUL, RNTX, TAG, 8BIT, OR, K9, WF. The nameplate prefix differs from the tag (e.g. `BAx` = BLEG, `Nebula…` = NBE). Keep a per-tournament alias table, and confirm it with me.
- **No in-game timer is visible.** Use video time, with t=0 at the match start screen.

---

## 7. Architecture

```
               videos_root (read-only)                        dashboard (local)
 <tournament>\<stage>\<day>\*.mp4                          React (Vite) ⇄ Express API
            │                                                        ▲
            ▼                                                        │
 catalog → type detect → overlay detect → ingest (ffmpeg)    MongoDB (local or Atlas)
            │                                                        ▲
            ▼                                                        │
 classify → segment → [zone · sidebar · register · detect → teamid → track]
            │                                                        │
            ▼                                                        │
       analytics (drops, rotations, edge/center, survival) ──► Parquet ─┘
```

### 7.1 Repo layout
```
bgmi-analytics/
  settings.yaml                     # paths (any drive), ffmpeg, thresholds
  config/
    overlays/kie_bmsd2026.yaml      # pixel boxes, colors per broadcaster layout
    overlays/_refs/                 # reference crops for overlay auto-detection
    maps/erangel.png miramar.png rondo.png sanhok.png vikendi.png
    maps/compounds/<map>.geojson    # named compounds (Pochinki, School...) for naming drops
    teams/<tournament>.yaml         # roster, tags, nameplate aliases, colors
  bgmi/
    catalog.py  videotype.py  overlay_detect.py  ingest.py
    classify.py segment.py  zone.py  sidebar.py  register.py
    detect.py   synth.py   teamid.py  track.py
    analytics/  drops.py rotations.py positioning.py survival.py playstyle.py
    store.py    quality.py  debug.py
  cli.py                            # typer CLI
  training/                         # labeling export, YOLO configs, notebooks
  dashboard/
    server/  (Express, Mongoose)
    web/     (React + Vite + TypeScript, map canvas/SVG, Recharts)
  tests/     (pytest + saved frame fixtures; jest for API)
  docs/      (README, architecture, how to add an overlay)
```

### 7.2 Settings (all paths configurable — the project may live on any drive)
```yaml
videos_root:  "X:\\bgmi-videos"
project_root: "X:\\bgmi-analytics"
data_root:    "X:\\bgmi-analytics\\data"
ffmpeg_bin:   "C:\\Users\\Hp\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-9.0.2-full_build\\bin"
sample_every_s: 3
keep_frames: false
min_free_gb: 5
mongo_uri: "mongodb://localhost:27017/bgmi"
```

### 7.3 Data model (Parquet → MongoDB collections)
| Table | Key fields |
|---|---|
| `videos` | video_id, path, tournament, stage, day, video_type, overlay, duration_s, fps, status, error |
| `matches` | match_id, video_id, match_no, map, start_s, end_s, plane_path (x1,y1,x2,y2 in map coords) |
| `frames` | match_id, t_s, scene, reg_ok, reg_conf, H (3x3) |
| `zone_timeline` | match_id, t_s, cx, cy, r (map coords, 0–1), phase, conf |
| `alive_timeline` | match_id, t_s, team, alive (0–4), finishes, rank, conf |
| `positions` | match_id, t_s, team, player?, map_x, map_y, dist_to_zone_center, in_zone, conf, track_id |
| `team_match_metrics` | match_id, team, drop_spot, drop_contested, rotation_delay_s[], avg_speed_kmh, edge_ratio, elim_phase, placement |
| `teams` | tournament, tag, org_name, aliases[], plate_color, roster[] |

---

## 8. Phases, gates and acceptance criteria

| Phase | Scope (features) | Gate (must pass to continue) |
|---|---|---|
| **0 — Setup** (wk 1) | Install Python latest + venv, PyTorch with CUDA, repo, settings, CLI skeleton, catalog (F1, F2, F4, F5) | `python cli.py catalog` lists every video with correct tournament, stage, day and type |
| **1 — Ingest + zone + sidebar** (wk 1–3) | F3, F8, F9, F10, F12, F39 (basic) | Classifier ≥ 98% on 200 labelled frames · 6 matches found in Day 3, boundaries within ±30 s · zone on ≥ 95% of map frames in Match 15 · alive counts match my manual check at 10 timestamps |
| **2 — Map registration** (wk 4–6) | F13, F14, F11 | Error < 1% of map width on 50 hand-checked frames incl. zoomed late game · low-confidence frames rejected, never guessed |
| **3 — Detection, team ID, tracking** (wk 7–10) | F15, F16, F17, F19, F18 (stretch) | ≥ 85% of alive teams located per mid-game frame · team ID ≥ 90% correct · trained on ≥ 3 overlays once available |
| **4 — Analytics** (wk 11–13) | F21–F25, F27, then F26, F28–F30 | Drop table and rotation calls match my manual VOD review for 2 matches |
| **5 — Dashboard** (wk 14–16) | F33–F35, F39, then F36–F38, F40 | A friend can scout one team without help · replay runs smoothly for a 30-min match |
| **6 — Scale** (wk 17+) | All BMSD days, more tournaments, F6, F7 | New overlay added in ≤ 1 day using the wizard |
| **7 — Gameplay** (later) | F20, link fights to map positions | Kill feed parsed ≥ 90% correct on 1 match |

---

## 9. Non-functional requirements

- **Speed (local):** keyframe-only decode for scanning; 1 frame / 3 s for matches; multiprocessing (16 threads); GPU batching for YOLO. Target: a 6-hour VOD fully processed in **under 60 minutes** on the RTX 3050.
- **Accuracy over coverage:** every value carries a confidence score. Drop low-confidence values; never invent.
- **Reproducible:** pinned `requirements.txt` / `package-lock.json`; models versioned in `data/models/` with metadata.
- **Config-driven:** no pixel box, color or threshold hard-coded; everything in YAML.
- **Safe with my files:** never modify, move or delete anything under `videos_root`. Ask before deleting outside `data_root`. Ask before system-wide installs other than Python.
- **Observability:** structured logs per video; `data/debug/` overlays (circle, sidebar boxes, detections, registered map) for a sample of frames.
- **Tests:** pytest per module using saved frames with expected values; API tests with jest; CI-ready.
- **Docs:** README with setup, "add a new tournament overlay" guide, architecture diagram, and demo GIF.

---

## 10. Machine and environment

- Windows 11, PowerShell · AMD Ryzen 7 7840HS (16 threads) · 16 GB RAM · **NVIDIA RTX 3050 Laptop 6 GB** (use CUDA).
- Disk is tight (C: ~37 GB free, D: ~13 GB free; one 6 h VOD ≈ 4.5 GB). Recommend an external drive for videos; check free space before each job.
- Installed: ffmpeg 9.0.2 (winget, path above), Node.js 24, Git. **Not installed:** Python.
- Video folder layout:
  ```
  <videos_root>\
    BMSD 2026\
      Survival\Day 1\        (map stream .mp4 + gameplay .mp4 together)
      Quarter Finals\Day 1\  Day 2\
      Semi Finals\Day 1\ ...
      Finals\Day 1\  Day 2\  Day 3\
    <Another Tournament>\ ...
  ```

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Zoom/pan breaks registration | Mask overlays; use grid lines and place names as anchors; smooth the camera path over time; reject low-confidence frames |
| Detector doesn't generalize to new overlays or teams (the PandaScore lesson) | Synthetic composites (F19); detect "marker/plate" generically, then identify team separately by color + OCR |
| Late-game nameplate overlap | Team-level only; temporal tracking with gap filling; report confidence |
| Few matches per day | Process all stages and days and many tournaments |
| Broadcaster changes overlay mid-event | Overlay detection per segment, not per file |
| Disk space | Frame cleanup, streaming decode, external drive |
| Footage rights | Portfolio use only; seek permission or partnership (KIE / organizers / observer-data route) before commercial use |

---

## 12. Future ideas (post-v1)

- Live mode: process a live map stream with a ~10 s delay for analysts during scrims.
- Scrim support: teams upload their own observer recordings.
- Partnership route: official observer data feed from organizers instead of CV.
- Predictive models: zone end-location likelihood per map; drop-clash probability.
- Multi-language caster cards (Hindi / English).

---

## 13. Instructions for Claude Code

1. Start with **Phase 0** and then **Phase 1 on Match 15** (Erangel, about 2:05–2:35 into the BMSD Day 3 video).
2. Ask me for anything in §14 before you need it, not after.
3. After each phase, show: the gate metrics, 5–10 debug images, and a short list of what's next.
4. Keep commits small and messages clear. Push to GitHub as `Jeevan2410/bgmi-analytics` when I confirm.
5. Prefer simple, explainable methods first (OpenCV rules); add ML only where rules fail.

## 14. Inputs needed from me

- [ ] Final paths for `videos_root`, `project_root`, `data_root` (which drive)
- [ ] OK to install Python latest + packages (~3 GB with PyTorch CUDA)
- [ ] Reference images for Erangel, Miramar, Rondo (clean top-down)
- [ ] Confirm team alias table per tournament
- [ ] ~1 hour for Phase 1 manual checks; ~2–3 hours to correct pre-labels in Phase 3

---

### Sources
- [Pivaga](https://pivaga.com) · [Pivaga interactive maps](https://pivaga.com/interactivemaps) · [Pivaga BGIS analysis](https://www.pivaga.com/BGIS-25)
- [AFK Gaming BGMI leaderboards](https://afkgaming.com/bgmi-leaderboards) · [Escharts BGMI teams](https://escharts.com/teams/bgmi)
- [chicken-dinner (PUBG API wrapper)](https://pypi.org/project/chicken-dinner/0.9.0/)
- [LHM.gg PUBG Mobile observing tools](https://lhm.gg/blog/best-tools-for-pubg-mobile-spectating-observing-and-hud-management)
- [PandaScore — LoL minimap coordinates with deep learning](https://pandascore.co/blog/league-of-legends-getting-champion-coordinates-from-the-minimap-using-deep-learning)
- [From Broadcast to Minimap — SoccerNet GSR (arXiv 2504.06357)](https://arxiv.org/abs/2504.06357v1)
- [Synthetic-to-Real Transfer for LoL Minimap Detection (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/42235)
- [KRAFTON India esports 930M views (Tribune, PNN)](https://www.tribuneindia.com/news/business/krafton-india-esports-records-930-million-views-establishing-bgmi-esports-as-indias-definitive-mobile-esports-platform/)
