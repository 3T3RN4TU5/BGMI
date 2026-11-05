"""PRD build phase 1 pipeline: classify -> zone + team table -> segment -> files.

Reads frames straight from ffmpeg (nothing extracted to disk), and writes to
data_root/out/<video_id>/:
  frames.parquet   one row per sampled frame: scene, zone, match number
  teams.parquet    one row per frame x table row: rank, team, finishes, alive
  matches.json     match boundaries, numbers and maps
  quality.json     per-video confidence summary (F39 basic)
and annotated debug images to data_root/debug/<video_id>/phase1/.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd

from .diskguard import assert_writable
from .scene import classify_scene
from .segment import FrameObs, match_sig, number_matches, segment
from .sidebar import GlyphBook, RowRead, read_sidebar
from .zone import ZoneResult, detect_zone, flag_jumps


def glyph_path(data_root: Path, overlay_id: str) -> Path:
    return Path(data_root) / "calib" / overlay_id / "glyphs.npz"


@dataclass
class FrameRecord:
    t: float
    scene: str
    scene_conf: float
    zone: ZoneResult | None = None
    rows: list[RowRead] = field(default_factory=list)
    sig: np.ndarray | None = None


def analyze_frame(t: float, frame: np.ndarray, overlay: dict[str, Any], book: GlyphBook) -> FrameRecord:
    sc = classify_scene(frame, overlay)
    rec = FrameRecord(t, sc.scene, sc.confidence)
    if sc.scene == "map_view":
        rec.zone = detect_zone(frame, overlay, "map_view")
        rec.rows = read_sidebar(frame, overlay, book)
        rec.sig = match_sig(frame, overlay, "map_view")
    elif sc.scene == "match_start":
        rec.zone = detect_zone(frame, overlay, "match_start")
    return rec


# ---- debug drawing ---------------------------------------------------------

def draw_debug(frame: np.ndarray, rec: FrameRecord, overlay: dict[str, Any], match_no: int | None) -> np.ndarray:
    img = frame.copy()
    head = f"t={_hms(rec.t)}  {rec.scene} ({rec.scene_conf:.2f})" + (f"  match {match_no}" if match_no else "")
    if rec.zone is not None:
        z = rec.zone
        if z.found:
            cv2.circle(img, (int(z.cx), int(z.cy)), int(z.r), (255, 0, 255), 2)
            cv2.drawMarker(img, (int(z.cx), int(z.cy)), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
            head += f"  zone r={z.r:.0f} conf {z.confidence:.2f}"
        else:
            head += "  zone: none"
    _label(img, head, (300, 22), scale=0.55)
    if rec.rows:
        sb = overlay["layouts"]["map_view"]["sidebar_left"]
        for r in rec.rows:
            y = int(round(float(sb["first_row_y"]) + float(sb["row_h"]) * (r.rank - 1))) + 27
            bars = "".join({"alive": "A", "knocked": "K", "dead": "-", "unknown": "?"}[s] for s in r.bars)
            txt = f"{r.tag or '??':<4} fin {'?' if r.fin is None else r.fin} alive {'?' if r.alive is None else r.alive} [{bars}]"
            color = (0, 255, 0) if r.confidence >= 0.8 else (0, 165, 255) if r.confidence >= 0.5 else (0, 0, 255)
            _label(img, txt, (296, y), scale=0.42, color=color)
    return img


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
           color: tuple[int, int, int] = (255, 255, 255)) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _hms(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---- the run ---------------------------------------------------------------

@dataclass
class RunResult:
    out_dir: Path
    debug_dir: Path
    frames: pd.DataFrame
    teams: pd.DataFrame
    matches: list[dict[str, Any]]
    quality: dict[str, Any]


def run(settings, video: dict[str, Any], overlay: dict[str, Any], book: GlyphBook,
        frames: Iterable[tuple[float, np.ndarray]], schedule: dict[str, Any] | None,
        debug_every: int = 20, max_debug: int = 60,
        from_video_start: bool = True, first_match: int | None = None,
        progress: Callable[[float], None] | None = None) -> RunResult:
    vid = video["video_id"]
    out_dir = Path(settings.data_root) / "out" / vid
    dbg_dir = Path(settings.debug_dir) / vid / "phase1"
    for d in (out_dir, dbg_dir):
        assert_writable(settings, d)
        d.mkdir(parents=True, exist_ok=True)

    recs: list[FrameRecord] = []
    pending: list[tuple[FrameRecord, np.ndarray]] = []   # frames kept for debug images
    n_map = 0
    for t, frame in frames:
        rec = analyze_frame(t, frame, overlay, book)
        recs.append(rec)
        low = (rec.scene == "map_view" and (not (rec.zone and rec.zone.found)
                                            or any(r.confidence < 0.5 for r in rec.rows)))
        if rec.scene == "map_view":
            n_map += 1
        if (rec.scene != "other" and (n_map % max(debug_every, 1) == 1 or rec.scene == "match_start" or low)
                and len(pending) < max_debug):
            pending.append((rec, frame.copy()))
        if progress:
            progress(t)

    # zones: lower confidence of rings unlike their neighbours (map-view frames in order)
    mv = [r for r in recs if r.scene == "map_view"]
    for r, z in zip(mv, flag_jumps([r.zone for r in mv])):
        r.zone = z

    segs = segment([FrameObs(r.t, r.scene, r.sig) for r in recs])
    number_matches(segs, book, schedule, from_video_start, first_match)
    kept = [s for s in segs if s.kept]

    def match_of(t: float) -> int | None:
        for s in kept:
            if s.start_t <= t <= s.end_t:
                return s.number
        return None

    frame_rows, team_rows = [], []
    for r in recs:
        z = r.zone
        frame_rows.append({
            "video_id": vid, "t": r.t, "scene": r.scene, "scene_conf": r.scene_conf, "match_no": match_of(r.t),
            "zone_found": bool(z and z.found), "zone_cx": z.cx if z else None, "zone_cy": z.cy if z else None,
            "zone_r": z.r if z else None, "zone_conf": z.confidence if z else None,
        })
        for row in r.rows:
            d = asdict(row)
            d["bars"] = ",".join(row.bars)
            d["health"] = ",".join(f"{h:g}" for h in row.health)
            team_rows.append({"video_id": vid, "t": r.t, "match_no": match_of(r.t), **d})
    fdf, tdf = pd.DataFrame(frame_rows), pd.DataFrame(team_rows)
    fdf.to_parquet(out_dir / "frames.parquet", index=False)
    tdf.to_parquet(out_dir / "teams.parquet", index=False)
    matches = [s.as_dict() for s in kept]
    dropped = [s.as_dict() for s in segs if not s.kept]
    (out_dir / "matches.json").write_text(json.dumps({"matches": matches, "dropped": dropped}, indent=2))

    quality = quality_summary(fdf, tdf, matches)
    (out_dir / "quality.json").write_text(json.dumps(quality, indent=2))

    for rec, frame in pending:
        img = draw_debug(frame, rec, overlay, match_of(rec.t))
        cv2.imwrite(str(dbg_dir / f"{_hms(rec.t).replace(':', '')}_{rec.scene}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return RunResult(out_dir, dbg_dir, fdf, tdf, matches, quality)


def quality_summary(fdf: pd.DataFrame, tdf: pd.DataFrame, matches: list[dict[str, Any]]) -> dict[str, Any]:
    q: dict[str, Any] = {"frames": int(len(fdf))}
    if fdf.empty:
        return q
    q["scenes"] = {k: int(v) for k, v in fdf["scene"].value_counts().items()}
    mv = fdf[fdf["scene"] == "map_view"]
    q["map_frames"] = int(len(mv))
    q["zone_found_share"] = round(float(mv["zone_found"].mean()), 3) if len(mv) else None
    per_match = {}
    for m in matches:
        sub = mv[mv["match_no"] == m["match_no"]]
        per_match[str(m["match_no"])] = {
            "map_frames": int(len(sub)),
            "zone_found_share": round(float(sub["zone_found"].mean()), 3) if len(sub) else None,
        }
    if not tdf.empty:
        q["team_rows"] = int(len(tdf))
        q["tag_read_share"] = round(float(tdf["tag"].notna().mean()), 3)
        q["alive_read_share"] = round(float(tdf["alive"].notna().mean()), 3)
        q["fin_read_share"] = round(float(tdf["fin"].notna().mean()), 3)
        q["low_conf_rows"] = int((tdf["confidence"] < 0.5).sum())
        for m in matches:
            sub = tdf[tdf["match_no"] == m["match_no"]]
            if len(sub):
                per_match[str(m["match_no"])]["alive_read_share"] = round(float(sub["alive"].notna().mean()), 3)
    q["per_match"] = per_match
    return q
