"""Video type detection (F2): map_stream / gameplay / unknown.

Method (simple and explainable, PRD §13.5):
1. Sample N frames spread over the video (skipping pre-show and awards).
2. For each known overlay, look at its left-sidebar box (the 16-row team
   table). A sample "looks like the map view" when (a) that box shows the
   same pixels as several other samples (the table is a fixed graphic,
   while gameplay there is full-motion 3D) and (b) its brightness repeats
   at the row pitch (the table's rows).
3. score = share of samples that look like the map view. score >=
   min_score -> map_stream. Otherwise gameplay when frames decoded fine,
   unknown when they did not. The filename is only a hint and is recorded
   for conflicts.

Thresholds live in settings.yaml (videotype) and the overlay YAML
(videotype.*). They are first guesses until
tuned on real BMSD files in PRD build phase 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .overlays import box


@dataclass(frozen=True)
class TypeResult:
    video_type: str          # map_stream | gameplay | unknown
    score: float             # share of samples matching a map-stream layout
    overlay: str | None      # best-matching overlay id (for map_stream)
    frames_ok: int
    frames_total: int
    hint: str
    hint_conflict: bool
    detail: str


def sample_times(duration_s: float, n: int, skip_frac: float) -> list[float]:
    if duration_s <= 0 or n <= 0:
        return []
    start, end = duration_s * skip_frac, duration_s * (1 - skip_frac)
    if n == 1:
        return [(start + end) / 2]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _match_counts(frames: list[np.ndarray], region: tuple[int, int, int, int],
                  pixel_tol: int, static_frac: float) -> np.ndarray:
    """For each frame, how many OTHER frames show the same pixels in `region`."""
    x1, y1, x2, y2 = region
    # downscale 4x: fast and tolerant to compression noise
    crops = np.stack([f[y1:y2:4, x1:x2:4].astype(np.int16) for f in frames])
    n = len(crops)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        close = np.abs(crops - crops[i]).max(axis=3) <= pixel_tol
        same = close.reshape(n, -1).mean(axis=1) >= static_frac
        counts[i] = int(same.sum()) - 1
    return counts


def row_periodicity(frame: np.ndarray, sidebar: dict[str, Any], region: tuple[int, int, int, int]) -> float:
    """Autocorrelation of the sidebar's brightness profile at the row pitch (16-row team table)."""
    rows, first, pitch = int(sidebar.get("rows", 0)), int(sidebar.get("first_row_y", 0)), int(sidebar.get("row_h", 0))
    if rows < 3 or pitch < 4:
        return 1.0  # no table described for this overlay: skip the check
    x1, _, x2, _ = region
    band = frame[first:first + rows * pitch, x1:x2].astype(np.float32).mean(axis=2)
    profile = band.mean(axis=1)
    profile = profile - profile.mean()
    denom = float((profile * profile).sum())
    if denom <= 1e-6 or len(profile) <= pitch:
        return 0.0

    def ac(lag: int) -> float:
        return float((profile[:-lag] * profile[lag:]).sum() / denom)

    # Rows repeat every `pitch` px and are out of phase at half a pitch. A smooth
    # picture (desk, cams) correlates at both lags, so the difference stays near 0.
    return ac(pitch) - ac(max(1, pitch // 2))


def layout_score(frames: list[np.ndarray], overlay: dict[str, Any]) -> float:
    """Share of frames that look like this overlay's in-match map view."""
    if len(frames) < 3:
        return 0.0
    region = box(overlay, "layouts", "map_view", "sidebar_left")
    sidebar = overlay["layouts"]["map_view"]["sidebar_left"]
    vt = overlay.get("videotype") or {}
    counts = _match_counts(frames, region, int(vt.get("pixel_tol", 18)), float(vt.get("static_frac", 0.75)))
    min_matches = max(2, int(round(float(vt.get("min_match_frac", 0.15)) * len(frames))))
    min_period = float(vt.get("min_row_periodicity", 0.2))
    hits = [
        counts[i] >= min_matches and row_periodicity(f, sidebar, region) >= min_period
        for i, f in enumerate(frames)
    ]
    return float(np.mean(hits))


def classify_frames(frames: list[np.ndarray], overlays: dict[str, dict[str, Any]],
                    min_score: float) -> tuple[str, float, str | None]:
    best_score, best_id = 0.0, None
    for oid, ov in overlays.items():
        try:
            s = layout_score(frames, ov)
        except ValueError:
            continue
        if s > best_score:
            best_score, best_id = s, oid
    if best_score >= min_score:
        return "map_stream", best_score, best_id
    return "gameplay", best_score, None


def detect_type(duration_s: float, hint: str, grab: Callable[[float], np.ndarray | None],
                overlays: dict[str, dict[str, Any]], samples: int, skip_frac: float,
                min_score: float) -> TypeResult:
    times = sample_times(duration_s, samples, skip_frac)
    frames = [f for f in (grab(t) for t in times) if f is not None]
    if len(frames) < max(3, len(times) // 2):
        return TypeResult("unknown", 0.0, None, len(frames), len(times), hint, False,
                          "too few frames decoded")
    if not overlays:
        return TypeResult("unknown", 0.0, None, len(frames), len(times), hint, False,
                          "no overlay configs in config/overlays")
    vtype, score, overlay = classify_frames(frames, overlays, min_score)
    conflict = hint != "unknown" and hint != vtype
    return TypeResult(vtype, round(score, 3), overlay, len(frames), len(times), hint, conflict,
                      f"map-view layout in {score:.0%} of samples")


def detect_type_for_file(settings, path: Path, duration_s: float, hint: str,
                         overlays: dict[str, dict[str, Any]]) -> TypeResult:
    from .media import grab_frame

    size = (1280, 720)
    if overlays:
        w, h = next(iter(overlays.values()))["resolution"]
        size = (int(w), int(h))
    vt = settings.videotype
    return detect_type(duration_s, hint, lambda t: grab_frame(settings, path, t, size),
                       overlays, vt.samples, vt.skip_head_frac, vt.min_score)
