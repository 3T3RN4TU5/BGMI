"""Scene classifier (F8): map_view / match_start / other.

Simple and explainable: each layout lists a few "probes", fixed boxes on the
broadcaster overlay that are a solid colour only in that layout (the lime
header bar of the team table, the lime tag strips under the 16 team tiles of
the match-start screen). A probe passes when most of its pixels are close to
the expected colour; a layout matches when enough probes pass. Desk, player
cams, replays and ads have none of these graphics, so they fall to "other".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCENES = ("map_view", "match_start", "other")


@dataclass(frozen=True)
class SceneResult:
    scene: str              # map_view | match_start | other
    confidence: float       # 0-1
    passed: dict[str, int]  # layout -> probes passed
    detail: str


def probe_frac(frame: np.ndarray, probe: dict[str, Any]) -> float:
    """Share of the probe box's pixels within `tol` (per channel) of its colour."""
    x1, y1, x2, y2 = (int(v) for v in probe["box"])
    crop = frame[y1:y2, x1:x2].astype(np.int16)
    if crop.size == 0:
        return 0.0
    color = np.array(probe["color"], dtype=np.int16)
    close = np.abs(crop - color).max(axis=2) <= int(probe.get("tol", 40))
    return float(close.mean())


def classify_scene(frame: np.ndarray, overlay: dict[str, Any]) -> SceneResult:
    cfg = overlay.get("scene") or {}
    passed: dict[str, int] = {}
    ratio: dict[str, float] = {}     # passed / need
    margin: dict[str, float] = {}    # 0-1, how far past the bar
    for layout in ("map_view", "match_start"):
        spec = cfg.get(layout)
        if not spec:
            continue
        probes = spec["probes"]
        need = int(spec.get("need", len(probes)))
        ok = [probe_frac(frame, p) >= float(p.get("min_frac", 0.8)) for p in probes]
        passed[layout] = int(sum(ok))
        ratio[layout] = passed[layout] / max(need, 1)
        margin[layout] = (passed[layout] - need + 1) / max(len(probes) - need + 1, 1)
    hits = [k for k in passed if ratio[k] >= 1.0]
    if len(hits) == 1:
        k = hits[0]
        return SceneResult(k, round(0.75 + 0.25 * min(margin[k], 1.0), 3), passed,
                           f"{k}: {passed[k]} probes passed")
    if len(hits) > 1:  # should not happen with a sane overlay: report it, never guess
        return SceneResult("other", 0.3, passed, "several layouts matched: " + ", ".join(hits))
    near = max(ratio.values(), default=0.0)
    return SceneResult("other", round(1.0 - 0.5 * near, 3), passed, "no overlay layout matched")
