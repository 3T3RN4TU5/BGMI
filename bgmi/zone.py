"""Safe-zone circle (F10): the thin white ring drawn on the map.

1. Candidate pixels: inside the map viewport, keep thin bright lines (a
   morphological top-hat removes anything wider than a few px), low saturation
   and high value (white, not team-coloured plates), in elongated connected
   pieces (drops letters and icons).
2. RANSAC circle fit on those pixels, scored by how many 5-degree sectors of
   the ring are covered (roads and grid lines cover few sectors of any circle).
3. confidence = share of the ring's *visible* sectors that were found. Below
   min_coverage the frame gets no zone (never a guess).
4. flag_jumps(): lower the confidence of a ring unlike any neighbouring frame's.

Output is in screen pixels. Screen -> map coordinates needs map registration
(F13, PRD build phase 2) because the camera zooms and pans.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .overlays import box

SECTORS = 72


@dataclass(frozen=True)
class ZoneResult:
    found: bool
    cx: float | None        # screen px (full frame)
    cy: float | None
    r: float | None
    coverage: float         # share of visible ring sectors with ring pixels
    confidence: float
    detail: str


def ring_candidates(view: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Binary mask of thin, white, elongated line pixels."""
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    k = int(cfg.get("tophat_k", 7))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ((th > int(cfg.get("tophat_min", 40)))
         & (hsv[..., 1] < int(cfg.get("s_max", 60)))
         & (hsv[..., 2] > int(cfg.get("v_min", 170)))).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        _, _, w, h, area = st[i]
        # long, not a pure horizontal/vertical line, and sparse in its box (a curve, not a letter)
        keep[i] = max(w, h) >= 12 and min(w, h) >= 3 and area / float(w * h) < 0.5
    return keep[lab]


def _circle3(p: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return ux, uy, float(np.hypot(x1 - ux, y1 - uy))


def _visible_sectors(cx: float, cy: float, r: float, w: int, h: int) -> np.ndarray:
    th = (np.arange(SECTORS) + 0.5) / SECTORS * 2 * np.pi - np.pi
    px, py = cx + r * np.cos(th), cy + r * np.sin(th)
    return (px >= 0) & (px < w) & (py >= 0) & (py < h)


def _sectors(pts: np.ndarray, cx: float, cy: float) -> np.ndarray:
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return np.clip(((ang + np.pi) / (2 * np.pi) * SECTORS).astype(int), 0, SECTORS - 1)


def fit_ring(pts: np.ndarray, w: int, h: int, cfg: dict[str, Any], seed: int = 0):
    """RANSAC: the circle whose inliers cover the most ring sectors."""
    if len(pts) < 30:
        return None
    rng = np.random.default_rng(seed)
    tol = float(cfg.get("inlier_px", 2.0))
    rmin, rmax = float(cfg.get("r_min", 40)), float(cfg.get("r_max", 1500))
    pf = pts.astype(np.float64)
    # score on a random subset: plenty for a ring, and keeps a frame well under 0.5 s
    cap = int(cfg.get("max_points", 2500))
    sub = pf[rng.choice(len(pf), cap, replace=False)] if len(pf) > cap else pf
    best = None
    best_inl = 0
    for _ in range(int(cfg.get("iterations", 1500))):
        c = _circle3(sub[rng.choice(len(sub), 3, replace=False)])
        if c is None or not rmin <= c[2] <= rmax:
            continue
        inl = np.abs(np.hypot(sub[:, 0] - c[0], sub[:, 1] - c[1]) - c[2]) < tol
        n_inl = int(inl.sum())
        if n_inl < 10 or n_inl < 0.5 * best_inl:
            continue                     # cannot beat the best ring: skip the sector count
        vis = _visible_sectors(*c, w, h)
        if vis.sum() < 8:
            continue
        covered = np.zeros(SECTORS, dtype=bool)
        covered[_sectors(sub[inl], c[0], c[1])] = True
        score = int((covered & vis).sum())
        if best is None or score > best[0]:
            best = (score, c, None, vis)
            best_inl = max(best_inl, n_inl)
    if best is None:
        return None
    # least-squares refine on the inliers (algebraic fit)
    score, c, _, vis = best
    q = pf[np.abs(np.hypot(pf[:, 0] - c[0], pf[:, 1] - c[1]) - c[2]) < tol]
    if len(q) >= 10:
        a = np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))]
        b = (q ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = sol[0], sol[1]
        r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0)))
        if rmin <= r <= rmax:
            c = (float(cx), float(cy), r)
    # final coverage on all points around the refined circle
    vis = _visible_sectors(*c, w, h)
    inl = np.abs(np.hypot(pf[:, 0] - c[0], pf[:, 1] - c[1]) - c[2]) < tol
    covered = np.zeros(SECTORS, dtype=bool)
    covered[_sectors(pf[inl], c[0], c[1])] = True
    return c, int((covered & vis).sum()), int(vis.sum())


def detect_zone(frame: np.ndarray, overlay: dict[str, Any], layout: str = "map_view") -> ZoneResult:
    cfg = overlay.get("zone") or {}
    if layout == "match_start":
        x1, y1, x2, y2 = box(overlay, "layouts", "match_start", "map_box")
    else:
        x1, y1, x2, y2 = box(overlay, "layouts", "map_view", "viewport")
    view = frame[y1:y2, x1:x2]
    mask = ring_candidates(view, cfg)
    ys, xs = np.nonzero(mask)
    fit = fit_ring(np.stack([xs, ys], 1), view.shape[1], view.shape[0], cfg)
    if fit is None:
        return ZoneResult(False, None, None, None, 0.0, 0.0, f"{len(xs)} ring pixels, no circle")
    (cx, cy, r), covered, visible = fit
    coverage = covered / max(visible, 1)
    found = coverage >= float(cfg.get("min_coverage", 0.6))
    return ZoneResult(
        found,
        round(cx + x1, 1) if found else None,
        round(cy + y1, 1) if found else None,
        round(r, 1) if found else None,
        round(coverage, 3),
        round(coverage, 3) if found else 0.0,
        f"ring covers {covered}/{visible} visible sectors",
    )


def flag_jumps(zones: Sequence[ZoneResult], window: int = 2) -> list[ZoneResult]:
    """Halve the confidence of a zone that disagrees with all of its neighbours.

    Values are never changed: the camera pans and zooms, so screen positions of
    neighbouring frames can legitimately differ. Real temporal smoothing happens
    in map coordinates after map registration (PRD build phase 2).
    """
    out: list[ZoneResult] = []
    for i, z in enumerate(zones):
        nb = [zones[j] for j in range(max(0, i - window), min(len(zones), i + window + 1))
              if j != i and zones[j].found]
        if not z.found or not nb:
            out.append(z)
            continue
        agrees = any(abs(n.r - z.r) <= 0.2 * z.r and np.hypot(n.cx - z.cx, n.cy - z.cy) <= 0.3 * z.r for n in nb)
        out.append(z if agrees else ZoneResult(True, z.cx, z.cy, z.r, z.coverage, round(z.confidence * 0.5, 3),
                                               z.detail + "; differs from neighbours"))
    return out
