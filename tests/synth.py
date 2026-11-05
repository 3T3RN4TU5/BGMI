"""Synthetic frames drawn from the overlay config (no broadcast footage in git)."""
from __future__ import annotations

import cv2
import numpy as np

from tests.conftest import ROOT

W, H = 1280, 720
LIME = (70, 205, 149)
TAGS = ["GENS", "NBE", "MYTH", "GODL", "NINZ", "WELT", "GDR", "V1BA",
        "BLEG", "SOUL", "RNTX", "TAG", "8BIT", "OR", "K9", "WF"]


def overlay():
    from bgmi.overlays import load_overlays
    return load_overlays(ROOT / "config")["kie_bmsd2026"]


def _row_top(ov, k):
    sb = ov["layouts"]["map_view"]["sidebar_left"]
    return int(round(float(sb["first_row_y"]) + float(sb["row_h"]) * k))


def map_view(ov, rows=None, rng=None, ring=None, match_digit="5"):
    """Map view: lime header + finish-leaders bar, team table, map with an optional white ring.

    rows: list of (tag, fin, bars) with bars a 4-char string of A (alive), K (knocked), D (dead).
    ring: (cx, cy, r) in full-frame px.
    """
    rng = rng or np.random.default_rng(0)
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :292] = (40, 22, 12)
    f[:, 987:] = (40, 22, 12)
    f[2:26, 0:290] = LIME
    f[256:278, 1008:1040] = LIME
    # map: smooth green-brown terrain with mild noise
    terrain = cv2.resize(rng.integers(40, 110, (18, 20, 3), dtype=np.uint8), (695, 720), interpolation=cv2.INTER_CUBIC)
    f[:, 292:987] = terrain
    if ring is not None:
        cx, cy, r = ring
        cv2.circle(f, (int(cx), int(cy)), int(r), (255, 255, 255), 1, cv2.LINE_AA)
    # match number (units digit) in lime
    x1, y1, x2, y2 = ov["layouts"]["map_view"]["match_no"]
    cv2.putText(f, match_digit, (x1 + 2, y2 - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, LIME, 1)
    rows = rows if rows is not None else [(t, "00", "AAAA") for t in TAGS]
    sb = ov["layouts"]["map_view"]["sidebar_left"]
    for k, (tag, fin, bars) in enumerate(rows):
        t = _row_top(ov, k)
        cv2.putText(f, tag, (int(sb["tag_x"][0]) + 4, t + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        for (dx1, _), d in zip(sb["fin_digits_x"], fin):
            cv2.putText(f, d, (int(dx1) + 2, t + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        for x, b in zip(sb["bars"]["x"], bars):
            y1b, y2b = t + 11, t + 33
            color = {"A": (250, 250, 250), "K": (60, 60, 230), "D": (105, 88, 78)}[b]
            f[y1b:y2b, int(x):int(x) + 4] = color
    return f


def match_start(ov):
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :] = (60, 30, 40)
    for p in ov["scene"]["match_start"]["probes"]:
        x1, y1, x2, y2 = p["box"]
        f[y1:y2 + 10, x1:x2] = LIME
    return f


def desk(rng=None):
    rng = rng or np.random.default_rng(1)
    f = np.full((H, W, 3), 90, np.uint8)
    f[200:600, 100:500] = (150, 120, 100)
    return f + rng.integers(0, 8, (H, W, 3), dtype=np.uint8)


def cluttered_ring(ov, seed, ring=(660, 420, 130), n_clutter=110, n_plates=9):
    """A small ring mostly hidden: thin white clutter arcs (roads, letters) and dark player plates.

    Only ~5% of the ring mask is ring, as on real late-game frames.
    """
    rng = np.random.default_rng(seed)
    f = map_view(ov, ring=ring, rng=rng)
    for _ in range(n_clutter):
        c = (int(rng.integers(300, 980)), int(rng.integers(10, 710)))
        ax = (int(rng.integers(20, 120)), int(rng.integers(20, 120)))
        a0 = int(rng.integers(0, 360))
        cv2.ellipse(f, c, ax, int(rng.integers(0, 180)), a0, a0 + int(rng.integers(40, 160)),
                    (245, 245, 245), 1, cv2.LINE_AA)
    cx, cy, r = ring
    for _ in range(n_plates):
        th = rng.uniform(0, 2 * np.pi)
        x, y = int(cx + r * np.cos(th)), int(cy + r * np.sin(th))
        f[y - 9:y + 9, x - 16:x + 16] = (60, 40, 30)
    return f
