"""Team table reader (F12): rank, team, finishes and alive players per row.

The left table of the map view lists the 16 teams sorted by the current
standings, so the row order changes during a match. For each row:

- team: the tag text (e.g. "GODL") is matched against templates learned
  from labelled frames (`cli.py calibrate`). Each tag is used once per frame.
- finishes: the two FIN. digits, matched against learned digit templates.
  A digit never seen in calibration gives None, never a guess.
- alive: the 4 STATUS bars. White = alive (bar height = health), red =
  knocked, grey/dark = dead (mapping in the overlay YAML `bars`).

Templates are cut from the broadcast, so they live in data_root (never in git).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class RowRead:
    rank: int                    # 1-16, row position in the table
    tag: str | None
    tag_score: float
    tag_conf: float
    fin: int | None
    fin_conf: float
    bars: tuple[str, ...]        # per player: alive | knocked | dead | unknown
    health: tuple[float, ...]    # 0-1 bar height for alive players, else 0
    alive: int | None
    eliminated: bool
    confidence: float


def _sb(overlay: dict[str, Any]) -> dict[str, Any]:
    return overlay["layouts"]["map_view"]["sidebar_left"]


def row_top(overlay: dict[str, Any], k: int) -> int:
    sb = _sb(overlay)
    return int(round(float(sb["first_row_y"]) + float(sb["row_h"]) * k))


def text_crop(frame: np.ndarray, overlay: dict[str, Any], k: int, x_range: Iterable[int]) -> np.ndarray:
    """Row k's text in [x1, x2) as float 0-1: grey/white text bright, coloured noise removed."""
    sb = _sb(overlay)
    t = row_top(overlay, k)
    dy1, dy2 = sb.get("text_y", [11, 33])
    x1, x2 = x_range
    c = frame[t + int(dy1):t + int(dy2), int(x1):int(x2)]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    v[hsv[..., 1] > 70] = 0          # the overlay's green "glitch" dots are saturated, text is not
    lo, hi = float(np.median(v)), float(np.percentile(v, 99))
    return np.clip((v - lo) / max(hi - lo, 1.0), 0, 1)


def text_brightness(frame: np.ndarray, overlay: dict[str, Any], k: int) -> float:
    sb = _sb(overlay)
    t = row_top(overlay, k)
    dy1, dy2 = sb.get("text_y", [11, 33])
    x1, x2 = sb["tag_x"]
    return float(np.percentile(frame[t + int(dy1):t + int(dy2), int(x1):int(x2)].max(axis=2), 99))


def match(img: np.ndarray, tpl: np.ndarray) -> float:
    """Best normalised correlation of tpl (trimmed 2 px each side) over img: tolerates 2 px shifts."""
    t = tpl[2:-2, 2:-2]
    if t.shape[0] > img.shape[0] or t.shape[1] > img.shape[1] or float(t.std()) < 1e-6:
        return -1.0
    return float(cv2.matchTemplate(img.astype(np.float32), t.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())


@dataclass
class GlyphBook:
    """Templates by label: 'tag:GODL' -> [crops], 'digit:4' -> [crops]."""
    items: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, key: str, crop: np.ndarray, max_per_key: int = 6) -> None:
        lst = self.items.setdefault(key, [])
        if len(lst) < max_per_key:
            lst.append(crop.astype(np.float32))

    def labels(self, kind: str) -> list[str]:
        return sorted(k.split(":", 1)[1] for k in self.items if k.startswith(kind + ":"))

    def score(self, kind: str, label: str, img: np.ndarray) -> float:
        return max((match(img, t) for t in self.items.get(f"{kind}:{label}", [])), default=-1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{k}|{i}": v for k, lst in self.items.items() for i, v in enumerate(lst)}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "GlyphBook":
        book = cls()
        if Path(path).exists():
            with np.load(path) as data:
                for name in sorted(data.files, key=lambda n: (n.rsplit("|", 1)[0], int(n.rsplit("|", 1)[1]))):
                    book.items.setdefault(name.rsplit("|", 1)[0], []).append(data[name])
        return book


def digit_crops(frame: np.ndarray, overlay: dict[str, Any], k: int) -> list[np.ndarray]:
    """The two FIN. digits of row k."""
    return [text_crop(frame, overlay, k, xr) for xr in _sb(overlay)["fin_digits_x"]]


def learn(book: GlyphBook, frame: np.ndarray, overlay: dict[str, Any],
          tags: list[str], fins: list[str] | None = None) -> None:
    """Add templates from one labelled map-view frame (tags/fins top to bottom)."""
    sb = _sb(overlay)
    if len(tags) != int(sb["rows"]):
        raise ValueError(f"need {sb['rows']} tags, got {len(tags)}")
    for k, tag in enumerate(tags):
        book.add(f"tag:{tag}", text_crop(frame, overlay, k, sb["tag_x"]))
        if fins:
            fin = fins[k]
            if len(fin) != 2 or not fin.isdigit():
                raise ValueError(f"FIN must be two digits like 04, got {fin!r}")
            for d, crop in zip(fin, digit_crops(frame, overlay, k)):
                book.add(f"digit:{d}", crop)


def read_bars(frame: np.ndarray, overlay: dict[str, Any], k: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    sb, cfg = _sb(overlay), overlay.get("bars") or {}
    t = row_top(overlay, k)
    dy1, dy2 = sb.get("text_y", [11, 33])
    w = int(sb["bars"].get("w", 4))
    states, health = [], []
    min_px = int(cfg.get("min_px", 2))
    # what each colour means differs between tournaments: set per overlay
    meaning = {"white": "alive", "red": "knocked", "grey": "dead", **(cfg.get("meaning") or {})}
    for x in sb["bars"]["x"]:
        col = frame[t + int(dy1):t + int(dy2), int(x):int(x) + w].astype(np.int16)
        b, g, r = col[..., 0], col[..., 1], col[..., 2]
        white_rows = (np.minimum(np.minimum(b, g), r) >= int(cfg.get("white_min", 200))).any(axis=1)
        red_rows = ((r >= int(cfg.get("red_min_r", 170)))
                    & (r - np.maximum(g, b) >= int(cfg.get("red_min_gap", 90)))).any(axis=1)
        n_white, n_red = int(white_rows.sum()), int(red_rows.sum())
        if n_red >= max(min_px, 3) and n_red >= n_white:
            states.append(meaning["red"])
            health.append(0.0)
        elif n_white >= min_px:
            states.append(meaning["white"])
            health.append(round(n_white / len(white_rows), 2) if meaning["white"] == "alive" else 0.0)
        else:
            # an empty slot is grey (dead); anything else (e.g. a pop-up over the table) is unknown
            med = np.median(col.reshape(-1, 3), axis=0)
            grey = float(med.max() - med.min()) <= 50 and 40 <= float(med.mean()) <= 170
            states.append(meaning["grey"] if grey else "unknown")
            health.append(0.0)
    return tuple(states), tuple(health)


def _assign(scores: np.ndarray) -> list[int]:
    """Greedy one-to-one assignment, highest score first. -1 when nothing is left."""
    n_rows, n_lab = scores.shape
    out = [-1] * n_rows
    used_r, used_l = set(), set()
    for flat in np.argsort(-scores, axis=None):
        i, j = divmod(int(flat), n_lab)
        if i in used_r or j in used_l:
            continue
        out[i] = j
        used_r.add(i)
        used_l.add(j)
        if len(used_r) == n_rows:
            break
    return out


def read_sidebar(frame: np.ndarray, overlay: dict[str, Any], book: GlyphBook,
                 min_tag_score: float = 0.6) -> list[RowRead]:
    sb = _sb(overlay)
    n = int(sb["rows"])
    tags = book.labels("tag")
    digits = book.labels("digit")
    crops = [text_crop(frame, overlay, k, sb["tag_x"]) for k in range(n)]
    scores = np.array([[book.score("tag", t, c) for t in tags] for c in crops]) if tags else np.zeros((n, 0))
    assign = _assign(scores) if tags else [-1] * n
    knocked_alive = bool((overlay.get("bars") or {}).get("knocked_counts_alive", True))
    rows: list[RowRead] = []
    for k in range(n):
        # tag
        tag, tscore, tconf = None, 0.0, 0.0
        j = assign[k]
        if j >= 0:
            tscore = float(scores[k, j])
            others = np.delete(scores[k], j)
            second = float(others.max()) if len(others) else -1.0
            if tscore >= min_tag_score:
                tag = tags[j]
                # clear winner (margin >= 0.1) -> 1.0; neck and neck -> 0.5
                tconf = round(min(1.0, 0.5 + 5 * max(tscore - second, 0)), 3)
        # finishes
        fin, fconf = None, 0.0
        if digits:
            vals, confs = [], []
            for crop in digit_crops(frame, overlay, k):
                ds = sorted(((book.score("digit", d, crop), d) for d in digits), reverse=True)
                best, second = ds[0], ds[1] if len(ds) > 1 else (-1.0, "")
                if best[0] < 0.6:
                    vals = None
                    break
                vals.append(best[1])
                confs.append(min(1.0, 0.5 + 5 * max(best[0] - second[0], 0)))
            if vals is not None:
                fin, fconf = int("".join(vals)), round(min(confs), 3)
        # bars
        states, health = read_bars(frame, overlay, k)
        if "unknown" in states:
            alive = None
        else:
            alive = sum(s == "alive" or (knocked_alive and s == "knocked") for s in states)
        eliminated = alive == 0
        bconf = 1.0 if alive is not None else 0.0
        rows.append(RowRead(k + 1, tag, round(tscore, 3), tconf, fin, fconf, states, health,
                            alive, eliminated, round(min(tconf, bconf), 3)))
    return rows
