"""Match segmentation (F9): start/end time, match number and map per match.

Every map-view frame shows "DAY 3 - MATCH N" in the right panel. Within one
match that number crop looks the same; when it changes after a cut away from
the map, a new match began.
The full-map match-start screen opens a match. Desk, cams and ads between
matches are ignored (they belong to no match).

- start = first match-start screen (or first map-view frame) of the run
- end   = last map-view frame of the run
- number: read from the crop when learned templates match (`calibrate
  --match-no`), by majority vote over the segment's map frames. Otherwise
  counted on from a read match, or from the schedule's first match only when
  the run starts at the video start. Else unknown (never a guess).
- map: from the tournament schedule config (the overlay shows no map name).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .overlays import box
from .sidebar import GlyphBook, match


@dataclass
class FrameObs:
    t: float
    scene: str                       # map_view | match_start | other
    sig: np.ndarray | None = None    # normalised crop of the match-number box (map_view only)


@dataclass
class Segment:
    start_t: float
    end_t: float
    n_map: int = 0
    n_start: int = 0
    ref_sig: np.ndarray | None = field(default=None, repr=False)
    sigs: list[np.ndarray] = field(default_factory=list, repr=False)
    number: int | None = None
    number_source: str = ""          # read | counted | schedule | given | unknown
    number_votes: str = ""           # e.g. "41/44 frames read 15"
    map_name: str | None = None
    kept: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_no": self.number, "match_no_source": self.number_source,
            "match_no_votes": self.number_votes, "map": self.map_name,
            "start_s": round(self.start_t, 1), "end_s": round(self.end_t, 1),
            "duration_s": round(self.end_t - self.start_t, 1),
            "map_frames": self.n_map, "start_screens": self.n_start, "note": self.note,
        }


def match_sig(frame: np.ndarray, overlay: dict[str, Any], layout: str = "map_view") -> np.ndarray:
    """The match-number crop ("15"), brightness-normalised to 0-1."""
    x1, y1, x2, y2 = box(overlay, "layouts", layout, "match_no")
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = float(np.percentile(g, 5)), float(np.percentile(g, 99))
    return np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)


def _same(a: np.ndarray | None, b: np.ndarray | None, min_corr: float) -> bool:
    if a is None or b is None or a.shape != b.shape:
        return True       # nothing to compare: do not split on missing data
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den < 1e-6:
        return True
    return float((a * b).sum()) / den >= min_corr


def segment(obs: Sequence[FrameObs], cfg: dict[str, Any] | None = None) -> list[Segment]:
    cfg = cfg or {}
    # Same match: the number crop correlates >= min_corr (measured 0.99-1.00 within
    # Match 15, 0.96 between "15" and "16": the digits are 8 px tall).
    min_corr = float(cfg.get("sig_min_corr", 0.985))
    # A changed number only splits after a cut away from the map of at least min_gap_s,
    # so one noisy frame cannot split a match.
    min_gap = float(cfg.get("min_gap_s", 60))
    segs: list[Segment] = []
    cur: Segment | None = None
    for o in sorted(obs, key=lambda x: x.t):
        if o.scene == "match_start":
            if cur is None or cur.n_map > 0:
                cur = Segment(o.t, o.t)
                segs.append(cur)
            cur.n_start += 1
        elif o.scene == "map_view":
            # a start screen's segment has no reference yet, so its first map frame joins it
            if cur is None or (not _same(o.sig, cur.ref_sig, min_corr) and o.t - cur.end_t >= min_gap):
                cur = Segment(o.t, o.t)
                segs.append(cur)
            if o.sig is not None:
                if cur.ref_sig is None:
                    cur.ref_sig = o.sig
                cur.sigs.append(o.sig)
            cur.n_map += 1
            cur.end_t = o.t
    min_frames = int(cfg.get("min_map_frames", 20))
    min_dur = float(cfg.get("min_duration_s", 300))
    for s in segs:
        if s.n_map < min_frames or s.end_t - s.start_t < min_dur:
            s.kept = False
            s.note = f"dropped: {s.n_map} map frames over {s.end_t - s.start_t:.0f} s"
    return segs


def read_match_no(sig: np.ndarray | None, book: GlyphBook, min_score: float = 0.97,
                  min_margin: float = 0.015) -> int | None:
    """Match number of one crop from learned templates (calibrate --match-no). None unless clear."""
    if sig is None:
        return None
    scores = sorted(((max(match(sig, t) for t in book.items[f"matchno:{lab}"]), lab)
                     for lab in book.labels("matchno")), reverse=True)
    if not scores or scores[0][0] < min_score:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < min_margin:
        return None
    return int(scores[0][1])


def vote_match_no(sigs: Sequence[np.ndarray], book: GlyphBook, min_votes: int = 3,
                  min_share: float = 0.6, max_frames: int = 60) -> tuple[int | None, str]:
    """Majority vote over a segment's frames, so one faded or covered frame cannot decide.

    Returns (number or None, "k/n frames read N").
    """
    if not sigs or not book.labels("matchno"):
        return None, ""
    step = max(1, len(sigs) // max_frames)
    reads = [read_match_no(s, book) for s in sigs[::step]]
    got = [r for r in reads if r is not None]
    if not got:
        return None, f"0/{len(reads)} frames read"
    best = max(set(got), key=got.count)
    k = got.count(best)
    detail = f"{k}/{len(reads)} frames read {best}"
    if k >= min(min_votes, len(reads)) and k / len(got) >= min_share:
        return best, detail
    return None, detail


def number_matches(segs: Sequence[Segment], book: GlyphBook, schedule: dict[str, Any] | None,
                   from_video_start: bool = True, first_match: int | None = None) -> None:
    """Fill number/map for kept segments.

    Read numbers win. Unread segments are counted on from a read one. With nothing
    read, numbering starts at `first_match` when given (e.g. `analyze --first-match`),
    else at the schedule's first match only if the run started at the video start:
    a run over part of a day cannot know how many matches it skipped, so its
    numbers stay unknown instead of guessed.
    """
    kept = [s for s in segs if s.kept]
    sched_first = int((schedule or {}).get("first_match", 1))
    maps = list((schedule or {}).get("maps", []))
    for s in kept:
        n, s.number_votes = vote_match_no(s.sigs or ([s.ref_sig] if s.ref_sig is not None else []), book)
        if n is not None:
            s.number, s.number_source = n, "read"
    read = next(((i, s.number) for i, s in enumerate(kept) if s.number is not None), None)
    if read is not None:
        anchor, source = read, "counted"
    elif first_match is not None:
        anchor, source = (0, int(first_match)), "given"
    elif from_video_start and schedule:
        anchor, source = (0, sched_first), "schedule"
    else:
        anchor, source = None, "unknown"
    for i, s in enumerate(kept):
        if anchor is None:
            s.number_source = "unknown"
            s.note = (s.note + "; " if s.note else "") + "match number not read: run `calibrate --match-no` or pass --first-match"
            continue
        expect = anchor[1] + (i - anchor[0])
        if s.number is None:
            s.number, s.number_source = expect, source
        elif s.number != expect:
            s.note = (s.note + "; " if s.note else "") + f"read {s.number}, order suggests {expect}"
        idx = s.number - sched_first
        s.map_name = maps[idx] if 0 <= idx < len(maps) else None
