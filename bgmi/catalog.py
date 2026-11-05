"""Watch-folder catalog (F1).

Layout: <videos_root>/<tournament>/<stage>/<day>/<file>.mp4
A file directly under <stage> (no day folder) gets day = None.
Files at any other depth are reported as skipped, never guessed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .settings import Settings

_DAY_RE = re.compile(r"^\s*(?:day|d)\s*[-_ ]?\s*(\d{1,2})\s*$", re.IGNORECASE)
_SPACES = re.compile(r"[\s_\-]+")

MAP_HINTS = ("map stream", "mapstream", "map_stream", "map-stream", "map view", "observer", " map ")
GAMEPLAY_HINTS = ("gameplay", "game play", "main stream", "mainstream", "pov", "broadcast")


@dataclass(frozen=True)
class CatalogEntry:
    video_id: str
    path: Path
    rel_path: str
    tournament: str
    stage: str
    stage_raw: str
    day: str | None
    day_no: int | None
    size_bytes: int
    type_hint: str  # map_stream | gameplay | unknown (filename only)


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str


def normalize_stage(raw: str, aliases: dict[str, str]) -> str:
    """'Quater Finals' -> 'Quarter Finals', 'SF' -> 'Semi Finals', 'Grand Finals' -> 'Finals'."""
    key = _SPACES.sub(" ", raw).strip().lower()
    if key in aliases:
        return aliases[key]
    # common typo + plural forms not listed explicitly
    if key.rstrip("s") in aliases:
        return aliases[key.rstrip("s")]
    return " ".join(w.capitalize() if w.islower() else w for w in _SPACES.sub(" ", raw).strip().split(" "))


def parse_day(raw: str) -> tuple[str, int] | None:
    m = _DAY_RE.match(raw)
    if not m:
        return None
    n = int(m.group(1))
    return f"Day {n}", n


def filename_hint(name: str) -> str:
    low = f" {_SPACES.sub(' ', Path(name).stem.lower())} "
    if any(h in low for h in MAP_HINTS):
        return "map_stream"
    if any(h in low for h in GAMEPLAY_HINTS):
        return "gameplay"
    return "unknown"


def make_video_id(rel_path: str, size_bytes: int) -> str:
    """Stable id: same file at the same relative path keeps its id (mtime is ignored on purpose)."""
    key = f"{rel_path.replace(chr(92), '/').lower()}|{size_bytes}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


def scan(settings: Settings) -> tuple[list[CatalogEntry], list[SkippedFile]]:
    root = settings.videos_root
    if not root.is_dir():
        raise FileNotFoundError(f"videos_root does not exist: {root}")
    entries: list[CatalogEntry] = []
    skipped: list[SkippedFile] = []
    for path in _iter_videos(root, settings.video_extensions):
        rel = path.relative_to(root)
        parts = rel.parts[:-1]
        if len(parts) not in (2, 3):
            skipped.append(SkippedFile(path, f"expected <tournament>/<stage>/<day>/file, got depth {len(parts)}"))
            continue
        tournament, stage_raw = parts[0].strip(), parts[1].strip()
        day: str | None = None
        day_no: int | None = None
        if len(parts) == 3:
            parsed = parse_day(parts[2])
            if parsed is None:
                skipped.append(SkippedFile(path, f"day folder '{parts[2]}' is not like 'Day 3'"))
                continue
            day, day_no = parsed
        size = path.stat().st_size
        rel_str = rel.as_posix()
        entries.append(CatalogEntry(
            video_id=make_video_id(rel_str, size),
            path=path,
            rel_path=rel_str,
            tournament=tournament,
            stage=normalize_stage(stage_raw, settings.stage_aliases),
            stage_raw=stage_raw,
            day=day,
            day_no=day_no,
            size_bytes=size,
            type_hint=filename_hint(path.name),
        ))
    entries.sort(key=lambda e: (e.tournament.lower(), e.stage, e.day_no or 0, e.path.name.lower()))
    return entries, skipped


def _iter_videos(root: Path, exts: tuple[str, ...]) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            yield p
