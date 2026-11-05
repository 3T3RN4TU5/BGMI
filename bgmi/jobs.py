"""Resumable, idempotent processing queue (F4).

State lives in SQLite at data_root/state/pipeline.db so a crash mid-run never
corrupts it. Status flow: queued -> running -> done | failed.
On startup, any video left 'running' by a crash goes back to 'queued'.
Re-cataloguing never resets a 'done' video unless the file itself changed
(a changed file gets a new video_id).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STATUSES = ("queued", "running", "done", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    rel_path      TEXT NOT NULL,
    tournament    TEXT NOT NULL,
    stage         TEXT NOT NULL,
    stage_raw     TEXT NOT NULL,
    day           TEXT,
    day_no        INTEGER,
    size_bytes    INTEGER NOT NULL,
    type_hint     TEXT NOT NULL,
    video_type    TEXT NOT NULL DEFAULT 'unknown',
    type_score    REAL,
    type_detail   TEXT,
    overlay       TEXT,
    duration_s    REAL,
    fps           REAL,
    width         INTEGER,
    height        INTEGER,
    status        TEXT NOT NULL DEFAULT 'queued',
    error         TEXT,
    missing       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_videos_status ON videos(status);
"""

_CATALOG_FIELDS = ("path", "rel_path", "tournament", "stage", "stage_raw", "day", "day_no",
                   "size_bytes", "type_hint")
_UPDATABLE = {"video_type", "type_score", "type_detail", "overlay", "duration_s", "fps",
              "width", "height", "error"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Queue:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- catalog sync ---------------------------------------------------
    def upsert_catalog(self, entries: Iterable[Any]) -> dict[str, int]:
        """Insert new videos as 'queued', refresh catalog fields of known ones,
        and flag videos whose file disappeared. Returns counts."""
        seen: set[str] = set()
        new = updated = 0
        now = _now()
        with closing(self._conn()) as c, c:
            for e in entries:
                row = {k: (str(v) if k == "path" else v) for k, v in asdict(e).items()}
                seen.add(row["video_id"])
                exists = c.execute("SELECT 1 FROM videos WHERE video_id=?", (row["video_id"],)).fetchone()
                if exists:
                    sets = ", ".join(f"{k}=?" for k in _CATALOG_FIELDS)
                    c.execute(f"UPDATE videos SET {sets}, missing=0, updated_at=? WHERE video_id=?",
                              [row[k] for k in _CATALOG_FIELDS] + [now, row["video_id"]])
                    updated += 1
                else:
                    cols = ("video_id",) + _CATALOG_FIELDS
                    c.execute(
                        f"INSERT INTO videos ({', '.join(cols)}, status, created_at, updated_at) "
                        f"VALUES ({', '.join('?' * len(cols))}, 'queued', ?, ?)",
                        [row[k] for k in cols] + [now, now],
                    )
                    new += 1
            missing = 0
            for (vid,) in c.execute("SELECT video_id FROM videos WHERE missing=0").fetchall():
                if vid not in seen:
                    c.execute("UPDATE videos SET missing=1, updated_at=? WHERE video_id=?", (now, vid))
                    missing += 1
        return {"new": new, "updated": updated, "missing": missing}

    def update_fields(self, video_id: str, **fields: Any) -> None:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with closing(self._conn()) as c, c:
            c.execute(f"UPDATE videos SET {sets}, updated_at=? WHERE video_id=?",
                      list(fields.values()) + [_now(), video_id])

    # ---- status transitions --------------------------------------------
    def recover_stale(self) -> int:
        """Crash recovery: 'running' -> 'queued'."""
        with closing(self._conn()) as c, c:
            cur = c.execute("UPDATE videos SET status='queued', updated_at=? WHERE status='running'", (_now(),))
            return cur.rowcount

    def claim_next(self, video_type: str | None = "map_stream") -> sqlite3.Row | None:
        """Atomically move the next queued video to 'running' and return it."""
        with closing(self._conn()) as c:
            c.execute("BEGIN IMMEDIATE")
            q = "SELECT * FROM videos WHERE status='queued' AND missing=0"
            args: list[Any] = []
            if video_type:
                q += " AND video_type=?"
                args.append(video_type)
            row = c.execute(q + " ORDER BY tournament, stage, day_no, rel_path LIMIT 1", args).fetchone()
            if row is not None:
                c.execute("UPDATE videos SET status='running', error=NULL, updated_at=? WHERE video_id=?",
                          (_now(), row["video_id"]))
            c.commit()
            return row

    def mark_done(self, video_id: str) -> None:
        self._set_status(video_id, "done", None, expect="running")

    def mark_failed(self, video_id: str, error: str) -> None:
        self._set_status(video_id, "failed", error[:2000], expect="running")

    def requeue(self, video_id: str | None = None, only_failed: bool = True) -> int:
        q = "UPDATE videos SET status='queued', error=NULL, updated_at=?"
        args: list[Any] = [_now()]
        conds = []
        if only_failed:
            conds.append("status='failed'")
        if video_id:
            conds.append("video_id=?")
            args.append(video_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        with closing(self._conn()) as c, c:
            return c.execute(q, args).rowcount

    def _set_status(self, video_id: str, status: str, error: str | None, expect: str) -> None:
        with closing(self._conn()) as c, c:
            cur = c.execute("UPDATE videos SET status=?, error=?, updated_at=? WHERE video_id=? AND status=?",
                            (status, error, _now(), video_id, expect))
            if cur.rowcount != 1:
                raise RuntimeError(f"{video_id}: expected status '{expect}' before '{status}'")

    # ---- reads -----------------------------------------------------------
    def all(self, include_missing: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM videos" + ("" if include_missing else " WHERE missing=0")
        with closing(self._conn()) as c:
            return c.execute(q + " ORDER BY tournament, stage, day_no, rel_path").fetchall()

    def get(self, video_id: str) -> sqlite3.Row | None:
        with closing(self._conn()) as c:
            return c.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()

    def counts(self) -> dict[str, int]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM videos WHERE missing=0 GROUP BY status").fetchall()
        out = {s: 0 for s in STATUSES}
        out.update({r["status"]: r["n"] for r in rows})
        return out
