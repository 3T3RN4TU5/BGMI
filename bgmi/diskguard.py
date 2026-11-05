"""Disk-space guard and file-safety rules (F5, NFR-08).

- A job refuses to start when free space on data_root's drive is below min_free_gb.
- Nothing under videos_root is ever written, moved or deleted.
- Deletions are only allowed inside data_root.
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .settings import Settings


class DiskSpaceError(RuntimeError):
    pass


class UnsafePathError(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    """Free space in GB on the drive holding `path` (walks up to an existing folder)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1024**3


def ensure_free_space(settings: Settings, needed_gb: float = 0.0) -> float:
    """Raise DiskSpaceError unless free space >= min_free_gb + needed_gb. Returns free GB."""
    free = free_gb(settings.data_root)
    required = settings.min_free_gb + needed_gb
    if free < required:
        raise DiskSpaceError(
            f"Only {free:.1f} GB free on the data drive; need at least {required:.1f} GB. "
            "Free some space or move data_root to a bigger drive."
        )
    return free


def _is_within(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def assert_writable(settings: Settings, path: Path) -> None:
    """Refuse any write under videos_root."""
    if _is_within(path, settings.videos_root):
        raise UnsafePathError(f"Refusing to write under videos_root: {path}")


def safe_rmtree(settings: Settings, path: Path) -> None:
    """Delete a folder, but only if it is strictly inside data_root."""
    path = Path(path)
    if _is_within(path, settings.videos_root):
        raise UnsafePathError(f"Refusing to delete under videos_root: {path}")
    if not _is_within(path, settings.data_root) or path.resolve() == settings.data_root.resolve():
        raise UnsafePathError(f"Refusing to delete outside data_root: {path}")
    if path.exists():
        shutil.rmtree(path)


@contextmanager
def frames_dir(settings: Settings, video_id: str) -> Iterator[Path]:
    """Temporary frame folder for one video; removed afterwards unless keep_frames."""
    folder = settings.work_dir / video_id / "frames"
    assert_writable(settings, folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        yield folder
    finally:
        if not settings.keep_frames:
            safe_rmtree(settings, folder)
