"""Load and validate settings.yaml (all paths configurable, any drive)."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS_FILE = "settings.yaml"


class SettingsError(ValueError):
    """settings.yaml is missing or invalid."""


@dataclass(frozen=True)
class VideoTypeSettings:
    samples: int = 24
    skip_head_frac: float = 0.05
    min_score: float = 0.35


@dataclass(frozen=True)
class Settings:
    videos_root: Path
    project_root: Path
    data_root: Path
    ffmpeg_bin: str = ""
    sample_every_s: float = 3
    keep_frames: bool = False
    min_free_gb: float = 5
    workers: int = 14
    mongo_uri: str = ""
    video_extensions: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".ts")
    stage_aliases: dict[str, str] = field(default_factory=dict)
    videotype: VideoTypeSettings = field(default_factory=VideoTypeSettings)

    # ---- derived paths -------------------------------------------------
    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def state_db(self) -> Path:
        return self.data_root / "state" / "pipeline.db"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def work_dir(self) -> Path:
        return self.data_root / "work"

    @property
    def debug_dir(self) -> Path:
        return self.data_root / "debug"

    def tool(self, name: str) -> str:
        """Resolve an ffmpeg-family executable (ffmpeg, ffprobe)."""
        exe = name + (".exe" if sys.platform == "win32" else "")
        if self.ffmpeg_bin:
            candidate = Path(self.ffmpeg_bin) / exe
            if candidate.exists():
                return str(candidate)
        found = shutil.which(name)
        if found:
            return found
        raise SettingsError(
            f"Cannot find {exe}. Set ffmpeg_bin in settings.yaml to the folder that contains it."
        )


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw or raw[key] in (None, ""):
        raise SettingsError(f"settings.yaml: '{key}' is required")
    return raw[key]


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load settings from `path`, $BGMI_SETTINGS, or ./settings.yaml."""
    path = Path(path or os.environ.get("BGMI_SETTINGS") or DEFAULT_SETTINGS_FILE)
    if not path.exists():
        raise SettingsError(
            f"{path} not found. Copy settings.example.yaml to settings.yaml and edit the paths."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SettingsError(f"{path} must be a YAML mapping")

    vt_raw = raw.get("videotype") or {}
    videotype = VideoTypeSettings(
        samples=int(vt_raw.get("samples", 24)),
        skip_head_frac=float(vt_raw.get("skip_head_frac", 0.05)),
        min_score=float(vt_raw.get("min_score", 0.35)),
    )
    aliases = {str(k).strip().lower(): str(v) for k, v in (raw.get("stage_aliases") or {}).items()}
    exts = tuple(e.lower() for e in raw.get("video_extensions", Settings.video_extensions))

    settings = Settings(
        videos_root=Path(_require(raw, "videos_root")),
        project_root=Path(raw.get("project_root") or path.resolve().parent),
        data_root=Path(_require(raw, "data_root")),
        ffmpeg_bin=str(raw.get("ffmpeg_bin") or ""),
        sample_every_s=float(raw.get("sample_every_s", 3)),
        keep_frames=bool(raw.get("keep_frames", False)),
        min_free_gb=float(raw.get("min_free_gb", 5)),
        workers=int(raw.get("workers", 14)),
        mongo_uri=os.environ.get("BGMI_MONGO_URI") or str(raw.get("mongo_uri") or ""),
        video_extensions=exts,
        stage_aliases=aliases,
        videotype=videotype,
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    vr, dr = s.videos_root.resolve(), s.data_root.resolve()
    if dr == vr or vr in dr.parents:
        raise SettingsError("data_root must not be inside videos_root (videos_root is read-only)")
    if s.min_free_gb < 0 or s.sample_every_s <= 0 or s.workers < 1:
        raise SettingsError("min_free_gb >= 0, sample_every_s > 0 and workers >= 1 are required")
    if not 0 <= s.videotype.skip_head_frac < 0.5:
        raise SettingsError("videotype.skip_head_frac must be in [0, 0.5)")
