import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bgmi.settings import load_settings  # noqa: E402


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides):
        videos = tmp_path / "videos"
        data = tmp_path / "data"
        videos.mkdir(exist_ok=True)
        raw = yaml.safe_load((ROOT / "settings.example.yaml").read_text(encoding="utf-8"))
        raw.update({"videos_root": str(videos), "data_root": str(data),
                    "project_root": str(ROOT), "ffmpeg_bin": ""})
        raw.update(overrides)
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_settings(path)
    return _make


requires_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH")
