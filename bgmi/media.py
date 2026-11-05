"""Thin ffprobe / ffmpeg wrappers. Read-only on input videos."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import Settings


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"timed out: {' '.join(cmd[:3])} ...") from exc


def probe(settings: Settings, path: Path) -> ProbeInfo:
    cmd = [
        settings.tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    res = _run(cmd, timeout=60)
    if res.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {res.stderr.decode(errors='replace').strip()}")
    data = json.loads(res.stdout or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"no video stream in {path}")
    st = streams[0]
    return ProbeInfo(
        duration_s=float((data.get("format") or {}).get("duration") or 0.0),
        fps=_parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate")),
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
        codec=str(st.get("codec_name") or ""),
    )


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def grab_frame(settings: Settings, path: Path, t_s: float, size: tuple[int, int] = (1280, 720)) -> np.ndarray | None:
    """Decode one frame near t_s (fast keyframe seek), scaled to `size`, as BGR uint8."""
    w, h = size
    cmd = [
        settings.tool("ffmpeg"), "-v", "error", "-ss", f"{max(t_s, 0):.3f}", "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    res = _run(cmd, timeout=60)
    if res.returncode != 0 or len(res.stdout) != w * h * 3:
        return None
    return np.frombuffer(res.stdout, dtype=np.uint8).reshape(h, w, 3)
