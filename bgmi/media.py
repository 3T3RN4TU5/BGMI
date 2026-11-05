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


def iter_frames(settings: Settings, path: Path, start_s: float, end_s: float, every_s: float,
                size: tuple[int, int] = (1280, 720)):
    """Yield (t_s, BGR frame) every `every_s` seconds in [start_s, end_s), streamed from one ffmpeg.

    Frames go through a pipe and are never written to disk.
    """
    w, h = size
    dur = max(end_s - start_s, 0)
    if dur <= 0 or every_s <= 0:
        return
    cmd = [
        settings.tool("ffmpeg"), "-v", "error", "-ss", f"{max(start_s, 0):.3f}", "-i", str(path),
        "-t", f"{dur:.3f}", "-vf", f"fps=1/{every_s:g},scale={w}:{h}", "-an",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    n = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        i = 0
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield start_s + i * every_s, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
