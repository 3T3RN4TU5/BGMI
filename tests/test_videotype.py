"""Video type detection on synthetic frames that mimic the BMSD layout."""
import numpy as np
import pytest

from bgmi.overlays import load_overlays
from bgmi.settings import load_settings  # noqa: F401
from bgmi.videotype import classify_frames, detect_type, row_periodicity, sample_times
from tests.conftest import ROOT, requires_ffmpeg

W, H = 1280, 720
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def overlays():
    return load_overlays(ROOT / "config")


def map_frame(alive_noise=True):
    """Static 16-row team table on the left, changing map in the middle."""
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :292] = (30, 25, 20)
    for i in range(16):
        y = 54 + i * 40
        f[y:y + 30, 8:284] = (70, 60, 50)              # row panel
        f[y + 8:y + 22, 20:120] = (220, 220, 220)      # team tag text block
    if alive_noise:                                    # a few alive bars flicker
        i = RNG.integers(0, 16)
        f[54 + i * 40 + 10:54 + i * 40 + 20, 220:280] = (0, 200, 0)
    f[:, 292:987] = RNG.integers(40, 160, (H, 695, 3), dtype=np.uint8)   # map + zoom changes
    f[:, 987:] = (25, 20, 15)
    return f


def desk_frame():
    """Analyst desk / player cams: same shot repeats but no team table."""
    f = np.full((H, W, 3), 90, np.uint8)
    f[200:600, 100:500] = (150, 120, 100)
    f += RNG.integers(0, 6, (H, W, 3), dtype=np.uint8)
    return f


def gameplay_frame():
    return RNG.integers(0, 255, (H, W, 3), dtype=np.uint8)


def test_sample_times():
    t = sample_times(1000, 5, 0.1)
    assert t[0] == pytest.approx(100) and t[-1] == pytest.approx(900) and len(t) == 5
    assert sample_times(0, 5, 0.1) == []


def test_row_periodicity(overlays):
    ov = overlays["kie_bmsd2026"]
    sb = ov["layouts"]["map_view"]["sidebar_left"]
    assert row_periodicity(map_frame(), sb, (0, 0, 292, 720)) > 0.5
    assert row_periodicity(desk_frame(), sb, (0, 0, 292, 720)) < 0.2


def test_map_stream_with_cutaways_is_map_stream(overlays):
    frames = [map_frame() for _ in range(12)] + [desk_frame() for _ in range(8)] + \
             [gameplay_frame() for _ in range(4)]          # replays/ads
    vtype, score, ov = classify_frames(frames, overlays, 0.35)
    assert vtype == "map_stream" and ov == "kie_bmsd2026"
    assert score == pytest.approx(0.5)


def test_gameplay_with_desk_segments_is_not_map_stream(overlays):
    frames = [gameplay_frame() for _ in range(16)] + [desk_frame() for _ in range(8)]
    vtype, score, _ = classify_frames(frames, overlays, 0.35)
    assert vtype == "gameplay" and score == 0


def test_detect_type_hint_conflict_and_unknown(overlays):
    frames = iter([map_frame() for _ in range(24)])
    res = detect_type(3600, "gameplay", lambda t: next(frames), overlays, 24, 0.05, 0.35)
    assert res.video_type == "map_stream" and res.hint_conflict
    res2 = detect_type(3600, "map_stream", lambda t: None, overlays, 24, 0.05, 0.35)
    assert res2.video_type == "unknown" and res2.frames_ok == 0


@requires_ffmpeg
def test_end_to_end_on_tiny_video(make_settings, tmp_path, overlays):
    """Encode a tiny synthetic map stream, then probe + detect it through ffmpeg."""
    import cv2

    from bgmi.media import probe
    from bgmi.videotype import detect_type_for_file

    s = make_settings()
    path = s.videos_root / "T" / "Finals" / "Day 1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (W, H))
    for i in range(40):
        vw.write(map_frame() if i % 4 else desk_frame())
    vw.release()
    info = probe(s, path)
    assert (info.width, info.height) == (W, H) and info.duration_s == pytest.approx(20, abs=1)
    res = detect_type_for_file(s, path, info.duration_s, "unknown", overlays)
    assert res.video_type == "map_stream", res
