import pytest

from bgmi.diskguard import (DiskSpaceError, UnsafePathError, assert_writable, ensure_free_space,
                            frames_dir, safe_rmtree)
from bgmi.settings import SettingsError


def test_refuses_when_disk_low(make_settings):
    s = make_settings(min_free_gb=10**9)
    with pytest.raises(DiskSpaceError):
        ensure_free_space(s)


def test_passes_when_disk_ok(make_settings):
    assert ensure_free_space(make_settings(min_free_gb=0)) >= 0


def test_never_touches_videos_root(make_settings):
    s = make_settings()
    video = s.videos_root / "T" / "x.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"v")
    with pytest.raises(UnsafePathError):
        assert_writable(s, video)
    with pytest.raises(UnsafePathError):
        safe_rmtree(s, video.parent)
    assert video.exists()


def test_rmtree_only_inside_data_root(make_settings, tmp_path):
    s = make_settings()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(UnsafePathError):
        safe_rmtree(s, outside)
    with pytest.raises(UnsafePathError):
        safe_rmtree(s, s.data_root)          # never the whole data_root
    assert outside.exists()


def test_frames_dir_cleaned_unless_keep(make_settings):
    s = make_settings(keep_frames=False)
    with frames_dir(s, "vid1") as d:
        (d / "f.png").write_bytes(b"x")
    assert not d.exists()
    s2 = make_settings(keep_frames=True)
    with frames_dir(s2, "vid2") as d2:
        (d2 / "f.png").write_bytes(b"x")
    assert d2.exists()


def test_data_root_inside_videos_root_rejected(make_settings, tmp_path):
    with pytest.raises(SettingsError):
        make_settings(data_root=str(tmp_path / "videos" / "data"))
