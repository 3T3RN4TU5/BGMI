import pytest

from bgmi.catalog import filename_hint, make_video_id, normalize_stage, parse_day, scan


@pytest.mark.parametrize("raw,expected", [
    ("Quater Finals", "Quarter Finals"),
    ("Quater", "Quarter Finals"),
    ("Quarter Finals", "Quarter Finals"),
    ("QF", "Quarter Finals"),
    ("SF", "Semi Finals"),
    ("Semi Finals", "Semi Finals"),
    ("Grand Finals", "Finals"),
    ("grand_final", "Finals"),
    ("Finals", "Finals"),
    ("Survival", "Survival"),
    ("league stage", "League"),
])
def test_normalize_stage(make_settings, raw, expected):
    s = make_settings(stage_aliases={**make_settings().stage_aliases, "quater": "Quarter Finals"})
    assert normalize_stage(raw, s.stage_aliases) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Day 1", ("Day 1", 1)), ("day3", ("Day 3", 3)), ("D2", ("Day 2", 2)),
    ("Day-12", ("Day 12", 12)), ("Finals", None), ("Day", None),
])
def test_parse_day(raw, expected):
    assert parse_day(raw) == expected


@pytest.mark.parametrize("name,hint", [
    ("BMSD Day 3 Map Stream.mp4", "map_stream"),
    ("map_stream_day3.mp4", "map_stream"),
    ("Grand Finals Day 3 Gameplay.mp4", "gameplay"),
    ("Grand Finals Day 3.mp4", "unknown"),
    ("roadmap.mp4", "unknown"),
])
def test_filename_hint(name, hint):
    assert filename_hint(name) == hint


def test_video_id_stable_and_path_separator_agnostic():
    assert make_video_id("A/B/Day 1/x.mp4", 10) == make_video_id("A\\B\\Day 1\\x.mp4", 10)
    assert make_video_id("A/B/Day 1/x.mp4", 10) != make_video_id("A/B/Day 1/x.mp4", 11)


def test_scan_layout(make_settings):
    s = make_settings()
    root = s.videos_root
    files = {
        "BMSD 2026/Finals/Day 3/map stream.mp4": b"a" * 10,
        "BMSD 2026/Finals/Day 3/gameplay.mp4": b"b" * 10,
        "BMSD 2026/Quater Finals/Day 1/Map Stream.MP4": b"c",
        "BMSD 2026/SF/vod.mp4": b"d",                    # no day folder: allowed
        "BMSD 2026/stray.mp4": b"e",                     # too shallow: skipped
        "BMSD 2026/Finals/Extra/clip.mp4": b"f",         # bad day folder: skipped
        "BMSD 2026/Finals/Day 3/notes.txt": b"g",        # not a video: ignored
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    entries, skipped = scan(s)
    got = {(e.tournament, e.stage, e.day, e.path.name, e.type_hint) for e in entries}
    assert got == {
        ("BMSD 2026", "Finals", "Day 3", "map stream.mp4", "map_stream"),
        ("BMSD 2026", "Finals", "Day 3", "gameplay.mp4", "gameplay"),
        ("BMSD 2026", "Quarter Finals", "Day 1", "Map Stream.MP4", "map_stream"),
        ("BMSD 2026", "Semi Finals", None, "vod.mp4", "unknown"),
    }
    assert {p.path.name for p in skipped} == {"stray.mp4", "clip.mp4"}


def test_scan_missing_root(make_settings, tmp_path):
    s = make_settings(videos_root=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        scan(s)
