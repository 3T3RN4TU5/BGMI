"""PRD build phase 1 on synthetic frames: scene, zone, team table, segmentation."""
import json

import numpy as np
import pytest

from bgmi.analyze import run
from bgmi.scene import classify_scene
from bgmi.segment import FrameObs, number_matches, segment
from bgmi.sidebar import GlyphBook, learn, read_sidebar
from bgmi.zone import ZoneResult, detect_zone, flag_jumps
from tests import synth


@pytest.fixture(scope="module")
def ov():
    return synth.overlay()


# ---- scene -----------------------------------------------------------------

def test_scene_classifier(ov):
    assert classify_scene(synth.map_view(ov), ov).scene == "map_view"
    assert classify_scene(synth.match_start(ov), ov).scene == "match_start"
    res = classify_scene(synth.desk(), ov)
    assert res.scene == "other" and res.confidence == 1.0


def test_scene_match_start_tolerates_eliminated_tiles(ov):
    f = synth.match_start(ov)
    for p in ov["scene"]["match_start"]["probes"][:3]:     # three teams out: tiles turn red
        x1, y1, x2, y2 = p["box"]
        f[y1:y2 + 10, x1:x2] = (40, 40, 220)
    assert classify_scene(f, ov).scene == "match_start"
    for p in ov["scene"]["match_start"]["probes"][3:6]:
        x1, y1, x2, y2 = p["box"]
        f[y1:y2 + 10, x1:x2] = (40, 40, 220)
    res = classify_scene(f, ov)
    assert res.scene == "other" and res.confidence < 1.0


# ---- zone ------------------------------------------------------------------

@pytest.mark.parametrize("ring", [(640, 360, 200), (600, 300, 90), (640, 700, 260)])
def test_zone_finds_ring(ov, ring):
    f = synth.map_view(ov, ring=ring)
    z = detect_zone(f, ov)
    assert z.found, z
    assert z.cx == pytest.approx(ring[0], abs=3) and z.cy == pytest.approx(ring[1], abs=3)
    assert z.r == pytest.approx(ring[2], abs=3)


def test_zone_none_without_ring(ov):
    f = synth.map_view(ov)
    f[300, 300:900] = 255            # a straight white line (grid) is not a zone
    z = detect_zone(f, ov)
    assert not z.found and z.cx is None and z.confidence == 0


def test_flag_jumps_keeps_values():
    a = ZoneResult(True, 100, 100, 50, 0.9, 0.9, "")
    b = ZoneResult(True, 400, 400, 200, 0.9, 0.9, "")
    out = flag_jumps([a, a, b, a, a])
    assert out[2].cx == 400 and out[2].confidence == pytest.approx(0.45)
    assert out[0].confidence == 0.9


# ---- team table -------------------------------------------------------------

def _rows(order, fins, bars):
    return list(zip(order, fins, bars))


def test_sidebar_reads_shuffled_table(ov):
    tags = synth.TAGS
    fins = [f"{i % 8:02d}" for i in range(16)]
    book = GlyphBook()
    learn(book, synth.map_view(ov, _rows(tags, fins, ["AAAA"] * 16)), ov, tags, fins)
    rng = np.random.default_rng(3)
    order = list(rng.permutation(tags))
    fins2 = [f"{(i * 3) % 8:02d}" for i in range(16)]
    bars = ["AAAA", "AKDD", "DDDD", "AADD"] * 4
    reads = read_sidebar(synth.map_view(ov, _rows(order, fins2, bars)), ov, book)
    assert [r.tag for r in reads] == order
    assert [r.fin for r in reads] == [int(x) for x in fins2]
    assert [r.alive for r in reads] == [4, 2, 0, 2] * 4
    assert [r.eliminated for r in reads] == [False, False, True, False] * 4
    assert reads[1].bars == ("alive", "knocked", "dead", "dead")


def test_sidebar_unknown_digit_is_none_not_a_guess(ov):
    tags = synth.TAGS
    book = GlyphBook()
    learn(book, synth.map_view(ov, _rows(tags, ["00"] * 16, ["AAAA"] * 16)), ov, tags, ["00"] * 16)
    reads = read_sidebar(synth.map_view(ov, _rows(tags, ["07"] * 16, ["AAAA"] * 16)), ov, book)
    assert all(r.fin is None for r in reads)
    assert all(r.tag == t for r, t in zip(reads, tags))


def test_sidebar_without_templates_reads_bars_only(ov):
    reads = read_sidebar(synth.map_view(ov), ov, GlyphBook())
    assert all(r.tag is None and r.alive == 4 for r in reads)
    assert all(r.confidence == 0 for r in reads)


def test_learn_rejects_bad_labels(ov):
    with pytest.raises(ValueError):
        learn(GlyphBook(), synth.map_view(ov), ov, ["A", "B"])


def test_glyphbook_roundtrip(tmp_path, ov):
    book = GlyphBook()
    learn(book, synth.map_view(ov), ov, synth.TAGS, ["00"] * 16)
    p = tmp_path / "g.npz"
    book.save(p)
    back = GlyphBook.load(p)
    assert back.labels("tag") == sorted(synth.TAGS) and back.labels("digit") == ["0"]


# ---- segmentation -----------------------------------------------------------

def _sig(v):
    s = np.zeros((15, 13), np.float32)
    s[3:12, 3 + v:6 + v] = 1
    return s


def test_segment_two_matches_with_cutaways():
    obs = [FrameObs(0, "other")]
    obs += [FrameObs(30, "match_start")]
    obs += [FrameObs(33 + 3 * i, "map_view", _sig(0)) for i in range(150)]          # match A
    obs += [FrameObs(500 + i, "other") for i in range(0, 40, 3)]                   # 40 s cam cutaway
    obs += [FrameObs(540 + 3 * i, "map_view", _sig(0)) for i in range(50)]
    obs += [FrameObs(700 + 3 * i, "other") for i in range(100)]                     # desk
    obs += [FrameObs(1000, "match_start")]
    obs += [FrameObs(1003 + 3 * i, "map_view", _sig(4)) for i in range(150)]       # match B
    segs = [s for s in segment(obs) if s.kept]
    assert len(segs) == 2
    assert segs[0].start_t == 30 and segs[0].end_t == 540 + 3 * 49
    assert segs[1].start_t == 1000 and segs[1].n_start == 1


def test_segment_number_change_without_gap_does_not_split():
    obs = [FrameObs(3 * i, "map_view", _sig(0 if i != 50 else 4)) for i in range(200)]
    assert len([s for s in segment(obs) if s.kept]) == 1


def test_segment_drops_short_runs():
    obs = [FrameObs(3 * i, "map_view", _sig(0)) for i in range(10)]
    segs = segment(obs)
    assert len(segs) == 1 and not segs[0].kept and "dropped" in segs[0].note


def test_number_matches_schedule_and_read_anchor():
    book = GlyphBook()
    book.add("matchno:16", _sig(4))
    obs = [FrameObs(3 * i, "map_view", _sig(0)) for i in range(150)]
    obs += [FrameObs(1000 + 3 * i, "map_view", _sig(4)) for i in range(150)]
    obs += [FrameObs(2000 + 3 * i, "map_view", _sig(8)) for i in range(150)]
    segs = segment(obs)
    number_matches(segs, book, {"first_match": 13, "maps": ["Rondo", "Erangel", "Erangel", "Erangel", "Miramar"]})
    assert [(s.number, s.number_source, s.map_name) for s in segs] == [
        (15, "counted", "Erangel"), (16, "read", "Erangel"), (17, "counted", "Miramar")]


def _one_match(sig_id=0, n=150):
    return segment([FrameObs(3 * i, "map_view", _sig(sig_id)) for i in range(n)])


SCHED = {"first_match": 13, "maps": ["Rondo", "Erangel", "Erangel", "Erangel", "Miramar"]}


def test_number_unknown_for_partial_range_without_read():
    # a run from 2:05:00 must not assume it starts at the day's first match
    segs = _one_match()
    number_matches(segs, GlyphBook(), SCHED, from_video_start=False)
    assert (segs[0].number, segs[0].number_source, segs[0].map_name) == (None, "unknown", None)
    assert "--first-match" in segs[0].note


def test_number_given_first_match_and_schedule_from_start():
    segs = _one_match()
    number_matches(segs, GlyphBook(), SCHED, from_video_start=False, first_match=15)
    assert (segs[0].number, segs[0].number_source, segs[0].map_name) == (15, "given", "Erangel")
    segs = _one_match()
    number_matches(segs, GlyphBook(), SCHED, from_video_start=True)
    assert (segs[0].number, segs[0].number_source, segs[0].map_name) == (13, "schedule", "Rondo")


def test_number_read_by_vote_survives_bad_first_frame():
    book = GlyphBook()
    book.add("matchno:15", _sig(0))
    book.add("matchno:16", _sig(4))
    obs = [FrameObs(0, "map_view", np.zeros_like(_sig(0)))]       # a faded first frame
    obs += [FrameObs(3 + 3 * i, "map_view", _sig(0)) for i in range(150)]
    segs = segment(obs)
    number_matches(segs, book, SCHED, from_video_start=False, first_match=13)
    assert (segs[0].number, segs[0].number_source, segs[0].map_name) == (15, "read", "Erangel")
    assert segs[0].number_votes.endswith("frames read 15")


# ---- whole run ---------------------------------------------------------------

def test_run_writes_outputs(make_settings, ov):
    s = make_settings()
    tags = synth.TAGS
    book = GlyphBook()
    learn(book, synth.map_view(ov, _rows(tags, ["00"] * 16, ["AAAA"] * 16)), ov, tags, ["00"] * 16)
    frames = [(0.0, synth.desk()), (3.0, synth.match_start(ov))]
    frames += [(6.0 + 3 * i, synth.map_view(ov, ring=(640, 360, 200 - i))) for i in range(110)]
    frames += [(400.0, synth.desk())]
    res = run(s, {"video_id": "abc123"}, ov, book, iter(frames), {"first_match": 15, "maps": ["Erangel"]},
              debug_every=50)
    assert len(res.matches) == 1 and res.matches[0]["match_no"] == 15
    assert res.matches[0]["start_s"] == 3.0 and res.matches[0]["map"] == "Erangel"
    assert res.quality["zone_found_share"] == 1.0 and res.quality["tag_read_share"] == 1.0
    assert (res.out_dir / "frames.parquet").exists() and (res.out_dir / "teams.parquet").exists()
    assert json.loads((res.out_dir / "matches.json").read_text())["matches"][0]["map_frames"] == 110
    assert len(list(res.debug_dir.glob("*.jpg"))) >= 3
    assert len(res.teams) == 110 * 16 and set(res.teams["alive"]) == {4}


def test_long_full_map_screen_leaves_room_for_map_debug_images(make_settings, ov):
    s = make_settings()
    tags = synth.TAGS
    book = GlyphBook()
    learn(book, synth.map_view(ov, _rows(tags, ["00"] * 16, ["AAAA"] * 16)), ov, tags, ["00"] * 16)
    start = synth.match_start(ov)
    view = synth.map_view(ov, ring=(640, 360, 200))
    frames = [(3.0 * i, start) for i in range(80)]
    frames += [(240.0 + 3 * i, view) for i in range(110)]
    res = run(s, {"video_id": "abc124"}, ov, book, iter(frames), None, debug_every=20, max_debug=12)
    names = [p.name for p in res.debug_dir.glob("*.jpg")]
    assert any("map_view" in n for n in names)
    assert sum("match_start" in n for n in names) <= 4


def test_bar_meaning_is_per_overlay(ov):
    import copy
    ov2 = copy.deepcopy(ov)
    ov2["bars"]["meaning"] = {"red": "dead", "grey": "knocked"}
    reads = read_sidebar(synth.map_view(ov2, _rows(synth.TAGS, ["00"] * 16, ["AKDD"] * 16)), ov2, GlyphBook())
    assert reads[0].bars == ("alive", "dead", "knocked", "knocked") and reads[0].alive == 3
