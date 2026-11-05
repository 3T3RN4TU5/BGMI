import pytest

from bgmi.catalog import scan
from bgmi.jobs import Queue


def _setup(make_settings, names=("a.mp4", "b.mp4")):
    s = make_settings()
    for n in names:
        p = s.videos_root / "T" / "Finals" / "Day 1" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(n.encode())
    q = Queue(s.state_db)
    entries, _ = scan(s)
    q.upsert_catalog(entries)
    for e in entries:
        q.update_fields(e.video_id, video_type="map_stream")
    return s, q, entries


def test_full_lifecycle_and_idempotent_recatalog(make_settings):
    s, q, entries = _setup(make_settings)
    assert q.counts()["queued"] == 2
    row = q.claim_next()
    assert row is not None and q.get(row["video_id"])["status"] == "running"
    q.mark_done(row["video_id"])
    # re-cataloguing the same files keeps 'done'
    again, _ = scan(s)
    assert q.upsert_catalog(again) == {"new": 0, "updated": 2, "missing": 0}
    assert q.get(row["video_id"])["status"] == "done"
    assert q.counts() == {"queued": 1, "running": 0, "done": 1, "failed": 0}


def test_crash_recovery_requeues_running(make_settings):
    _, q, _ = _setup(make_settings)
    q.claim_next()
    assert q.recover_stale() == 1
    assert q.counts()["queued"] == 2


def test_failed_then_requeue(make_settings):
    _, q, _ = _setup(make_settings)
    row = q.claim_next()
    q.mark_failed(row["video_id"], "unknown overlay")
    assert q.get(row["video_id"])["error"] == "unknown overlay"
    assert q.requeue() == 1
    assert q.get(row["video_id"])["status"] == "queued"


def test_invalid_transition_rejected(make_settings):
    _, q, entries = _setup(make_settings)
    with pytest.raises(RuntimeError):
        q.mark_done(entries[0].video_id)   # still queued, not running


def test_claim_only_map_streams(make_settings):
    _, q, entries = _setup(make_settings)
    for e in entries:
        q.update_fields(e.video_id, video_type="gameplay")
    assert q.claim_next() is None


def test_missing_file_flagged(make_settings):
    s, q, entries = _setup(make_settings)
    entries[0].path.unlink()
    remaining, _ = scan(s)
    assert q.upsert_catalog(remaining)["missing"] == 1
    assert len(q.all()) == 1


def test_update_fields_whitelist(make_settings):
    _, q, entries = _setup(make_settings)
    with pytest.raises(ValueError):
        q.update_fields(entries[0].video_id, status="done")
