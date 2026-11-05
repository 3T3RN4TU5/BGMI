"""Drop Zone Intel command line.

    python cli.py doctor            check Python, GPU, ffmpeg, paths, disk
    python cli.py catalog           scan videos_root, probe and type-detect new videos
    python cli.py status            show the processing queue
    python cli.py requeue           put failed videos back in the queue
    python cli.py calibrate         learn team-tag / digit templates from a labelled frame
    python cli.py analyze           phase 1: scenes, matches, zone and team table for a time range
    python cli.py scenes            sample frames for the scene-classifier check (contact sheets + CSV)
"""
from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from bgmi import __version__
from bgmi.catalog import scan
from bgmi.diskguard import DiskSpaceError, assert_writable, ensure_free_space, free_gb
from bgmi.jobs import Queue
from bgmi.logs import log, setup_logging
from bgmi.media import MediaError, probe
from bgmi.overlays import load_overlays
from bgmi.settings import Settings, SettingsError, load_settings
from bgmi.videotype import detect_type_for_file

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Drop Zone Intel pipeline")
console = Console()
logger = logging.getLogger("bgmi.cli")

SettingsOpt = typer.Option(None, "--settings", "-s", help="Path to settings.yaml")


def _settings(path: Optional[Path]) -> Settings:
    try:
        return load_settings(path)
    except SettingsError as exc:
        console.print(f"[red]Settings error:[/red] {exc}")
        raise typer.Exit(2)


def _fmt_dur(seconds: float | None) -> str:
    if not seconds:
        return "-"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


@app.command()
def doctor(settings_path: Optional[Path] = SettingsOpt) -> None:
    """Check the environment: Python, PyTorch + CUDA, ffmpeg, paths and free disk."""
    s = _settings(settings_path)
    ok = True
    table = Table(title=f"Drop Zone Intel {__version__} - doctor")
    table.add_column("Check")
    table.add_column("Result")

    def row(name: str, good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        table.add_row(name, f"[{'green' if good else 'red'}]{text}[/]")

    row("Python", sys.version_info >= (3, 11), platform.python_version())
    try:
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "no CUDA GPU visible"
        row("PyTorch", cuda, f"{torch.__version__} - {gpu}")
    except ImportError:
        row("PyTorch", False, "not installed (run scripts/setup.ps1)")
    for tool in ("ffmpeg", "ffprobe"):
        try:
            row(tool, True, s.tool(tool))
        except SettingsError as exc:
            row(tool, False, str(exc))
    row("videos_root", s.videos_root.is_dir(), str(s.videos_root))
    row("data_root", True, f"{s.data_root} ({'exists' if s.data_root.exists() else 'will be created'})")
    free = free_gb(s.data_root)
    row("Free disk (data drive)", free >= s.min_free_gb, f"{free:.1f} GB (minimum {s.min_free_gb:g} GB)")
    try:
        overlays = load_overlays(s.config_dir)
        row("Overlay configs", bool(overlays), ", ".join(overlays) or "none in config/overlays")
    except ValueError as exc:
        row("Overlay configs", False, str(exc))
    row("MongoDB URI", True, "set" if s.mongo_uri else "not set yet (needed from PRD build phase 4)")
    console.print(table)
    raise typer.Exit(0 if ok else 1)


@app.command()
def catalog(
    settings_path: Optional[Path] = SettingsOpt,
    detect: bool = typer.Option(True, "--detect/--no-detect", help="Sample frames to detect video type"),
    redetect: bool = typer.Option(False, help="Re-run type detection for every video"),
) -> None:
    """Scan videos_root and list every video with tournament, stage, day and type."""
    s = _settings(settings_path)
    log_path = setup_logging(s.logs_dir, "catalog")
    try:
        entries, skipped = scan(s)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    q = Queue(s.state_db)
    counts = q.upsert_catalog(entries)
    log(logger, "catalog scanned", **counts, skipped=len(skipped))

    overlays = load_overlays(s.config_dir) if detect else {}
    for e in entries:
        row = q.get(e.video_id)
        if row["duration_s"] is None:
            try:
                info = probe(s, e.path)
                q.update_fields(e.video_id, duration_s=info.duration_s, fps=info.fps,
                                width=info.width, height=info.height)
            except (MediaError, SettingsError) as exc:
                q.update_fields(e.video_id, error=f"probe: {exc}")
                log(logger, "probe failed", logging.WARNING, video_id=e.video_id, error=str(exc))
                continue
        row = q.get(e.video_id)
        if detect and (redetect or row["type_score"] is None):
            with console.status(f"Detecting type: {e.rel_path}"):
                res = detect_type_for_file(s, e.path, row["duration_s"] or 0, e.type_hint, overlays)
            detail = res.detail + (f"; filename suggests {res.hint}" if res.hint_conflict else "")
            q.update_fields(e.video_id, video_type=res.video_type, type_score=res.score,
                            type_detail=detail, overlay=res.overlay)
            log(logger, "video type", video_id=e.video_id, **res.__dict__)

    _print_videos(q, title=f"Catalog: {counts['new']} new, {counts['updated']} known, {counts['missing']} missing")
    for sk in skipped:
        console.print(f"[yellow]skipped[/yellow] {sk.path}: {sk.reason}")
    console.print(f"[dim]log: {log_path}[/dim]")


@app.command()
def status(settings_path: Optional[Path] = SettingsOpt) -> None:
    """Show every video and its processing status."""
    s = _settings(settings_path)
    q = Queue(s.state_db)
    _print_videos(q, title="Processing queue " + " · ".join(f"{k} {v}" for k, v in q.counts().items()))


@app.command()
def requeue(
    video_id: Optional[str] = typer.Argument(None, help="Only this video (default: all failed)"),
    settings_path: Optional[Path] = SettingsOpt,
) -> None:
    """Put failed videos back in the queue."""
    s = _settings(settings_path)
    n = Queue(s.state_db).requeue(video_id)
    console.print(f"Requeued {n} video(s).")


@app.command()
def diskcheck(settings_path: Optional[Path] = SettingsOpt) -> None:
    """Fail (exit 1) when the data drive is below min_free_gb."""
    s = _settings(settings_path)
    try:
        free = ensure_free_space(s)
        console.print(f"[green]{free:.1f} GB free[/green] (minimum {s.min_free_gb:g} GB)")
    except DiskSpaceError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _hms_to_s(text: Optional[str]) -> Optional[float]:
    """'2:05:00', '125:00' or '7500' -> seconds."""
    if text is None or text == "":
        return None
    parts = [float(p) for p in str(text).split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def _find_video(q: Queue, prefix: str):
    rows = [r for r in q.all() if r["video_id"].startswith(prefix)]
    if len(rows) != 1:
        console.print(f"[red]{'No' if not rows else 'More than one'} video matches id '{prefix}'.[/red] "
                      "Run `python cli.py status` for the ids.")
        raise typer.Exit(2)
    return rows[0]


def _overlay(s: Settings, video_row, overlay_id: Optional[str]) -> dict:
    overlays = load_overlays(s.config_dir)
    oid = overlay_id or (video_row["overlay"] if video_row is not None else None) or next(iter(overlays), None)
    if oid not in overlays:
        console.print(f"[red]Unknown overlay '{oid}'.[/red] Known: {', '.join(overlays) or 'none'}")
        raise typer.Exit(2)
    return overlays[oid]


def _schedule(s: Settings, video_row) -> Optional[dict]:
    import yaml  # noqa: PLC0415

    key = f"{video_row['stage']}/{video_row['day']}"
    for f in sorted((s.config_dir / "tournaments").glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if data.get("tournament") == video_row["tournament"] and key in (data.get("schedule") or {}):
            return data["schedule"][key]
    return None


@app.command()
def calibrate(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="A map-view frame (.jpg/.png, 1280x720)"),
    tags: str = typer.Option(..., help="Team tags top to bottom, space separated (16)"),
    fins: Optional[str] = typer.Option(None, help="FIN. values top to bottom, e.g. '04 02 01 ...'"),
    match_no: Optional[int] = typer.Option(None, help="Match number shown on this frame"),
    overlay_id: Optional[str] = typer.Option(None, "--overlay"),
    settings_path: Optional[Path] = SettingsOpt,
) -> None:
    """Learn team-tag, digit and match-number templates from one labelled frame.

    Templates are cut from the broadcast, so they are saved under data_root/calib (never in git).
    """
    import cv2  # noqa: PLC0415

    from bgmi.analyze import glyph_path  # noqa: PLC0415
    from bgmi.scene import classify_scene  # noqa: PLC0415
    from bgmi.segment import match_sig  # noqa: PLC0415
    from bgmi.sidebar import GlyphBook, learn  # noqa: PLC0415

    s = _settings(settings_path)
    ov = _overlay(s, None, overlay_id)
    frame = cv2.imread(str(image))
    w, h = ov["resolution"]
    if frame is None or frame.shape[:2] != (int(h), int(w)):
        console.print(f"[red]{image} must be a {w}x{h} image[/red]")
        raise typer.Exit(2)
    sc = classify_scene(frame, ov)
    if sc.scene != "map_view":
        console.print(f"[red]{image.name} does not look like the map view ({sc.detail}).[/red]")
        raise typer.Exit(2)
    path = glyph_path(s.data_root, ov["id"])
    book = GlyphBook.load(path)
    try:
        learn(book, frame, ov, tags.split(), fins.split() if fins else None)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    if match_no is not None:
        book.add(f"matchno:{match_no}", match_sig(frame, ov, "map_view"))
    book.save(path)
    console.print(f"Saved {sum(len(v) for v in book.items.values())} templates "
                  f"({len(book.labels('tag'))} teams, digits {''.join(book.labels('digit')) or '-'}, "
                  f"match numbers {', '.join(book.labels('matchno')) or '-'}) to {path}")


@app.command()
def analyze(
    video_id: str = typer.Argument(..., help="Video id or its first characters (see `status`)"),
    start: Optional[str] = typer.Option(None, help="Start time, e.g. 2:05:00 (default: video start)"),
    end: Optional[str] = typer.Option(None, help="End time, e.g. 2:40:00 (default: video end)"),
    every: Optional[float] = typer.Option(None, help="Seconds between sampled frames (default: sample_every_s)"),
    debug_every: int = typer.Option(20, help="Save a debug image every N map frames"),
    first_match: Optional[int] = typer.Option(None, help="Number of the first match in this range, "
                                              "used only when no match number can be read"),
    overlay_id: Optional[str] = typer.Option(None, "--overlay"),
    settings_path: Optional[Path] = SettingsOpt,
) -> None:
    """PRD build phase 1: scenes, match boundaries, zone and team table for a time range."""
    from bgmi.analyze import glyph_path, run  # noqa: PLC0415
    from bgmi.media import iter_frames  # noqa: PLC0415
    from bgmi.sidebar import GlyphBook  # noqa: PLC0415

    s = _settings(settings_path)
    log_path = setup_logging(s.logs_dir, "analyze")
    q = Queue(s.state_db)
    v = _find_video(q, video_id)
    ov = _overlay(s, v, overlay_id)
    book = GlyphBook.load(glyph_path(s.data_root, ov["id"]))
    if not book.labels("tag"):
        console.print("[yellow]No team templates yet: team names will be empty. "
                      "Run `python cli.py calibrate` first.[/yellow]")
    if not book.labels("matchno"):
        console.print("[yellow]No match-number templates: numbers come from --first-match or the schedule. "
                      "Run `python cli.py calibrate IMAGE --tags ... --match-no N` to read them.[/yellow]")
    try:
        ensure_free_space(s)
    except DiskSpaceError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    t0 = _hms_to_s(start) or 0.0
    t1 = _hms_to_s(end) or float(v["duration_s"] or 0)
    step = every or s.sample_every_s
    path = s.videos_root / v["rel_path"]
    schedule = _schedule(s, v)
    log(logger, "analyze start", video_id=v["video_id"], start=t0, end=t1, every=step)
    with console.status("Analysing...") as st:
        res = run(s, dict(v), ov, book, iter_frames(s, path, t0, t1, step), schedule,
                  debug_every=debug_every, from_video_start=t0 < 1.0, first_match=first_match,
                  progress=lambda t: st.update(f"Analysing {_fmt_dur(t)} of {_fmt_dur(t1)}"))
    log(logger, "analyze done", video_id=v["video_id"], **{k: v_ for k, v_ in res.quality.items() if k != "per_match"})

    table = Table(title="Matches found")
    for col in ("match", "map", "start", "end", "length", "map frames", "number from", "note"):
        table.add_column(col)
    for m in res.matches:
        src = m["match_no_source"] + (f" ({m['match_no_votes']})" if m.get("match_no_votes") else "")
        table.add_row("?" if m["match_no"] is None else str(m["match_no"]), m["map"] or "-",
                      _fmt_dur(m["start_s"]), _fmt_dur(m["end_s"]),
                      _fmt_dur(m["duration_s"]), str(m["map_frames"]), src, m["note"])
    console.print(table)
    qd = res.quality
    console.print(f"Frames: {qd.get('frames')}  scenes: {qd.get('scenes')}")
    console.print(f"Zone found on {(qd.get('zone_found_share') or 0):.0%} of map frames; "
                  f"team read {(qd.get('tag_read_share') or 0):.0%}, alive read {(qd.get('alive_read_share') or 0):.0%}, "
                  f"finishes read {(qd.get('fin_read_share') or 0):.0%} of table rows")
    console.print(f"Output: {res.out_dir}\nDebug images: {res.debug_dir}\n[dim]log: {log_path}[/dim]")


@app.command()
def grab(
    video_id: str = typer.Argument(..., help="Video id or its first characters"),
    at: list[str] = typer.Option(..., "--at", help="Time to grab, e.g. 2:21:24 (repeat --at)"),
    overlay_id: Optional[str] = typer.Option(None, "--overlay"),
    settings_path: Optional[Path] = SettingsOpt,
) -> None:
    """Save single frames to data_root/debug/<video>/grab/ and print scene + zone diagnostics."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from bgmi.media import iter_frames  # noqa: PLC0415
    from bgmi.overlays import box  # noqa: PLC0415
    from bgmi.scene import classify_scene  # noqa: PLC0415
    from bgmi.zone import detect_zone, ring_candidates  # noqa: PLC0415

    s = _settings(settings_path)
    v = _find_video(Queue(s.state_db), video_id)
    ov = _overlay(s, v, overlay_id)
    out = Path(s.debug_dir) / v["video_id"] / "grab"
    assert_writable(s, out)
    out.mkdir(parents=True, exist_ok=True)
    path = s.videos_root / v["rel_path"]
    for a in at:
        t = _hms_to_s(a) or 0.0
        got = next(iter(iter_frames(s, path, t, t + 1.0, 1.0)), None)
        if got is None:
            console.print(f"{a}: no frame")
            continue
        frame = got[1]
        stem = a.replace(":", "")
        cv2.imwrite(str(out / f"{stem}.png"), frame)
        sc = classify_scene(frame, ov)
        line = f"{a}: scene {sc.scene} ({sc.confidence:.2f})"
        if sc.scene in ("map_view", "match_start"):
            z = detect_zone(frame, ov, sc.scene)
            x1, y1, x2, y2 = (box(ov, "layouts", "map_view", "viewport") if sc.scene == "map_view"
                              else box(ov, "layouts", "match_start", "map_box"))
            m = ring_candidates(frame[y1:y2, x1:x2], ov.get("zone") or {})
            cv2.imwrite(str(out / f"{stem}_ringmask.png"), (m * 255).astype(np.uint8))
            line += f"; zone found={z.found} r={z.r} cov={z.coverage} ({z.detail}); ring px {int(m.sum())}"
        console.print(line)
    console.print(f"Saved to {out}")


@app.command()
def scenes(
    video_id: str = typer.Argument(..., help="Video id or its first characters"),
    n: int = typer.Option(200, help="Frames to sample across the video"),
    score: bool = typer.Option(False, help="Score the filled-in CSV instead of sampling"),
    overlay_id: Optional[str] = typer.Option(None, "--overlay"),
    settings_path: Optional[Path] = SettingsOpt,
) -> None:
    """Scene-classifier check: sample N frames, write contact sheets + a CSV to fill in, then --score."""
    import csv  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from bgmi.media import grab_frame  # noqa: PLC0415
    from bgmi.scene import classify_scene  # noqa: PLC0415

    s = _settings(settings_path)
    q = Queue(s.state_db)
    v = _find_video(q, video_id)
    out = s.debug_dir / v["video_id"] / "scenes"
    csv_path = out / "scene_check.csv"
    if score:
        if not csv_path.exists():
            console.print(f"[red]{csv_path} not found: run without --score first.[/red]")
            raise typer.Exit(2)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        done = [r for r in rows if (r.get("truth") or "").strip()]
        if not done:
            console.print("Fill the 'truth' column (map_view / match_start / other) first; leave it as is when right.")
            raise typer.Exit(2)
        wrong = [r for r in done if r["truth"].strip() != r["predicted"]]
        acc = 1 - len(wrong) / len(done)
        console.print(f"Scene accuracy: {acc:.1%} on {len(done)} labelled frames (gate: >= 98%)")
        for r in wrong:
            console.print(f"  #{r['idx']} at {r['time']}: predicted {r['predicted']}, truth {r['truth']}")
        raise typer.Exit(0 if acc >= 0.98 else 1)

    ov = _overlay(s, v, overlay_id)
    out.mkdir(parents=True, exist_ok=True)
    dur = float(v["duration_s"] or 0)
    times = [dur * (i + 0.5) / n for i in range(n)]
    w, h = (int(x) for x in ov["resolution"])
    thumbs, rows = [], []
    with console.status("Sampling frames...") as st:
        for i, t in enumerate(times):
            st.update(f"Frame {i + 1}/{n}")
            f = grab_frame(s, s.videos_root / v["rel_path"], t, (w, h))
            pred = classify_scene(f, ov).scene if f is not None else "unreadable"
            rows.append({"idx": i + 1, "time": _fmt_dur(t), "predicted": pred, "truth": pred})
            th = cv2.resize(f, (256, 144)) if f is not None else np.zeros((144, 256, 3), np.uint8)
            cv2.rectangle(th, (0, 0), (256, 18), (0, 0, 0), -1)
            cv2.putText(th, f"#{i + 1} {pred}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            thumbs.append(th)
    for k in range(0, len(thumbs), 25):
        page = thumbs[k:k + 25] + [np.zeros((144, 256, 3), np.uint8)] * (25 - len(thumbs[k:k + 25]))
        sheet = np.vstack([np.hstack(page[r * 5:(r + 1) * 5]) for r in range(5)])
        cv2.imwrite(str(out / f"sheet_{k // 25 + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["idx", "time", "predicted", "truth"])
        wr.writeheader()
        wr.writerows(rows)
    counts = {k: sum(r["predicted"] == k for r in rows) for k in ("map_view", "match_start", "other", "unreadable")}
    console.print(f"Predicted: {counts}")
    console.print(f"Contact sheets + CSV in {out}. Look through the sheets, fix the 'truth' column where the "
                  "label is wrong, then run with --score.")


def _print_videos(q: Queue, title: str) -> None:
    table = Table(title=title, show_lines=False)
    for col in ("id", "tournament", "stage", "day", "file", "type", "score", "duration", "status"):
        table.add_column(col)
    colors = {"map_stream": "cyan", "gameplay": "magenta", "unknown": "yellow"}
    for r in q.all():
        vt = r["video_type"]
        table.add_row(
            r["video_id"][:8], r["tournament"], r["stage"], r["day"] or "-", Path(r["rel_path"]).name,
            f"[{colors.get(vt, 'white')}]{vt}[/]",
            "-" if r["type_score"] is None else f"{r['type_score']:.2f}",
            _fmt_dur(r["duration_s"]),
            r["status"] + (f" ({r['error'][:40]})" if r["error"] else ""),
        )
    console.print(table)


if __name__ == "__main__":
    app()
