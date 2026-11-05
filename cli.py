"""Drop Zone Intel command line.

    python cli.py doctor            check Python, GPU, ffmpeg, paths, disk
    python cli.py catalog           scan videos_root, probe and type-detect new videos
    python cli.py status            show the processing queue
    python cli.py requeue           put failed videos back in the queue
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
from bgmi.diskguard import DiskSpaceError, ensure_free_space, free_gb
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
