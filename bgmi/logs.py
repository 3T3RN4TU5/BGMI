"""Structured JSON-lines logging under data_root/logs (NFR-11)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(logs_dir: Path, run_name: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = logs_dir / f"{stamp}-{run_name}.jsonl"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("bgmi")
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    root.propagate = False
    return path


def log(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
