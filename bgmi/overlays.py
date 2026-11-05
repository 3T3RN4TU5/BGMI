"""Load broadcaster overlay configs from config/overlays/*.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class OverlayError(ValueError):
    pass


def load_overlays(config_dir: Path) -> dict[str, dict[str, Any]]:
    folder = Path(config_dir) / "overlays"
    overlays: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        oid = data.get("id")
        if not oid:
            raise OverlayError(f"{f.name}: 'id' is required")
        if oid in overlays:
            raise OverlayError(f"duplicate overlay id '{oid}' in {f.name}")
        res = data.get("resolution")
        if not (isinstance(res, list) and len(res) == 2):
            raise OverlayError(f"{f.name}: 'resolution: [w, h]' is required")
        overlays[oid] = data
    return overlays


def box(overlay: dict[str, Any], *keys: str) -> tuple[int, int, int, int]:
    """Read an [x1, y1, x2, y2] box by key path, e.g. box(o, 'layouts', 'map_view', 'viewport')."""
    node: Any = overlay
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise OverlayError(f"overlay '{overlay.get('id')}' has no {'.'.join(keys)}")
        node = node[k]
    if isinstance(node, dict):
        node = node.get("box")
    if not (isinstance(node, list) and len(node) == 4):
        raise OverlayError(f"overlay '{overlay.get('id')}': {'.'.join(keys)} must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (int(v) for v in node)
    if not (x2 > x1 and y2 > y1):
        raise OverlayError(f"overlay '{overlay.get('id')}': {'.'.join(keys)} has an empty box")
    return x1, y1, x2, y2
