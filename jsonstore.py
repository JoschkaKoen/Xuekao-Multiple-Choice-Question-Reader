"""Atomic JSON cache helpers — a unit is 'done' iff its file exists AND parses."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash mid-write can't leave a half file


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def valid(path: Path) -> bool:
    return read_json(path) is not None
