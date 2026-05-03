from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def config_path() -> Path:
    return Path.home() / ".video_cliper" / "config.json"


def _read_all() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_last_export_dir() -> Optional[str]:
    value = _read_all().get("last_export_dir")
    if isinstance(value, str) and value:
        return value
    return None


def save_last_export_dir(directory: str) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_all()
    data["last_export_dir"] = directory
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
