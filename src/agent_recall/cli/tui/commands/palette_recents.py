from __future__ import annotations

import json
from pathlib import Path


def load_recents(config_dir: Path, max_items: int = 8) -> list[str]:
    recents_file = config_dir / "palette_recents.json"
    if not recents_file.exists():
        return []
    try:
        data = json.loads(recents_file.read_text())
        if not isinstance(data, list):
            return []
        recents = [str(item) for item in data if isinstance(item, str) and item.strip()]
        return recents[:max_items]
    except (json.JSONDecodeError, OSError):
        return []


def record_recent(config_dir: Path, action_id: str, max_items: int = 8) -> None:
    if not action_id.strip():
        return
    recents_file = config_dir / "palette_recents.json"
    config_dir.mkdir(parents=True, exist_ok=True)
    existing = load_recents(config_dir, max_items=max_items * 2)
    deduped = [action_id] + [r for r in existing if r != action_id]
    trimmed = deduped[:max_items]
    try:
        recents_file.write_text(json.dumps(trimmed, indent=2))
    except OSError:
        pass
