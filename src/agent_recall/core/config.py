from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_recall.storage.models import AgentRecallConfig


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_config(agent_dir: Path) -> AgentRecallConfig:
    """Load and resolve .agent/config.yaml with optional extends."""
    main_path = agent_dir / "config.yaml"
    data = _load_yaml(main_path)

    extends = data.get("extends") or []
    merged: dict[str, Any] = {}

    for extend_path in extends:
        path = Path(extend_path).expanduser()
        if not path.is_absolute():
            path = (agent_dir / path).resolve()
        merged = _deep_merge(merged, _load_yaml(path))

    merged = _deep_merge(merged, data)
    return AgentRecallConfig.model_validate(merged)
