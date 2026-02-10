from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class KnowledgeTier(StrEnum):
    GUARDRAILS = "GUARDRAILS"
    STYLE = "STYLE"
    RECENT = "RECENT"


TIER_FILES = {
    KnowledgeTier.GUARDRAILS: "GUARDRAILS.md",
    KnowledgeTier.STYLE: "STYLE.md",
    KnowledgeTier.RECENT: "RECENT.md",
}


class FileStorage:
    def __init__(self, agent_dir: Path):
        self.agent_dir = agent_dir

    def read_tier(self, tier: KnowledgeTier) -> str:
        path = self.agent_dir / TIER_FILES[tier]
        if path.exists():
            return path.read_text()
        return ""

    def write_tier(self, tier: KnowledgeTier, content: str) -> None:
        path = self.agent_dir / TIER_FILES[tier]
        path.write_text(content)

    def read_config(self) -> dict[str, Any]:
        path = self.agent_dir / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    def write_config(self, config: dict[str, Any]) -> None:
        path = self.agent_dir / "config.yaml"
        path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
