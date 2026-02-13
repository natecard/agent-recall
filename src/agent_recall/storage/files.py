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
    def __init__(self, agent_dir: Path, shared_tiers_dir: Path | None = None):
        self.agent_dir = agent_dir
        self.shared_tiers_dir = shared_tiers_dir

    def _local_tier_path(self, tier: KnowledgeTier) -> Path:
        return self.agent_dir / TIER_FILES[tier]

    def _shared_tier_path(self, tier: KnowledgeTier) -> Path | None:
        if self.shared_tiers_dir is None:
            return None
        return self.shared_tiers_dir / TIER_FILES[tier]

    def read_tier(self, tier: KnowledgeTier) -> str:
        shared_path = self._shared_tier_path(tier)
        if shared_path is not None and shared_path.exists():
            return shared_path.read_text()
        local_path = self._local_tier_path(tier)
        if local_path.exists():
            return local_path.read_text()
        return ""

    def write_tier(self, tier: KnowledgeTier, content: str) -> None:
        local_path = self._local_tier_path(tier)
        local_path.write_text(content)
        shared_path = self._shared_tier_path(tier)
        if shared_path is not None and shared_path != local_path:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_path.write_text(content)

    def read_config(self) -> dict[str, Any]:
        path = self.agent_dir / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    def write_config(self, config: dict[str, Any]) -> None:
        path = self.agent_dir / "config.yaml"
        path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
