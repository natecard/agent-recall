from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage


class RalphStatus(StrEnum):
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"


@dataclass
class RalphState:
    status: RalphStatus = RalphStatus.DISABLED
    total_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "total_iterations": self.total_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RalphState:
        status_raw = str(data.get("status", RalphStatus.DISABLED.value))
        try:
            status = RalphStatus(status_raw)
        except ValueError:
            status = RalphStatus.DISABLED
        total_iterations = int(data.get("total_iterations") or 0)
        return cls(status=status, total_iterations=total_iterations)


class RalphStateManager:
    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self.state_path = agent_dir / "ralph" / "ralph_state.json"

    def load(self) -> RalphState:
        if not self.state_path.exists():
            return RalphState()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RalphState()
        if not isinstance(payload, dict):
            return RalphState()
        return RalphState.from_dict(payload)

    def save(self, state: RalphState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2),
            encoding="utf-8",
        )


class RalphLoop:
    def __init__(self, agent_dir: Path, storage: Storage, files: FileStorage) -> None:
        self.agent_dir = agent_dir
        self.storage = storage
        self.files = files
        self.state = RalphStateManager(agent_dir)

    def enable(self) -> RalphState:
        state = self.state.load()
        state.status = RalphStatus.ENABLED
        self.state.save(state)
        return state

    def disable(self) -> RalphState:
        state = self.state.load()
        state.status = RalphStatus.DISABLED
        self.state.save(state)
        return state

    def initialize_from_prd(self, prd_path: Path) -> int:
        data = json.loads(prd_path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else None
        item_count = len(items) if isinstance(items, list) else 0
        self.enable()
        return item_count
