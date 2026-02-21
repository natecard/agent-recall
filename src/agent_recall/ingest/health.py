from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class SourceHealthResult:
    status: HealthStatus
    latency_ms: int | None = None
    last_seen_path: str | None = None
    error_message: str | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "last_seen_path": self.last_seen_path,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceHealthResult:
        status = HealthStatus(data.get("status", "unavailable"))
        latency_ms = data.get("latency_ms")
        last_seen_path = data.get("last_seen_path")
        error_message = data.get("error_message")
        checked_at_str = data.get("checked_at")
        if checked_at_str:
            try:
                checked_at = datetime.fromisoformat(checked_at_str)
            except (ValueError, TypeError):
                checked_at = datetime.now(UTC)
        else:
            checked_at = datetime.now(UTC)
        return cls(
            status=status,
            latency_ms=latency_ms,
            last_seen_path=last_seen_path,
            error_message=error_message,
            checked_at=checked_at,
        )


class SourceHealthStore:
    def __init__(self, agent_dir: Path):
        self.agent_dir = agent_dir
        self.health_file = agent_dir / "source_health.json"

    def load(self) -> dict[str, SourceHealthResult]:
        if not self.health_file.exists():
            return {}
        try:
            data = json.loads(self.health_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        results: dict[str, SourceHealthResult] = {}
        for source_name, result_data in data.items():
            if isinstance(result_data, dict):
                results[source_name] = SourceHealthResult.from_dict(result_data)
        return results

    def save(self, results: dict[str, SourceHealthResult]) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        data = {name: result.to_dict() for name, result in results.items()}
        self.health_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, source_name: str) -> SourceHealthResult | None:
        results = self.load()
        return results.get(source_name)

    def set(self, source_name: str, result: SourceHealthResult) -> None:
        results = self.load()
        results[source_name] = result
        self.save(results)
