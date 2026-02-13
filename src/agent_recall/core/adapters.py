from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ContextAdapter:
    name: str
    display_name: str

    def payload_path(self, output_dir: Path) -> Path:
        return output_dir / self.name / "context.json"


def get_default_adapters() -> list[ContextAdapter]:
    return [
        ContextAdapter(name="codex", display_name="OpenAI Codex"),
        ContextAdapter(name="cursor", display_name="Cursor"),
        ContextAdapter(name="claude-code", display_name="Claude Code"),
        ContextAdapter(name="opencode", display_name="OpenCode"),
    ]


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def build_adapter_payload(
    *,
    adapter: ContextAdapter,
    context: str,
    task: str | None,
    active_session_id: str | None,
    repo_path: Path,
    refreshed_at: datetime,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "adapter": adapter.name,
        "task": task,
        "active_session_id": active_session_id,
        "repo_path": str(repo_path),
        "refreshed_at": _normalize_timestamp(refreshed_at).isoformat(),
        "context": context,
    }


def write_adapter_payloads(
    *,
    context: str,
    task: str | None,
    active_session_id: str | None,
    repo_path: Path,
    refreshed_at: datetime,
    output_dir: Path,
    adapters: Iterable[ContextAdapter] | None = None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    adapter_list = list(adapters) if adapters is not None else get_default_adapters()

    for adapter in adapter_list:
        payload = build_adapter_payload(
            adapter=adapter,
            context=context,
            task=task,
            active_session_id=active_session_id,
            repo_path=repo_path,
            refreshed_at=refreshed_at,
        )
        payload_path = adapter.payload_path(output_dir)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2))
        written[adapter.name] = payload_path

    return written
