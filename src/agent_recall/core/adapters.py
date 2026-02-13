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


def _trim_context(context: str, token_budget: int | None) -> str:
    if token_budget is None:
        return context
    if token_budget <= 0:
        return context
    budget_chars = token_budget * 4
    if len(context) <= budget_chars:
        return context
    return context[:budget_chars]


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
    token_budget: int | None = None,
    per_adapter_budgets: dict[str, int] | None = None,
    per_provider_budgets: dict[str, int] | None = None,
    per_model_budgets: dict[str, int] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    adapter_list = list(adapters) if adapters is not None else get_default_adapters()
    per_adapter = per_adapter_budgets or {}
    per_provider = per_provider_budgets or {}
    per_model = per_model_budgets or {}
    normalized_provider = provider.strip().lower() if isinstance(provider, str) else None
    normalized_model = model.strip() if isinstance(model, str) else None

    for adapter in adapter_list:
        budget = per_adapter.get(adapter.name)
        if budget is None and normalized_model:
            budget = per_model.get(normalized_model)
        if budget is None and normalized_provider:
            budget = per_provider.get(normalized_provider)
        if budget is None:
            budget = token_budget
        trimmed_context = _trim_context(context, budget)
        payload = build_adapter_payload(
            adapter=adapter,
            context=trimmed_context,
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
