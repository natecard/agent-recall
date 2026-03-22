from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StatusInputs:
    stats: dict[str, Any]
    guardrails_chars: int
    style_chars: int
    recent_chars: int
    onboarding_complete: bool
    selected_sources: list[str] | None
    storage_backend: str
    shared_base_url: str
    compaction_backend: str
    adapter_enabled: bool
    adapter_output_dir: str
    adapter_token_budget: str
    adapter_per_adapter_budgets: str


def base_status_lines(inputs: StatusInputs) -> list[str]:
    shared_hint = (
        f" ({inputs.shared_base_url})"
        if inputs.storage_backend == "shared" and inputs.shared_base_url
        else ""
    )
    return [
        "[bold]Knowledge Base:[/bold]",
        f"  Processed sessions: {inputs.stats.get('processed_sessions', 0)}",
        f"  Log entries:        {inputs.stats.get('log_entries', 0)}",
        f"  Indexed chunks:     {inputs.stats.get('chunks', 0)}",
        "",
        "[bold]Tier Files:[/bold]",
        f"  GUARDRAILS.md: {inputs.guardrails_chars:,} chars",
        f"  STYLE.md:      {inputs.style_chars:,} chars",
        f"  RECENT.md:     {inputs.recent_chars:,} chars",
        "",
        "[bold]Onboarding:[/bold]",
        f"  Completed: {'yes' if inputs.onboarding_complete else 'no'}",
        f"  Agents:    {', '.join(inputs.selected_sources) if inputs.selected_sources else 'all'}",
        "",
        "[bold]Backends:[/bold]",
        f"  Storage:    {inputs.storage_backend}{shared_hint}",
        f"  Compaction: {inputs.compaction_backend}",
        (
            "  Next step: agent-recall sync --compact"
            if inputs.storage_backend == "local"
            else "  Next step: agent-recall sync --compact --source cursor"
        ),
        "",
        "[bold]Context Adapters:[/bold]",
        f"  Enabled: {'yes' if inputs.adapter_enabled else 'no'}",
        f"  Output dir: {inputs.adapter_output_dir}",
        f"  Token budget: {inputs.adapter_token_budget}",
        f"  Per-adapter budgets: {inputs.adapter_per_adapter_budgets}",
        "",
        "[bold]Rule Confidence:[/bold]",
    ]
