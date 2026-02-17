from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich import box
from rich.panel import Panel
from rich.table import Table


@dataclass(frozen=True)
class KnowledgeWidget:
    repo_name: str
    stats: dict[str, int]
    guardrails_len: int
    style_len: int
    recent_len: int
    total_tokens: int
    total_cost_usd: float
    format_usd: Callable[[float], str]

    def render(self) -> Panel:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Item", style="table_header", width=12, no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("Repository", self.repo_name)
        table.add_row("Processed", str(self.stats.get("processed_sessions", 0)))
        table.add_row("Logs", str(self.stats.get("log_entries", 0)))
        table.add_row("Chunks", str(self.stats.get("chunks", 0)))
        table.add_row("GUARDRAILS", f"{self.guardrails_len:,} chars")
        table.add_row("STYLE", f"{self.style_len:,} chars")
        table.add_row("RECENT", f"{self.recent_len:,} chars")
        table.add_row("Tokens", f"{self.total_tokens:,}")
        table.add_row("Cost", self.format_usd(self.total_cost_usd))
        return Panel(table, title="Knowledge Base", border_style="accent")
