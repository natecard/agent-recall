from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.panel import Panel
from rich.table import Table


@dataclass(frozen=True)
class SourcesWidget:
    source_table: Table
    compact_lines: list[str]
    last_synced_display: str
    compact: bool = False

    def render(self) -> Panel:
        if self.compact:
            lines = [*self.compact_lines, f"[dim]Last Synced:[/dim] {self.last_synced_display}"]
            content = "\n".join(lines)
        else:
            content = Group(
                self.source_table, f"[dim]Last Synced:[/dim] {self.last_synced_display}"
            )
        return Panel(content, title="Session Sources", border_style="accent")
