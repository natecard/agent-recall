from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.panel import Panel
from rich.table import Table


@dataclass(frozen=True)
class SettingsWidget:
    view: str
    ralph_agent_transport: str
    theme_name: str
    interactive_shell: bool
    repo_name: str
    active_agents_wrapped: str
    configured_agents_wrapped: str

    def render(self) -> Panel:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Setting", style="table_header")
        table.add_column("Value", overflow="fold")
        table.add_row("Current view", self.view)
        table.add_row("Ralph transport", self.ralph_agent_transport)
        table.add_row("Theme", self.theme_name)
        if self.view != "all":
            table.add_row(
                "Interactive shell",
                "yes" if self.interactive_shell else "no",
            )
            table.add_row("Repository", self.repo_name)
        table.add_row(
            "Active agents" if self.view == "all" else "Configured agents",
            self.active_agents_wrapped if self.view == "all" else self.configured_agents_wrapped,
        )
        return Panel(table, title="Settings", border_style="accent")
