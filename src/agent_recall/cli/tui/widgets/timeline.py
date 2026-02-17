from __future__ import annotations

from dataclasses import dataclass

from rich.panel import Panel


@dataclass(frozen=True)
class TimelineWidget:
    timeline_lines: list[str]

    def render(self) -> Panel:
        return Panel(
            "\n".join(self.timeline_lines),
            title="Iteration Timeline",
            border_style="accent",
        )
