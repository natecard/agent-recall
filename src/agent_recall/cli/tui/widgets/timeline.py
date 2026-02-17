from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass(frozen=True)
class TimelineWidget:
    timeline_lines: list[str]
    detail_title: str | None = None
    detail_body: str | None = None

    def render(self, *, detail: bool = False) -> Panel:
        if detail:
            return Panel(
                self._render_detail(),
                title=self.detail_title or "Iteration Detail",
                border_style="accent",
            )
        return Panel(
            "\n".join(self.timeline_lines),
            title="Iteration Timeline",
            border_style="accent",
        )

    def _render_detail(self) -> Group:
        renderables = []
        body = (self.detail_body or "").strip()
        if not body:
            body = "No detail available."
        renderables.append(Syntax(body, "markdown", word_wrap=False))
        return Group(*renderables)
