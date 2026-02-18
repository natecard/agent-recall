from __future__ import annotations

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class IterationDetailModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        *,
        title: str,
        summary_text: str,
        outcome_text: str,
        item_text: str,
        diff_text: str,
    ) -> None:
        super().__init__()
        self.title = title
        self.summary_text = summary_text
        self.outcome_text = outcome_text
        self.item_text = item_text
        self.diff_text = diff_text

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static(self.title or "Iteration Detail", classes="modal_title")
                yield Static(
                    "Use PgUp/PgDn or arrows to scroll. Esc closes.",
                    classes="modal_subtitle",
                )
                yield Static(id="iteration_detail_content")

    def on_mount(self) -> None:
        detail_widget = self.query_one("#iteration_detail_content", Static)
        body = (
            "## Summary\n"
            f"Outcome: {self.outcome_text}\n"
            f"Item: {self.item_text}\n\n"
            f"{self.summary_text}\n\n"
            "## Diff\n"
            f"{self.diff_text}"
        )
        detail_widget.update(Syntax(body, "markdown", word_wrap=False))
