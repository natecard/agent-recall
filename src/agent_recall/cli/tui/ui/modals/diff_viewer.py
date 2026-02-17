from __future__ import annotations

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class DiffViewerModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, *, diff_text: str, title: str):
        super().__init__()
        self.diff_text = diff_text
        self.title = title

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static(self.title or "Diff Viewer", classes="modal_title")
                yield Static(
                    "Use PgUp/PgDn or arrows to scroll. Esc closes.",
                    classes="modal_subtitle",
                )
                yield Static(id="diff_content")

    def on_mount(self) -> None:
        diff_widget = self.query_one("#diff_content", Static)
        syntax = Syntax(self.diff_text, "diff", word_wrap=False)
        diff_widget.update(syntax)
