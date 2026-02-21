from __future__ import annotations

from collections.abc import Callable

from rich.syntax import Syntax
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class IterationDetailModal(ModalScreen[None]):
    DEFAULT_CSS = """
    IterationDetailModal #annotation_edit_container {
        height: auto;
        display: none;
        padding: 2;
        background: $surface-darken-2;
    }
    IterationDetailModal #annotation_edit_container.visible {
        display: block;
    }
    IterationDetailModal #annotation_edit_input {
        width: 100%;
    }
    IterationDetailModal #annotation_help_footer {
        height: auto;
        padding: 2;
        background: $surface;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "handle_escape", "Close"),
        Binding("a", "edit_annotation", "Add/Edit Note"),
    ]

    def __init__(
        self,
        *,
        iteration: int,
        title: str,
        summary_text: str,
        outcome_text: str,
        item_text: str,
        diff_text: str,
        annotation_text: str | None = None,
        on_save_annotation: Callable[[int, str], None] | None = None,
    ) -> None:
        super().__init__()
        self.iteration = iteration
        self.title = title
        self.summary_text = summary_text
        self.outcome_text = outcome_text
        self.item_text = item_text
        self.diff_text = diff_text
        self.annotation_text = annotation_text or ""
        self._on_save_annotation = on_save_annotation
        self._editing_annotation = False

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static(self.title or "Iteration Detail", classes="modal_title")
                yield Static(
                    "Use PgUp/PgDn or arrows to scroll. 'a' to add/edit note. Esc closes.",
                    classes="modal_subtitle",
                )
                yield Static(id="iteration_detail_content")
                with Vertical(id="annotation_edit_container"):
                    yield Input(
                        placeholder="Enter annotation...",
                        id="annotation_edit_input",
                    )
                yield Static(
                    "[dim]a[/dim] Add/Edit Note  [dim]Escape[/dim] Close",
                    id="annotation_help_footer",
                )

    def on_mount(self) -> None:
        self._render_content()

    def _render_content(self) -> None:
        detail_widget = self.query_one("#iteration_detail_content", Static)
        annotation_display = (
            self.annotation_text.strip() if self.annotation_text else "[dim]No annotation[/dim]"
        )
        body = (
            "## Summary\n"
            f"Outcome: {self.outcome_text}\n"
            f"Item: {self.item_text}\n\n"
            f"{self.summary_text}\n\n"
            "## Annotation\n"
            f"{annotation_display}\n\n"
            "## Diff\n"
            f"{self.diff_text}"
        )
        detail_widget.update(Syntax(body, "markdown", word_wrap=False))

    def action_edit_annotation(self) -> None:
        if self._editing_annotation:
            return
        container = self.query_one("#annotation_edit_container", Vertical)
        input_widget = self.query_one("#annotation_edit_input", Input)
        input_widget.value = self.annotation_text
        container.add_class("visible")
        self._editing_annotation = True
        input_widget.focus()

    def action_handle_escape(self) -> None:
        if self._editing_annotation:
            container = self.query_one("#annotation_edit_container", Vertical)
            container.remove_class("visible")
            self._editing_annotation = False
            return
        self.dismiss(None)

    @on(Input.Submitted, "#annotation_edit_input")
    def _on_annotation_submitted(self, event: Input.Submitted) -> None:
        if not self._editing_annotation:
            return
        new_text = event.value.strip()
        self.annotation_text = new_text
        if self._on_save_annotation:
            self._on_save_annotation(self.iteration, new_text)
        container = self.query_one("#annotation_edit_container", Vertical)
        container.remove_class("visible")
        self._editing_annotation = False
        self._render_content()
