from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class ViewSelectModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    VIEWS = [
        ("overview", "Overview - High-level repository status"),
        ("llm", "LLM - Provider and model configuration"),
        ("knowledge", "Knowledge - Base artifacts and indexing"),
        ("settings", "Settings - Runtime and interface settings"),
        ("timeline", "Timeline - Iteration outcomes and summaries"),
        ("ralph", "Ralph - Ralph loop status and outcomes"),
        ("console", "Console - Recent command output history"),
        ("all", "All - Show all dashboard panels together"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="view_modal"):
                yield Static("Change View", classes="modal_title")
                yield Static("Select a dashboard view to display", classes="modal_subtitle")

                yield OptionList(id="view_modal_options")

    def on_mount(self) -> None:
        option_list = self.query_one("#view_modal_options", OptionList)
        options = [Option(desc, id=f"view:{vid}") for vid, desc in self.VIEWS]
        option_list.set_options(options)
        option_list.focus()

    def on_key(self, event: events.Key) -> None:
        if event.key not in {"up", "down"}:
            return
        event.prevent_default()
        event.stop()
        direction = -1 if event.key == "up" else 1
        self._move_highlight(direction)

    def _move_highlight(self, direction: int) -> None:
        option_list = self.query_one("#view_modal_options", OptionList)
        options = option_list.options
        if not options:
            return

        highlighted = option_list.highlighted
        if highlighted is None:
            index = 0 if direction > 0 else len(options) - 1
        else:
            index = highlighted + direction

        while 0 <= index < len(options):
            if not options[index].disabled:
                option_list.highlighted = index
                break
            index += direction

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "view_modal_options":
            view_id = str(event.option.id).replace("view:", "")
            self.dismiss(view_id)
