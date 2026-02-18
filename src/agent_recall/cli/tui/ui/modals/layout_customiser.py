from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Select, Static

LAYOUT_WIDGETS: list[tuple[str, str]] = [
    ("knowledge", "Knowledge Base"),
    ("sources", "Session Sources"),
    ("timeline", "Timeline"),
    ("ralph", "Ralph Status"),
    ("llm", "LLM Config"),
    ("settings", "Settings"),
]

BANNER_SIZE_OPTIONS: list[tuple[str, str]] = [
    ("hidden", "hidden"),
    ("compact", "compact"),
    ("normal", "normal"),
    ("large", "large"),
]


def default_widget_visibility() -> dict[str, bool]:
    return {key: True for key, _ in LAYOUT_WIDGETS}


def normalize_banner_size(value: object) -> str:
    size = str(value).strip().lower() if value is not None else ""
    valid = {option for option, _ in BANNER_SIZE_OPTIONS}
    return size if size in valid else "normal"


class LayoutCustomiserModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, widget_visibility: dict[str, bool], banner_size: str):
        super().__init__()
        self.widget_visibility = dict(widget_visibility)
        self.banner_size = normalize_banner_size(banner_size)

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="layout_modal"):
                yield Static("Customise Layout", classes="modal_title")
                yield Static(
                    "Toggle visible widgets and banner size",
                    classes="modal_subtitle",
                )
                yield Static("Visible Widgets", classes="section_title")
                for key, label in LAYOUT_WIDGETS:
                    yield Checkbox(
                        label,
                        value=bool(self.widget_visibility.get(key, True)),
                        id=f"layout_widget_{key}",
                    )
                yield Static("Title Banner Size", classes="section_title")
                yield Select(
                    BANNER_SIZE_OPTIONS,
                    value=self.banner_size,
                    allow_blank=False,
                    id="layout_banner_size",
                    classes="field_input",
                )
                yield Static("", id="layout_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Apply", variant="primary", id="layout_apply")
                    yield Button("Cancel", id="layout_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "layout_cancel":
            self.dismiss(None)
            return
        if event.button.id != "layout_apply":
            return

        error_widget = self.query_one("#layout_error", Static)
        banner_value = self.query_one("#layout_banner_size", Select).value
        if banner_value == Select.BLANK:
            error_widget.update("[red]Select a banner size[/red]")
            return

        widgets: dict[str, bool] = {}
        for key, _label in LAYOUT_WIDGETS:
            widgets[key] = bool(self.query_one(f"#layout_widget_{key}", Checkbox).value)

        self.dismiss(
            {
                "widgets": widgets,
                "banner_size": normalize_banner_size(banner_value),
            }
        )
