from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from agent_recall.cli.tui.constants import _RALPH_AGENT_TRANSPORT_OPTIONS


class SettingsModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        current_view: str,
        refresh_seconds: float,
        all_cursor_workspaces: bool,
        ralph_agent_transport: str = "pipe",
    ):
        super().__init__()
        self.current_view = current_view
        self.refresh_seconds = refresh_seconds
        self.all_cursor_workspaces = all_cursor_workspaces
        normalized_transport = str(ralph_agent_transport).strip().lower()
        self.ralph_agent_transport = (
            normalized_transport if normalized_transport in {"pipe", "auto", "pty"} else "pipe"
        )

    def compose(self) -> ComposeResult:
        views = [
            "overview",
            "sources",
            "llm",
            "knowledge",
            "settings",
            "timeline",
            "ralph",
            "console",
            "all",
        ]
        default_view = self.current_view if self.current_view in views else "overview"
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("TUI Settings", classes="modal_title")
                with Horizontal(classes="field_row"):
                    yield Static("Default view", classes="field_label")
                    yield Select(
                        [(view_name, view_name) for view_name in views],
                        value=default_view,
                        allow_blank=False,
                        id="settings_view",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Refresh", classes="field_label")
                    yield Input(
                        value=str(self.refresh_seconds),
                        placeholder="seconds (>=0.2)",
                        id="settings_refresh",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Ralph transport", classes="field_label")
                    yield Select(
                        _RALPH_AGENT_TRANSPORT_OPTIONS,
                        value=self.ralph_agent_transport,
                        allow_blank=False,
                        id="settings_ralph_transport",
                        classes="field_input",
                    )
                yield Static(
                    "[dim]pipe = most stable stream. auto = use PTY when available. "
                    "pty = force PTY (falls back to pipe).[/dim]",
                    id="settings_transport_help",
                )
                yield Checkbox(
                    "Include all Cursor workspaces",
                    value=self.all_cursor_workspaces,
                    id="settings_all_cursor",
                )
                yield Static("", id="settings_error")
                with Horizontal(classes="modal_actions"):
                    yield Button("Save", variant="primary", id="settings_save")
                    yield Button("Cancel", id="settings_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings_cancel":
            self.dismiss(None)
            return
        if event.button.id != "settings_save":
            return

        error_widget = self.query_one("#settings_error", Static)

        view_widget = self.query_one("#settings_view", Select)
        selected_view = view_widget.value
        if selected_view == Select.BLANK:
            error_widget.update("[red]View selection is required[/red]")
            return

        try:
            refresh_seconds = float(self.query_one("#settings_refresh", Input).value)
        except ValueError:
            error_widget.update("[red]Refresh must be a number[/red]")
            return
        if refresh_seconds < 0.2:
            error_widget.update("[red]Refresh must be >= 0.2[/red]")
            return

        transport_widget = self.query_one("#settings_ralph_transport", Select)
        transport_value = transport_widget.value
        if transport_value == Select.BLANK:
            error_widget.update("[red]Ralph transport selection is required[/red]")
            return
        ralph_agent_transport = str(transport_value).strip().lower()
        if ralph_agent_transport not in {"pipe", "auto", "pty"}:
            error_widget.update("[red]Invalid Ralph transport selection[/red]")
            return

        self.dismiss(
            {
                "view": selected_view,
                "refresh_seconds": refresh_seconds,
                "ralph_agent_transport": ralph_agent_transport,
                "all_cursor_workspaces": bool(
                    self.query_one("#settings_all_cursor", Checkbox).value
                ),
            }
        )
