from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from agent_recall.cli.tui.constants import (
    _COMPACT_MODE_OPTIONS,
    _RALPH_CLI_DEFAULT_MODELS,
    _RALPH_CLI_OPTIONS,
)
from agent_recall.cli.tui.logic.text_sanitizers import _clean_optional_text


class RalphConfigModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, defaults: dict[str, Any]):
        super().__init__()
        self.defaults = defaults

    def compose(self) -> ComposeResult:
        current_cli = _clean_optional_text(self.defaults.get("coding_cli", ""))
        current_model = _clean_optional_text(self.defaults.get("cli_model", ""))
        max_iter = self.defaults.get("max_iterations", 10)
        sleep_sec = self.defaults.get("sleep_seconds", 2)
        compact = _clean_optional_text(self.defaults.get("compact_mode", "always")) or "always"
        enabled = bool(self.defaults.get("enabled", False))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Ralph Configuration", classes="modal_title")
                yield Static(
                    "Coding agent, model, and loop behavior",
                    classes="modal_subtitle",
                )
                with Horizontal(classes="field_row"):
                    yield Static("Coding CLI", classes="field_label")
                    yield Select(
                        _RALPH_CLI_OPTIONS,
                        value=(
                            current_cli if current_cli in dict(_RALPH_CLI_OPTIONS) else Select.BLANK
                        ),
                        allow_blank=True,
                        id="ralph_cli",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("CLI Model", classes="field_label")
                    yield Input(
                        value=current_model,
                        placeholder="Model name (optional)",
                        id="ralph_cli_model",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Model list", classes="field_label")
                    yield Select(
                        [("Manual entry", "__manual__")],
                        value="__manual__",
                        allow_blank=False,
                        id="ralph_model_picker",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Max iterations", classes="field_label")
                    yield Input(
                        value=str(max_iter),
                        placeholder=">=1",
                        id="ralph_max_iterations",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Sleep seconds", classes="field_label")
                    yield Input(
                        value=str(sleep_sec),
                        placeholder=">=0",
                        id="ralph_sleep_seconds",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row"):
                    yield Static("Compact mode", classes="field_label")
                    yield Select(
                        _COMPACT_MODE_OPTIONS,
                        value=compact,
                        allow_blank=False,
                        id="ralph_compact_mode",
                        classes="field_input",
                    )
                yield Checkbox(
                    "Enabled",
                    value=enabled,
                    id="ralph_enabled",
                )
                yield Static("", id="ralph_error")
                with Horizontal(classes="modal_actions"):
                    yield Button(
                        "Apply",
                        variant="primary",
                        id="ralph_apply",
                    )
                    yield Button("Cancel", id="ralph_cancel")

    def on_mount(self) -> None:
        self._refresh_model_picker()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ralph_cli":
            self._refresh_model_picker()
            return
        if event.select.id != "ralph_model_picker":
            return
        selected = event.value
        if selected == Select.BLANK:
            return
        if str(selected) == "__manual__":
            return
        self.query_one("#ralph_cli_model", Input).value = str(selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ralph_cancel":
            self.dismiss(None)
            return
        if event.button.id != "ralph_apply":
            return

        error_w = self.query_one("#ralph_error", Static)

        cli_widget = self.query_one("#ralph_cli", Select)
        cli_value = cli_widget.value
        coding_cli = None if cli_value == Select.BLANK else str(cli_value)

        try:
            max_iter = int(self.query_one("#ralph_max_iterations", Input).value)
        except ValueError:
            error_w.update("[red]Max iterations must be an integer[/red]")
            return
        if max_iter < 1:
            error_w.update("[red]Max iterations must be >= 1[/red]")
            return

        try:
            sleep_sec = int(self.query_one("#ralph_sleep_seconds", Input).value)
        except ValueError:
            error_w.update("[red]Sleep seconds must be an integer[/red]")
            return
        if sleep_sec < 0:
            error_w.update("[red]Sleep seconds must be >= 0[/red]")
            return

        compact_widget = self.query_one("#ralph_compact_mode", Select)
        compact_value = compact_widget.value
        compact_mode = "always" if compact_value == Select.BLANK else str(compact_value)

        cli_model = _clean_optional_text(self.query_one("#ralph_cli_model", Input).value) or None

        self.dismiss(
            {
                "coding_cli": coding_cli,
                "cli_model": cli_model,
                "max_iterations": max_iter,
                "sleep_seconds": sleep_sec,
                "compact_mode": compact_mode,
                "enabled": bool(self.query_one("#ralph_enabled", Checkbox).value),
            }
        )

    def _refresh_model_picker(self) -> None:
        cli_widget = self.query_one("#ralph_cli", Select)
        cli_value = cli_widget.value
        cli_name = "" if cli_value == Select.BLANK else str(cli_value)

        models = _RALPH_CLI_DEFAULT_MODELS.get(cli_name, [])
        picker = self.query_one("#ralph_model_picker", Select)
        options: list[tuple[str, str]] = [("Manual entry", "__manual__")]
        options.extend((name, name) for name in models)
        picker.set_options(options)

        current_model = _clean_optional_text(self.query_one("#ralph_cli_model", Input).value)
        if current_model in models:
            picker.value = current_model
        else:
            picker.value = "__manual__"
