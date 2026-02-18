"""Compact coding agent configuration modal for onboarding step 3."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from agent_recall.cli.tui.constants import _RALPH_CLI_OPTIONS
from agent_recall.cli.tui.logic.text_sanitizers import _clean_optional_text
from agent_recall.cli.tui.types import DiscoverCodingModelsFn


class CodingAgentStepModal(ModalScreen[dict[str, Any] | None]):
    """Focused coding agent config: CLI, model, Ralph. No scroll."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(
        self,
        defaults: dict[str, Any],
        discover_coding_models: DiscoverCodingModelsFn,
    ):
        super().__init__()
        self.defaults = defaults
        self.discover_coding_models = discover_coding_models

    def compose(self) -> ComposeResult:
        cli = _clean_optional_text(self.defaults.get("coding_cli", ""))
        if cli not in dict(_RALPH_CLI_OPTIONS):
            cli = ""
        model = _clean_optional_text(self.defaults.get("cli_model", ""))
        ralph = bool(self.defaults.get("ralph_enabled", bool(cli)))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="modal_compact"):
                yield Static("Coding Agent", classes="modal_title")
                yield Static("Step 3 of 3 · Ralph loop model", classes="modal_subtitle")
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("CLI", classes="field_label")
                    yield Select(
                        _RALPH_CLI_OPTIONS,
                        value=cli if cli else Select.BLANK,
                        allow_blank=True,
                        id="setup_coding_cli",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("Model", classes="field_label")
                    yield Select(
                        [("Manual entry", "__manual__")],
                        value="__manual__",
                        allow_blank=False,
                        id="setup_cli_model_picker",
                        classes="field_input",
                    )
                with Horizontal(classes="field_row field_row_compact"):
                    yield Static("", classes="field_label")
                    yield Input(
                        value=model,
                        placeholder="Select above or type",
                        id="setup_cli_model",
                        classes="field_input",
                    )
                yield Checkbox("Enable Ralph loop", value=ralph, id="setup_ralph_enabled")
                with Horizontal(classes="modal_actions"):
                    yield Button("Back", id="agent_back")
                    yield Button("Skip", id="agent_skip")
                    yield Button("Finish", variant="primary", id="agent_finish")
                    yield Button("Cancel", id="agent_cancel")

    def on_mount(self) -> None:
        self._refresh_models()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "setup_coding_cli":
            self._refresh_models()
            return
        if event.select.id == "setup_cli_model_picker":
            val = event.value
            if val == Select.BLANK or str(val) == "__manual__":
                return
            self.query_one("#setup_cli_model", Input).value = str(val)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent_cancel":
            self.dismiss(None)
            return
        if event.button.id == "agent_back":
            self.dismiss({"_action": "back"})
            return
        if event.button.id == "agent_skip":
            self.dismiss(
                {
                    "_action": "finish",
                    "configure_coding_agent": True,
                    "coding_cli": None,
                    "cli_model": None,
                    "ralph_enabled": False,
                }
            )
            return
        if event.button.id != "agent_finish":
            return

        cli_raw = self.query_one("#setup_coding_cli", Select).value
        cli = None if cli_raw == Select.BLANK else str(cli_raw)
        model = _clean_optional_text(self.query_one("#setup_cli_model", Input).value) or None
        if cli is None:
            model = None
        ralph = bool(self.query_one("#setup_ralph_enabled", Checkbox).value)
        if ralph and cli is None:
            return

        self.dismiss(
            {
                "_action": "finish",
                "configure_coding_agent": True,
                "coding_cli": cli,
                "cli_model": model,
                "ralph_enabled": ralph,
            }
        )

    def _refresh_models(self) -> None:
        cli_raw = self.query_one("#setup_coding_cli", Select).value
        cli = "" if cli_raw == Select.BLANK else str(cli_raw)
        models, _ = self.discover_coding_models(cli) if cli else ([], None)

        picker = self.query_one("#setup_cli_model_picker", Select)
        opts: list[tuple[str, str]] = [("Manual entry", "__manual__")]
        opts.extend((m, m) for m in models)
        picker.set_options(opts)

        current = _clean_optional_text(self.query_one("#setup_cli_model", Input).value)
        picker.value = current if current in models else "__manual__"

        inp = self.query_one("#setup_cli_model", Input)
        inp.placeholder = f"Model for {cli}" if cli else "Select CLI first"
