from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from agent_recall.cli.tui.logic.text_sanitizers import _source_checkbox_id
from agent_recall.ingest.sources import SOURCE_DEFINITIONS


class SetupModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]

    def __init__(self, *, defaults: dict[str, Any]):
        super().__init__()
        self.defaults = defaults

    def compose(self) -> ComposeResult:
        selected_agents = {
            source for source in self.defaults.get("selected_agents", []) if isinstance(source, str)
        }
        repo_path = str(self.defaults.get("repository_path", Path.cwd()))

        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Repository Setup", classes="modal_title")
                yield Static(
                    "Step 1 of 3 · Repository and session sources",
                    classes="modal_subtitle",
                )
                yield Static(repo_path, id="setup_repo_path")
                with Horizontal(classes="field_row"):
                    yield Checkbox(
                        "Use this repository",
                        value=bool(self.defaults.get("repository_verified", True)),
                        id="setup_repository_verified",
                    )
                    yield Checkbox(
                        "Force reconfigure",
                        value=bool(self.defaults.get("force", False)),
                        id="setup_force",
                    )
                with Horizontal(classes="setup_agents field_row"):
                    for source in SOURCE_DEFINITIONS:
                        yield Checkbox(
                            source.display_name,
                            value=source.name in selected_agents,
                            id=_source_checkbox_id(source.name),
                        )
                yield Static("", id="setup_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Next", variant="primary", id="setup_next")
                    yield Button("Cancel", id="setup_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup_cancel":
            self.dismiss(None)
            return
        if event.button.id != "setup_next":
            return

        status = self.query_one("#setup_status", Static)

        repository_verified = bool(self.query_one("#setup_repository_verified", Checkbox).value)
        if not repository_verified:
            status.update("[error]Repository must be confirmed[/error]")
            return

        selected_agents: list[str] = []
        for source in SOURCE_DEFINITIONS:
            checkbox = self.query_one(f"#{_source_checkbox_id(source.name)}", Checkbox)
            if checkbox.value:
                selected_agents.append(source.name)
        if not selected_agents:
            status.update("[error]Choose at least one agent source[/error]")
            return

        self.dismiss(
            {
                "force": bool(self.query_one("#setup_force", Checkbox).value),
                "repository_verified": repository_verified,
                "selected_agents": selected_agents,
            }
        )
