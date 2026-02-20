from __future__ import annotations

from agent_recall.cli.tui.commands.local_router import handle_local_command


class _DummyApp:
    def __init__(self) -> None:
        self.current_view = "overview"
        self.status = ""
        self.actions: list[str] = []
        self.activity: list[str] = []
        self.refresh_count = 0
        self.backend_commands: list[str] = []

    def _append_activity(self, text: str) -> None:
        self.activity.append(text)

    def _refresh_dashboard_panel(self) -> None:
        self.refresh_count += 1

    def _run_backend_command(self, command: str, bypass_local: bool = False) -> None:
        self.backend_commands.append(command)

    def action_open_setup_modal(self) -> None:
        self.actions.append("setup")

    def action_open_model_modal(self) -> None:
        self.actions.append("model")

    def action_open_settings_modal(self) -> None:
        self.actions.append("settings")

    def action_toggle_terminal_panel(self) -> None:
        self.actions.append("terminal")

    def action_open_theme_modal(self) -> None:
        self.actions.append("theme")

    def action_open_session_run_modal(self) -> None:
        self.actions.append("run_select")

    def action_open_prd_select_modal(self) -> None:
        self.actions.append("ralph_select")

    def action_open_diff_viewer(self) -> None:
        self.actions.append("ralph_diff")

    def action_toggle_ralph_notifications(self) -> None:
        self.actions.append("ralph_notify")

    def action_open_ralph_config_modal(self) -> None:
        self.actions.append("ralph_config")


def test_local_router_setup_and_model_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "setup") is True
    assert handle_local_command(app, "config model") is True
    assert app.actions == ["setup", "model"]


def test_local_router_settings_preferences_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "settings") is True
    assert handle_local_command(app, "preferences") is True
    assert handle_local_command(app, "config preferences") is True
    assert app.actions == ["settings", "settings", "settings"]
    assert app.activity.count("Opened settings dialog.") == 3


def test_local_router_view_and_menu_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "view timeline") is True
    assert app.current_view == "timeline"
    assert app.status == "View: timeline"
    assert app.refresh_count == 1

    assert handle_local_command(app, "menu overview") is True
    assert app.current_view == "overview"
    assert app.refresh_count == 2


def test_local_router_theme_and_run_select_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "theme") is True
    assert handle_local_command(app, "theme list") is True
    assert handle_local_command(app, "run select") is True

    assert app.actions == ["theme", "theme", "run_select"]
    assert app.status == "Select conversations"


def test_local_router_ralph_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "ralph select") is True
    assert handle_local_command(app, "ralph view-diff") is True
    assert handle_local_command(app, "ralph notifications") is True
    assert handle_local_command(app, "ralph config") is True
    assert handle_local_command(app, "ralph terminal") is True

    assert app.actions == [
        "ralph_select",
        "ralph_diff",
        "ralph_notify",
        "ralph_config",
        "terminal",
    ]


def test_local_router_ralph_hooks_and_watch_commands() -> None:
    app = _DummyApp()

    assert handle_local_command(app, "ralph watch") is True
    assert handle_local_command(app, "ralph hooks install") is True
    assert handle_local_command(app, "ralph hooks uninstall") is True
    assert handle_local_command(app, "ralph plugin opencode-install") is True
    assert handle_local_command(app, "ralph plugin opencode-uninstall") is True

    assert app.backend_commands == [
        "ralph watch",
        "ralph hooks install",
        "ralph hooks uninstall",
        "ralph plugin opencode-install",
        "ralph plugin opencode-uninstall",
    ]


def test_local_router_non_local_command_returns_false() -> None:
    app = _DummyApp()
    assert handle_local_command(app, "status") is False
