from __future__ import annotations

import shlex


def handle_local_command(app, raw: str) -> bool:
    command = raw.strip()
    if not command:
        return True
    normalized = command[1:] if command.startswith("/") else command
    try:
        parts = shlex.split(normalized)
    except ValueError as exc:
        app.status = "Invalid command"
        app._append_activity(f"Invalid command: {exc}")
        return True
    if not parts:
        return True

    action = parts[0].lower()
    second = parts[1].lower() if len(parts) > 1 else ""

    if action in {"setup", "onboarding"}:
        app.action_open_setup_modal()
        app.status = "Setup"
        return True

    if action == "model":
        app.action_open_model_modal()
        app.status = "Model configuration"
        return True

    if action == "config" and second == "setup":
        app.action_open_setup_modal()
        app.status = "Setup"
        return True

    if action == "config" and second == "model":
        app.action_open_model_modal()
        app.status = "Model configuration"
        return True

    if action in {"settings", "preferences"}:
        app.action_open_settings_modal()
        app.status = "Settings"
        app._append_activity("Opened settings dialog.")
        return True

    if action == "layout":
        app.action_open_layout_modal()
        app.status = "Layout"
        app._append_activity("Opened layout dialog.")
        return True

    if action == "config" and second in {"settings", "preferences", "prefs"}:
        app.action_open_settings_modal()
        app.status = "Settings"
        app._append_activity("Opened settings dialog.")
        return True

    if action == "ralph" and second == "terminal":
        app.action_toggle_terminal_panel()
        app.status = "Terminal panel"
        return True

    if action in {"view", "menu"}:
        valid = {
            "overview",
            "sources",
            "llm",
            "knowledge",
            "settings",
            "timeline",
            "ralph",
            "console",
            "all",
        }
        if len(parts) == 1:
            app._append_activity(
                f"Current view: {app.current_view}. Available: {', '.join(sorted(valid))}"
            )
            return True
        requested = parts[1].strip().lower()
        if requested not in valid:
            app._append_activity(f"Unknown view '{requested}'.")
            return True
        app.current_view = requested
        app.status = f"View: {requested}"
        app._append_activity(f"Switched to {requested} view.")
        app._refresh_dashboard_panel()
        return True

    if action == "theme":
        if len(parts) == 1:
            app.action_open_theme_modal()
            return True
        if second == "list":
            app.action_open_theme_modal()
            return True
        if second == "set" and len(parts) == 2:
            app.action_open_theme_modal()
            return True
    if action == "run" and len(parts) > 1 and parts[1].lower() == "select":
        app.action_open_session_run_modal()
        app.status = "Select conversations"
        return True

    if action == "ralph" and second == "select":
        app.action_open_prd_select_modal()
        app.status = "Select PRD items"
        return True

    if action == "ralph" and second == "view-diff":
        app.action_open_diff_viewer()
        app.status = "View iteration diff"
        return True

    if action == "ralph" and second in {"notify", "notifications"}:
        app.action_toggle_ralph_notifications()
        app.status = "Ralph notifications"
        return True

    if action == "ralph" and second == "config":
        app.action_open_ralph_config_modal()
        app.status = "Ralph configuration"
        return True

    return False
