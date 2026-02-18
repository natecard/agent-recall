from __future__ import annotations


def handle_palette_action(app, action_id: str | None) -> None:
    if not action_id:
        return
    if action_id.startswith("view:"):
        app.current_view = action_id.split(":", 1)[1]
        app.status = f"View: {app.current_view}"
        app._append_activity(f"Switched to {app.current_view} view.")
        app._refresh_dashboard_panel()
        return

    if action_id == "setup":
        app.action_open_setup_modal()
        return
    if action_id == "model":
        app.action_open_model_modal()
        return
    if action_id == "settings":
        app.action_open_settings_modal()
        return
    if action_id == "layout":
        app.action_open_layout_modal()
        return
    if action_id == "ralph-config":
        app.action_open_ralph_config_modal()
        return
    if action_id == "ralph-run":
        app.action_run_ralph_loop()
        return
    if action_id == "ralph-view-diff":
        app.action_open_diff_viewer()
        return
    if action_id == "ralph-notifications":
        app.action_toggle_ralph_notifications()
        return
    if action_id == "ralph-terminal":
        app.action_toggle_terminal_panel()
        return
    if action_id == "theme":
        app.action_open_theme_modal()
        return
    if action_id == "run:select":
        app.action_open_session_run_modal()
        return
    if action_id == "quit":
        app.action_request_quit()
        return
    if action_id.startswith("run:"):
        app._run_backend_command(action_id.split(":", 1)[1])
        return
    if action_id.startswith("cmd:"):
        app._run_backend_command(action_id.split(":", 1)[1])
        return

    command_by_action = {
        "status": "status",
        "sync": "sync --no-compact",
        "knowledge-run": "run",
        "ralph-enable": "ralph enable",
        "ralph-disable": "ralph disable",
        "ralph-status": "ralph status",
        "ralph-select": "ralph select",
        "ralph-hooks-install": "ralph hooks install",
        "ralph-hooks-uninstall": "ralph hooks uninstall",
        "ralph-opencode-install": "ralph plugin opencode-install",
        "ralph-opencode-uninstall": "ralph plugin opencode-uninstall",
        "ralph-watch": "ralph watch",
        "sources": "sources",
        "sessions": "sessions",
    }
    command = command_by_action.get(action_id)
    if command:
        app._run_backend_command(command)
