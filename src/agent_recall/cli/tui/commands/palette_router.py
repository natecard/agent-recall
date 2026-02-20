from __future__ import annotations


def _normalize_action_id(action_id: str) -> str:
    """Strip duplicate suffix (e.g. status:1 -> status) from deduplicated option IDs."""
    if ":" in action_id:
        parts = action_id.rsplit(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return action_id


def handle_palette_action(app, action_id: str | None) -> None:
    if not action_id:
        return
    action_id = _normalize_action_id(action_id)
    if action_id == "view-select":
        app.action_open_view_modal()
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
        "sessions": "sessions",
    }
    command = command_by_action.get(action_id)
    if command:
        app._run_backend_command(command)
