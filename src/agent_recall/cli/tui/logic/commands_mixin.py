from __future__ import annotations

import importlib
import shlex
from pathlib import Path
from typing import Any

from textual import events
from textual.widgets import OptionList, Static

from agent_recall.cli.tui.commands.local_router import handle_local_command
from agent_recall.cli.tui.commands.palette_actions import (
    _is_knowledge_run_command,
    get_palette_actions,
)
from agent_recall.cli.tui.commands.palette_recents import load_recents
from agent_recall.cli.tui.commands.palette_router import handle_palette_action
from agent_recall.cli.tui.ui.modals import (
    LLMConfigStepModal,
    ModelConfigModal,
    RalphConfigModal,
    SettingsModal,
    SetupModal,
)
from agent_recall.cli.tui.ui.modals.coding_agent_step import CodingAgentStepModal
from agent_recall.cli.tui.ui.modals.command_palette import CommandPaletteModal
from agent_recall.cli.tui.ui.screens.diff_screen import IterationMetadata


class CommandsMixin:
    def action_toggle_terminal_panel(self: Any) -> None:
        if not self.terminal_supported:
            self._append_activity(
                "Embedded terminal not available. Run 'agent-recall tui --show-terminal'."
            )
            return
        self.terminal_panel_visible = not self.terminal_panel_visible
        self._update_terminal_panel_visibility(initial=False)
        try:
            from agent_recall.cli.main import _write_tui_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if agent_dir.exists():
                files = FileStorage(agent_dir)
                _write_tui_config(files, {"terminal_panel_visible": self.terminal_panel_visible})
        except Exception:  # noqa: BLE001
            pass

    def action_open_command_palette(self: Any) -> None:
        recents: list[str] = []
        config_dir: Path | None = None
        try:
            agent_dir = Path(".agent")
            if agent_dir.exists():
                config_dir = agent_dir
                recents = load_recents(config_dir)
        except Exception:  # noqa: BLE001
            pass
        self.push_screen(
            CommandPaletteModal(
                get_palette_actions(),
                recents=recents,
                config_dir=config_dir,
            ),
            self._handle_palette_action,
        )

    def action_open_settings_modal(self: Any) -> None:
        self.push_screen(
            SettingsModal(
                current_view=self.current_view,
                all_cursor_workspaces=self.all_cursor_workspaces,
                ralph_agent_transport=getattr(self, "ralph_agent_transport", "pipe"),
            ),
            self._apply_settings_modal_result,
        )

    def action_open_layout_modal(self: Any) -> None:
        self.push_screen(
            self._get_layout_modal_class()(
                widget_visibility=self.tui_widget_visibility,
                banner_size=self.tui_banner_size,
            ),
            self._apply_layout_modal_result,
        )

    def action_open_view_modal(self: Any) -> None:
        from agent_recall.cli.tui.ui.modals.view_select import ViewSelectModal

        self.push_screen(
            ViewSelectModal(),
            self._apply_view_modal_result,
        )

    def _apply_view_modal_result(self: Any, result: str | None) -> None:
        if not result:
            return
        self.current_view = result
        self.status = f"View: {self.current_view}"
        self._append_activity(f"Switched to {self.current_view} view.")
        self._refresh_dashboard_panel()

    def _get_layout_modal_class(self: Any):  # noqa: ANN001
        module = self._load_layout_module()
        return module.LayoutCustomiserModal

    def action_open_setup_modal(self: Any) -> None:
        self._pending_setup_payload = None
        self._open_setup_step_one_modal(self._setup_defaults_provider())

    def action_open_model_modal(self: Any) -> None:
        self.push_screen(
            ModelConfigModal(
                self._providers,
                self._model_defaults_provider(),
                self._discover_models,
                self._discover_coding_models,
            ),
            self._apply_model_modal_result,
        )

    def _open_setup_step_one_modal(self: Any, defaults: dict[str, Any]) -> None:
        self.push_screen(
            SetupModal(defaults=defaults),
            self._apply_setup_modal_result,
        )

    def _open_setup_step_two_modal(self: Any, defaults: dict[str, Any]) -> None:
        self.push_screen(
            LLMConfigStepModal(
                self._providers,
                defaults,
                self._discover_models,
            ),
            self._apply_setup_llm_modal_result,
        )

    def _open_setup_step_three_modal(self: Any, defaults: dict[str, Any]) -> None:
        self.push_screen(
            CodingAgentStepModal(
                defaults,
                self._discover_coding_models,
            ),
            self._apply_setup_coding_agent_modal_result,
        )

    def action_close_inline_picker(self: Any) -> None:
        if self._result_list_open:
            self._close_inline_result_list(announce=False)

    def on_resize(self: Any, event: events.Resize) -> None:
        _ = event
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
        # Debounce resize-driven refreshes to avoid event-loop saturation.
        self._resize_refresh_timer = self.set_timer(0.12, self._flush_resize_refresh)

    def _flush_resize_refresh(self: Any) -> None:
        self._resize_refresh_timer = None
        self._refresh_dashboard_panel()

    def action_refresh_now(self: Any) -> None:
        self._refresh_dashboard_panel()
        self._append_activity("Manual refresh complete.")

    def action_sync_conversations(self: Any) -> None:
        self._run_backend_command("sync --no-compact")

    def action_run_knowledge_update(self: Any) -> None:
        self._run_backend_command("run")

    def _teardown_runtime(self: Any) -> None:
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
            self._resize_refresh_timer = None
        self.workers.cancel_all()
        self._worker_context.clear()
        self._knowledge_run_workers.clear()

    def _refresh_dashboard_panel(self: Any) -> None:
        self._sync_runtime_theme()
        renderable = self._render_dashboard(
            all_cursor_workspaces=self.all_cursor_workspaces,
            include_banner_header=True,
            view=self.current_view,
            ralph_agent_transport=getattr(self, "ralph_agent_transport", "pipe"),
            show_slash_console=False,
        )
        self.query_one("#dashboard", Static).update(renderable)
        self._refresh_activity_panel()

    def _handle_palette_action(self: Any, action_id: str | None) -> None:
        handle_palette_action(self, action_id)

    def action_open_session_run_modal(self: Any) -> None:
        self.status = "Loading conversations"
        self._append_activity("Loading conversations for selection...")
        worker = self.run_worker(
            lambda: self._list_sessions_for_picker(200, self.all_cursor_workspaces),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "session_picker"

    def action_open_prd_select_modal(self: Any) -> None:
        if self._list_prd_items_for_picker is None:
            self.status = "PRD selection unavailable"
            self._append_activity("PRD selection is not configured.")
            return
        self.status = "Loading PRD items"
        self._append_activity("Loading PRD items for selection...")
        worker = self.run_worker(
            self._list_prd_items_for_picker,
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "prd_picker"

    def action_open_sessions_view_modal(self: Any, initial_filter: str = "") -> None:
        self.status = "Loading sessions"
        self._append_activity("Loading sessions for view...")
        worker = self.run_worker(
            lambda: self._get_sources_and_sessions_for_tui(200, self.all_cursor_workspaces),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = f"sessions_view:{initial_filter}"

    def _apply_sessions_view_modal_result(self: Any, result: dict[str, object] | None) -> None:
        if not result:
            return

        session_id = result.get("session_id")
        source = result.get("source")
        if not session_id or not source:
            return

        self.status = "Loading session..."
        self._append_activity(f"Loading {source} session details...")
        worker = self.run_worker(
            lambda: self._get_session_detail_for_tui(
                str(source), str(session_id), self.all_cursor_workspaces
            ),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = f"session_detail:{source}:{session_id}"

    def action_open_diff_viewer(self: Any) -> None:
        self.status = "Loading diff"
        self._append_activity("Loading latest iteration diff...")

        def _load_diff() -> tuple[str | None, int | None, IterationMetadata | None]:
            try:
                from agent_recall.ralph.iteration_store import IterationReportStore

                agent_dir = Path(".agent")
                store = IterationReportStore(agent_dir / "ralph")
                reports = store.load_recent(count=1)
                if not reports:
                    return None, None, None
                report = reports[0]
                diff_text = store.load_diff_for_iteration(report.iteration)
                meta = IterationMetadata(
                    iteration=report.iteration,
                    item_id=report.item_id,
                    item_title=report.item_title,
                    commit_hash=report.commit_hash,
                    completed_at=report.completed_at,
                    outcome=report.outcome.value if report.outcome else None,
                )
                return diff_text, report.iteration, meta
            except Exception:
                return None, None, None

        worker = self.run_worker(
            _load_diff,
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "diff_viewer"

    def action_open_ralph_config_modal(self: Any) -> None:
        """Open the Ralph configuration modal."""
        defaults: dict[str, Any] = {}
        try:
            from agent_recall.cli.ralph import read_ralph_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if agent_dir.exists():
                files = FileStorage(agent_dir)
                defaults = read_ralph_config(files)
        except Exception:  # noqa: BLE001
            pass
        self.push_screen(
            RalphConfigModal(
                defaults=defaults, discover_coding_models=self._discover_coding_models
            ),
            self._apply_ralph_config_modal_result,
        )

    def action_toggle_ralph_notifications(self: Any) -> None:
        """Toggle Ralph desktop notifications on/off."""
        try:
            from agent_recall.cli.ralph import read_ralph_config, write_ralph_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if not agent_dir.exists():
                self._append_activity("Not initialized. Run 'agent-recall init' first.")
                return
            files = FileStorage(agent_dir)
            ralph_cfg = read_ralph_config(files)
            notifications = ralph_cfg.get("notifications")
            if not isinstance(notifications, dict):
                notifications = {}
            current = bool(notifications.get("enabled"))
            notifications["enabled"] = not current
            updates: dict[str, object] = {"notifications": dict(notifications)}
            write_ralph_config(files, updates)
            state = "enabled" if notifications["enabled"] else "disabled"
            self._append_activity(f"Ralph notifications {state}.")
        except Exception as exc:  # noqa: BLE001
            self._append_activity(f"Failed to update notifications: {exc}")

    def _run_backend_command(self: Any, command: str, *, bypass_local: bool = False) -> None:
        if not bypass_local and self._handle_local_command(command):
            return
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
        self._append_activity(f"> {command}")
        self.status = f"Running: {command}"
        if _is_knowledge_run_command(command):
            self._append_activity("Knowledge run started. Loading...")
        viewport_width = max(int(self.size.width or 0), 80)
        viewport_height = max(int(self.size.height or 0), 24)
        worker = self.run_worker(
            lambda: self._execute_command(command, viewport_width, viewport_height),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        worker_key = id(worker)
        self._worker_context[worker_key] = "command"
        if _is_knowledge_run_command(command):
            self._knowledge_run_workers.add(worker_key)

    def _handle_local_command(self: Any, raw: str) -> bool:
        return handle_local_command(self, raw)

    def on_option_list_option_selected(self: Any, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "activity_result_list":
            option_id = event.option.id or ""
            if option_id.startswith("sync-source:"):
                source_name = option_id.split(":", 1)[1]
                if hasattr(self, "_run_source_sync"):
                    self._run_source_sync(source_name)
            return

    def _apply_setup_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            self._pending_setup_payload = None
            return
        self._pending_setup_payload = dict(result)
        model_defaults = dict(self._setup_defaults_provider())
        self.status = "Setup (step 2/3)"
        self._append_activity("Step 1 complete. Configure LLM.")
        self._open_setup_step_two_modal(model_defaults)

    def _apply_setup_llm_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            self._pending_setup_payload = None
            self.status = "Setup cancelled"
            self._append_activity("Setup cancelled.")
            return

        action = str(result.get("_action") or "").strip().lower()
        if action == "back":
            setup_defaults = dict(self._setup_defaults_provider())
            if isinstance(self._pending_setup_payload, dict):
                setup_defaults.update(self._pending_setup_payload)
            self.status = "Setup (step 1/3)"
            self._append_activity("Returned to step 1.")
            self._open_setup_step_one_modal(setup_defaults)
            return

        if action != "next":
            return

        merged: dict[str, Any] = {}
        if isinstance(self._pending_setup_payload, dict):
            merged.update(self._pending_setup_payload)
        merged.update(result)
        del merged["_action"]
        self._pending_setup_payload = merged

        self.status = "Setup (step 3/3)"
        self._append_activity("Configure coding agent.")
        self._open_setup_step_three_modal(merged)

    def _apply_setup_coding_agent_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            self._pending_setup_payload = None
            self.status = "Setup cancelled"
            self._append_activity("Setup cancelled.")
            return

        action = str(result.get("_action") or "").strip().lower()
        if action == "back":
            self.status = "Setup (step 2/3)"
            self._append_activity("Returned to LLM config.")
            self._open_setup_step_two_modal(
                dict(self._pending_setup_payload) if self._pending_setup_payload else {}
            )
            return

        payload: dict[str, Any] = {}
        if isinstance(self._pending_setup_payload, dict):
            payload.update(self._pending_setup_payload)
        payload.update(result)
        del payload["_action"]
        self._pending_setup_payload = None

        self.status = "Applying setup"
        self._append_activity("Applying setup...")
        worker = self.run_worker(
            lambda: self._run_setup_payload(payload),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "setup"

    def _apply_model_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.status = "Applying model configuration"
        self._append_activity("Applying model configuration...")
        worker = self.run_worker(
            lambda: self._run_model_config(
                result.get("provider"),
                result.get("model"),
                result.get("base_url"),
                result.get("temperature"),
                result.get("max_tokens"),
                bool(result.get("validate", True)),
            ),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "model"

    def _apply_settings_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.current_view = str(result.get("view", self.current_view))
        selected_transport = (
            str(result.get("ralph_agent_transport", getattr(self, "ralph_agent_transport", "pipe")))
            .strip()
            .lower()
        )
        self.ralph_agent_transport = (
            selected_transport if selected_transport in {"pipe", "auto", "pty"} else "pipe"
        )
        self.all_cursor_workspaces = bool(
            result.get("all_cursor_workspaces", self.all_cursor_workspaces)
        )
        try:
            from agent_recall.cli.main import _write_tui_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if agent_dir.exists():
                files = FileStorage(agent_dir)
                _write_tui_config(
                    files,
                    {
                        "default_view": self.current_view,
                        "ralph_agent_transport": self.ralph_agent_transport,
                        "all_cursor_workspaces": self.all_cursor_workspaces,
                    },
                )
        except Exception:  # noqa: BLE001
            pass
        self.status = "Settings updated"
        self._append_activity("Settings updated.")
        self._refresh_dashboard_panel()

    def _apply_layout_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        module = self._load_layout_module()
        default_widget_visibility = module.default_widget_visibility
        normalize_banner_size = module.normalize_banner_size

        widgets = result.get("widgets")
        if isinstance(widgets, dict):
            normalized: dict[str, bool] = default_widget_visibility()
            for key, value in widgets.items():
                if isinstance(key, str):
                    normalized[key] = bool(value)
            self.tui_widget_visibility = normalized
        banner_value = result.get("banner_size")
        self.tui_banner_size = normalize_banner_size(banner_value)
        self._apply_tui_layout_settings()

    def _update_terminal_panel_visibility(self: Any, *, initial: bool) -> None:
        terminal_panel = self.query_one("#terminal_panel", Static)
        if self.terminal_supported and self.terminal_panel_visible:
            terminal_panel.display = True
            terminal_panel.update(
                "Embedded terminal panel placeholder. "
                "Run Claude Code/Codex/OpenCode in a separate terminal for now."
            )
            if not initial:
                self.status = "Terminal panel visible"
                self._append_activity("Terminal panel opened.")
        else:
            terminal_panel.display = False
            terminal_panel.update("")
            if not initial:
                self.status = "Terminal panel hidden"
                self._append_activity("Terminal panel hidden.")

    def _load_layout_module(self: Any) -> object:
        if getattr(self, "_layout_module", None) is None:
            self._layout_module = importlib.import_module(
                "agent_recall.cli.tui.ui.modals.layout_customiser"
            )
        return self._layout_module

    def _apply_session_run_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        session_ids = [
            str(item).strip() for item in result.get("session_ids", []) if str(item).strip()
        ]
        if not session_ids:
            self._append_activity("No conversations selected.")
            return
        command_parts = ["sync", "--verbose"]
        for session_id in session_ids:
            command_parts.append("--session-id")
            command_parts.append(shlex.quote(session_id))
        command = " ".join(command_parts)
        self._append_activity(f"Selected {len(session_ids)} conversation(s) for knowledge run.")
        self._run_backend_command(command)

    def _apply_prd_select_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_activity("PRD selection cancelled.")
            return
        selected = result.get("selected_prd_ids")
        if selected is None:
            selected = []
        prds_arg = ",".join(str(x) for x in selected) if selected else ""
        command = f"ralph set-prds --prds {shlex.quote(prds_arg)}"
        self._append_activity(
            f"Selected {len(selected)} PRD item(s)."
            if selected
            else "Cleared PRD selection (all items)."
        )
        self._run_backend_command(command)

    def _apply_ralph_config_modal_result(self: Any, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_activity("Ralph configuration cancelled.")
            return

        action = result.get("action")
        if action:
            action_map = {
                "ralph_install_claude": "ralph hooks install",
                "ralph_uninstall_claude": "ralph hooks uninstall",
                "ralph_install_opencode": "ralph plugin opencode-install",
                "ralph_uninstall_opencode": "ralph plugin opencode-uninstall",
            }
            command = action_map.get(action)
            if command:
                self._run_backend_command(command)
            return

        try:
            from agent_recall.cli.ralph import write_ralph_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if agent_dir.exists():
                files = FileStorage(agent_dir)
                write_ralph_config(files, result)
                self._append_activity("Ralph configuration updated.")
            else:
                self._append_activity("Not initialized. Run 'agent-recall init' first.")
        except Exception as exc:  # noqa: BLE001
            self._append_activity(f"Failed to save Ralph config: {exc}")
