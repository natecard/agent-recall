from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from textual.widgets import Log, OptionList

from agent_recall.cli.stream_pipeline import (
    run_streaming_command,
    stream_debug_dir,
    stream_debug_enabled,
)
from agent_recall.cli.tui.constants import (
    _RALPH_STREAM_FLUSH_MAX_CHARS,
    _RALPH_STREAM_FLUSH_SECONDS,
)
from agent_recall.cli.tui.logic.debug_log import _write_debug_log
from agent_recall.cli.tui.logic.notifications import _read_notification_config
from agent_recall.ralph.notifications import dispatch_notification
from agent_recall.storage.models import RalphNotificationEvent


class RalphMixin:
    def action_run_ralph_loop(self: Any) -> None:
        """Run the Ralph loop with live output streaming to the activity console."""
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
            # region agent log
            _write_debug_log(
                hypothesis_id="H8",
                location="textual_tui.py:action_run_ralph_loop",
                message="Closed inline result list before starting Ralph loop",
                data={},
            )
            # endregion
        self.query_one("#activity_log", Log).display = True
        self.query_one("#activity_result_list", OptionList).display = False
        self.status = "Ralph loop starting"
        self._append_activity("Starting Ralph loop...")

        def _ralph_run_worker() -> dict[str, Any]:
            from agent_recall.cli.ralph import (
                build_agent_cmd_from_ralph_config,
                get_default_prd_path,
                get_default_script_path,
                read_ralph_config,
            )
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if not agent_dir.exists():
                self.call_from_thread(
                    self._append_activity, "Not initialized. Run 'agent-recall init' first."
                )
                return {"total_iterations": 0, "passed": 0, "failed": 0}

            files = FileStorage(agent_dir)
            ralph_cfg = read_ralph_config(files)
            notify_enabled, notify_events = _read_notification_config(ralph_cfg)

            enabled_value = ralph_cfg.get("enabled")
            if not (isinstance(enabled_value, bool) and enabled_value):
                self.call_from_thread(
                    self._append_activity, "Ralph loop is disabled. Enable it first."
                )
                return {"total_iterations": 0, "passed": 0, "failed": 0}

            agent_cmd = build_agent_cmd_from_ralph_config(ralph_cfg)
            if not agent_cmd:
                self.call_from_thread(
                    self._append_activity,
                    "Ralph coding CLI is not configured. Set it in Ralph Configuration first.",
                )
                return {"total_iterations": 0, "passed": 0, "failed": 0}

            max_iter_value = ralph_cfg.get("max_iterations")
            max_iterations = int(max_iter_value) if isinstance(max_iter_value, int | float) else 10
            sleep_value = ralph_cfg.get("sleep_seconds")
            sleep_seconds = int(sleep_value) if isinstance(sleep_value, int | float) else 2
            compact_value = ralph_cfg.get("compact_mode")
            compact_mode = (
                str(compact_value).strip().lower()
                if isinstance(compact_value, str) and str(compact_value).strip()
                else "always"
            )
            if compact_mode not in {"always", "on-failure", "off"}:
                compact_mode = "always"
            selected_transport = str(getattr(self, "ralph_agent_transport", "pipe")).strip().lower()
            if selected_transport not in {"pipe", "auto", "pty"}:
                selected_transport = "pipe"

            prd_path = get_default_prd_path()
            selected_value = ralph_cfg.get("selected_prd_ids")
            selected_ids: list[str] | None = None
            if isinstance(selected_value, list) and selected_value:
                selected_ids = [str(x) for x in selected_value if x]

            self.call_from_thread(setattr, self, "status", "Ralph loop running")
            self.call_from_thread(self._append_activity, f"Agent command: {agent_cmd}")

            script_path = get_default_script_path()
            if not script_path.exists():
                self.call_from_thread(
                    self._append_activity,
                    (
                        f"Ralph loop script not found: {script_path}. "
                        "Falling back to built-in loop mode."
                    ),
                )
                from agent_recall.cli.ralph import get_ralph_components
                from agent_recall.ralph.loop import RalphLoop

                coding_cli_value = ralph_cfg.get("coding_cli")
                coding_cli = (
                    str(coding_cli_value).strip()
                    if isinstance(coding_cli_value, str) and str(coding_cli_value).strip()
                    else ""
                )
                if not coding_cli:
                    self.call_from_thread(
                        self._append_activity,
                        "Ralph coding CLI is not configured. Set it in Ralph Configuration first.",
                    )
                    return {"total_iterations": 0, "passed": 0, "failed": 0}

                cli_model_value = ralph_cfg.get("cli_model")
                cli_model = (
                    str(cli_model_value).strip()
                    if isinstance(cli_model_value, str) and str(cli_model_value).strip()
                    else None
                )

                def _on_python_loop_event(event: dict[str, Any]) -> None:
                    event_type = str(event.get("event") or "")
                    if event_type == "output_line":
                        line = str(event.get("line") or "")
                        if line:
                            self.call_from_thread(self._append_activity, line)
                        return
                    if event_type == "iteration_started":
                        iteration = event.get("iteration")
                        item_id = event.get("item_id")
                        self.call_from_thread(
                            self._append_activity,
                            f"Iteration {iteration}: {item_id} started",
                        )
                        return
                    if event_type == "validation_complete" and not bool(event.get("success")):
                        hint = str(event.get("hint") or "Validation failed")
                        self.call_from_thread(self._append_activity, hint)
                        return
                    if event_type == "iteration_complete":
                        iteration = event.get("iteration")
                        outcome = event.get("outcome")
                        self.call_from_thread(
                            self._append_activity,
                            f"Iteration {iteration} complete ({outcome})",
                        )
                        return
                    if event_type == "budget_exceeded":
                        self.call_from_thread(
                            self._append_activity,
                            "Ralph loop stopped: configured cost budget exceeded.",
                        )

                try:
                    agent_dir_runtime, storage, loop_files = get_ralph_components()
                    loop = RalphLoop(agent_dir_runtime, storage, loop_files)
                    summary = asyncio.run(
                        loop.run_loop(
                            max_iterations=max_iterations,
                            selected_prd_ids=selected_ids,
                            progress_callback=_on_python_loop_event,
                            coding_cli=coding_cli,
                            cli_model=cli_model,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.call_from_thread(self._append_activity, f"Ralph loop failed: {exc}")
                    return {"total_iterations": 0, "passed": 0, "failed": 0}
                if notify_enabled:
                    dispatch_notification(
                        RalphNotificationEvent.LOOP_FINISHED,
                        enabled=notify_enabled,
                        enabled_events=notify_events,
                    )
                failed = int(summary.get("failed", 0))
                summary["exit_code"] = 0 if failed == 0 else 1
                return summary

            cmd = [
                str(script_path),
                "--agent-cmd",
                agent_cmd,
                "--max-iterations",
                str(max_iterations),
                "--compact-mode",
                compact_mode,
                "--agent-transport",
                selected_transport,
                "--sleep-seconds",
                str(sleep_seconds),
            ]
            if prd_path.exists():
                cmd.extend(["--prd-file", str(prd_path)])
            if selected_ids:
                selected_arg = ",".join(selected_ids)
                if selected_arg:
                    cmd.extend(["--prd-ids", selected_arg])

            pending_fragments: list[str] = []
            pending_chars = 0
            last_flush_monotonic = time.monotonic()

            def _flush_pending_fragments() -> None:
                nonlocal pending_fragments, pending_chars, last_flush_monotonic
                if not pending_fragments:
                    return
                combined = "".join(pending_fragments)
                pending_fragments = []
                pending_chars = 0
                self.call_from_thread(self._append_activity, combined)
                last_flush_monotonic = time.monotonic()

            def _on_stream_fragment(fragment: str) -> None:
                nonlocal pending_chars
                pending_fragments.append(fragment)
                pending_chars += len(fragment)
                now = time.monotonic()
                if (
                    pending_chars >= _RALPH_STREAM_FLUSH_MAX_CHARS
                    or (now - last_flush_monotonic) >= _RALPH_STREAM_FLUSH_SECONDS
                ):
                    _flush_pending_fragments()

            try:
                exit_code = run_streaming_command(
                    cmd,
                    cwd=Path.cwd(),
                    on_emit=_on_stream_fragment,
                    context="tui_ralph_run",
                    partial_flush_ms=120,
                    transport="pipe",
                )
                _flush_pending_fragments()
            except OSError as exc:
                self.call_from_thread(self._append_activity, f"Ralph loop failed: {exc}")
                return {"total_iterations": 0, "passed": 0, "failed": 0}
            if stream_debug_enabled():
                self.call_from_thread(
                    self._append_activity,
                    f"Stream debug artifacts: {stream_debug_dir(Path.cwd())}",
                )
            if notify_enabled:
                dispatch_notification(
                    RalphNotificationEvent.LOOP_FINISHED,
                    enabled=notify_enabled,
                    enabled_events=notify_events,
                )
            return {"total_iterations": 0, "passed": 0, "failed": 0, "exit_code": exit_code}

        worker = self.run_worker(
            _ralph_run_worker,
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = "ralph_run"

    def _clear_selected_prd_ids_after_successful_run(self: Any) -> None:
        try:
            from agent_recall.cli.ralph import read_ralph_config, write_ralph_config
            from agent_recall.storage.files import FileStorage

            agent_dir = Path(".agent")
            if not agent_dir.exists():
                return

            files = FileStorage(agent_dir)
            ralph_cfg = read_ralph_config(files)
            selected_value = ralph_cfg.get("selected_prd_ids")
            if not isinstance(selected_value, list):
                return

            selected_ids = [str(item).strip() for item in selected_value if str(item).strip()]
            if not selected_ids:
                return

            write_ralph_config(files, {"selected_prd_ids": None})
            self._append_activity("Cleared PRD selection after successful Ralph run.")
        except Exception as exc:  # noqa: BLE001
            self._append_activity(f"Failed to clear PRD selection: {exc}")
