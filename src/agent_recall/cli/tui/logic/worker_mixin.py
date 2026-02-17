from __future__ import annotations

from typing import Any, cast

from textual.worker import Worker, WorkerState

from agent_recall.cli.tui.logic.text_sanitizers import _strip_rich_markup
from agent_recall.cli.tui.ui.modals.diff_viewer import DiffViewerModal
from agent_recall.cli.tui.ui.modals.prd_select import PRDSelectModal
from agent_recall.cli.tui.ui.modals.session_run import SessionRunModal


class WorkerMixin:
    def on_worker_state_changed(self: Any, event: Worker.StateChanged) -> None:
        worker_key = id(event.worker)
        context = self._worker_context.get(worker_key)
        if context is None:
            return

        if event.state == WorkerState.RUNNING:
            return

        if event.state == WorkerState.SUCCESS:
            self._handle_worker_success(context, event.worker.result)
            self._worker_context.pop(worker_key, None)
            self._complete_knowledge_worker(worker_key, success=True)
            return

        if event.state == WorkerState.CANCELLED:
            self.status = "Operation cancelled"
            self._append_activity("Previous operation cancelled.")
            self._finalize_theme_commit(success=False)
            self._worker_context.pop(worker_key, None)
            self._complete_knowledge_worker(worker_key, success=False)
            self._refresh_dashboard_panel()
            return

        if event.state == WorkerState.ERROR:
            self.status = "Operation failed"
            error = event.worker.error
            if error is None:
                self._append_activity("Operation failed with an unknown error.")
            else:
                self._append_activity(f"Error: {error}")
            self._finalize_theme_commit(success=False)
            self._worker_context.pop(worker_key, None)
            self._complete_knowledge_worker(worker_key, success=False)
            self._refresh_dashboard_panel()

    def _handle_worker_success(self: Any, context: str, result: object) -> None:
        if context == "command":
            should_exit = False
            lines: list[str] = []
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], bool)
                and isinstance(result[1], list)
            ):
                should_exit = result[0]
                lines = [line for line in result[1] if isinstance(line, str)]

            cleaned_lines: list[str] = []
            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    cleaned_lines.append(cleaned)
                    self._append_activity(cleaned)

            command_header = cleaned_lines[0] if cleaned_lines else ""
            if cleaned_lines and (
                len(cleaned_lines) > 12
                or "/sources" in command_header
                or "/sessions" in command_header
            ):
                self._show_inline_result_list(cleaned_lines)

            self.status = "Last command executed"
            self._finalize_theme_commit(success=True)
            self._refresh_dashboard_panel()
            if should_exit:
                self.action_request_quit()
            return

        if context.startswith("sync-source:"):
            source_name = context.split(":", 1)[1]
            lines = (
                [line for line in result if isinstance(line, str)]
                if isinstance(result, list)
                else []
            )
            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)
            if source_name:
                self.status = f"Source sync complete: {source_name}"
                self._append_activity(f"Source sync complete: {source_name}.")
            else:
                self.status = "Source sync complete"
                self._append_activity("Source sync complete.")
            self._refresh_dashboard_panel()
            return

        if context == "session_picker":
            sessions: list[dict[str, Any]] = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        sessions.append({str(key): value for key, value in item.items()})
            if not sessions:
                self.status = "No conversations found"
                self._append_activity("No conversations available to select.")
                self._refresh_dashboard_panel()
                return
            self.status = "Select conversations"
            self._append_activity(f"Loaded {len(sessions)} conversation(s).")
            self.push_screen(
                SessionRunModal(sessions),
                self._apply_session_run_modal_result,
            )
            self._refresh_dashboard_panel()
            return

        if context == "prd_picker":
            if not isinstance(result, dict):
                self.status = "Failed to load PRD items"
                self._append_activity("Could not load PRD items.")
                self._refresh_dashboard_panel()
                return
            prd_data = cast(dict[str, Any], result)
            items = prd_data.get("items") or []
            selected_ids = prd_data.get("selected_ids") or []
            max_iterations = int(prd_data.get("max_iterations") or 10)
            if not items:
                self.status = "No PRD items found"
                self._append_activity("No PRD items available to select.")
                self._refresh_dashboard_panel()
                return
            self.status = "Select PRD items"
            self._append_activity(f"Loaded {len(items)} PRD item(s).")
            self.push_screen(
                PRDSelectModal(items, selected_ids, max_iterations),
                self._apply_prd_select_modal_result,
            )
            self._refresh_dashboard_panel()
            return

        if context == "diff_viewer":
            diff_text: str | None = None
            iteration = None
            if isinstance(result, tuple) and len(result) == 2:
                diff_text = result[0] if isinstance(result[0], str) else None
                iteration = result[1] if isinstance(result[1], int) else None
            if not diff_text:
                self.status = "No diff available"
                self._append_activity("No iteration diff found to display.")
                self._refresh_dashboard_panel()
                return
            title = "Iteration Diff" if iteration is None else f"Iteration {iteration:03d} Diff"
            self.status = "Viewing diff"
            self._append_activity("Opened iteration diff viewer.")
            self.push_screen(
                DiffViewerModal(diff_text=diff_text, title=title),
            )
            self._refresh_dashboard_panel()
            return

        if context == "setup":
            changed = False
            lines: list[str] = []
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], bool)
                and isinstance(result[1], list)
            ):
                changed = result[0]
                lines = [line for line in result[1] if isinstance(line, str)]

            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)

            self.onboarding_required = False
            self.status = "Setup completed" if changed else "Setup unchanged"
            self._append_activity(
                "Setup completed." if changed else "Setup already complete for this repository."
            )
            self._refresh_dashboard_panel()
            return

        if context == "model":
            lines = (
                [line for line in result if isinstance(line, str)]
                if isinstance(result, list)
                else []
            )
            for line in lines:
                cleaned = _strip_rich_markup(line).strip()
                if cleaned:
                    self._append_activity(cleaned)

            self.status = "Model configuration updated"
            self._append_activity("Model configuration updated.")
            self._refresh_dashboard_panel()
            return

        if context == "ralph_run":
            rd = dict(result) if isinstance(result, dict) else {}
            if "exit_code" in rd:
                exit_code = int(rd.get("exit_code") or 0)
                if exit_code == 0:
                    self._append_activity("Ralph loop completed successfully.")
                    self._clear_selected_prd_ids_after_successful_run()
                elif exit_code == 2:
                    self._append_activity("Ralph loop reached max iterations.")
                else:
                    self._append_activity(f"Ralph loop failed (exit {exit_code}).")
                self.status = "Ralph loop complete" if exit_code == 0 else "Ralph loop failed"
                self._refresh_dashboard_panel()
                return
            total = int(rd.get("total_iterations", 0))
            passed = int(rd.get("passed", 0))
            failed = int(rd.get("failed", 0))
            self._append_activity(
                f"Ralph run complete — {total} iteration(s), {passed} passed, {failed} failed."
            )
            self.status = "Ralph loop complete"
            self._refresh_dashboard_panel()

    def _complete_knowledge_worker(self: Any, worker_key: int, *, success: bool) -> None:
        if worker_key not in self._knowledge_run_workers:
            return
        self._knowledge_run_workers.discard(worker_key)
        if self._knowledge_run_workers:
            return
        if success:
            self._append_activity("Knowledge run completed.")
        else:
            self._append_activity("Knowledge run ended with errors.")
