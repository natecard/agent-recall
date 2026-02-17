from __future__ import annotations

import shlex
from typing import Any

from rich.theme import Theme

from agent_recall.cli.tui.ui.modals.theme_picker import ThemePickerModal


class ThemeMixin:
    def _sync_runtime_theme(self: Any) -> None:
        if self._theme_preview_active:
            return
        if self._theme_runtime_provider is None:
            return
        try:
            theme_name, theme = self._theme_runtime_provider()
        except Exception:  # noqa: BLE001
            return
        self._apply_theme(theme_name, theme)

    def _apply_theme(self: Any, theme_name: str, theme: Theme) -> None:
        if self._active_theme_name == theme_name:
            return
        if self._active_theme_name is not None:
            self.console.pop_theme()
        self.console.push_theme(theme)
        self._active_theme_name = theme_name

    def action_open_theme_modal(self: Any) -> None:
        themes, current_theme = self._theme_defaults_provider()
        if not themes:
            self.status = "No themes available"
            self._append_activity("No themes available.")
            return

        if self._result_list_open:
            self._close_inline_result_list(announce=False)

        runtime_theme_name = current_theme
        if self._theme_runtime_provider is not None:
            try:
                runtime_theme_name, _ = self._theme_runtime_provider()
            except Exception:  # noqa: BLE001
                runtime_theme_name = current_theme
        selected_theme = runtime_theme_name if runtime_theme_name in themes else current_theme

        self._theme_preview_origin = runtime_theme_name
        self._theme_preview_active = True
        self._theme_commit_inflight = False
        self.status = "Theme picker"
        self._append_activity("Theme picker opened. Preview follows selection.")
        self.push_screen(
            ThemePickerModal(
                themes,
                selected_theme,
                self._preview_theme_from_modal,
            ),
            self._apply_theme_modal_result,
        )

    def _resolve_theme_by_name(self: Any, theme_name: str) -> Theme | None:
        if not theme_name.strip():
            return None
        if self._theme_resolve_provider is None:
            return None
        try:
            return self._theme_resolve_provider(theme_name)
        except Exception:  # noqa: BLE001
            return None

    def _preview_theme_from_modal(self: Any, theme_name: str) -> None:
        theme = self._resolve_theme_by_name(theme_name)
        if theme is None:
            return
        self._apply_theme(theme_name, theme)
        self._refresh_dashboard_panel()

    def _restore_preview_origin(self: Any) -> None:
        origin = self._theme_preview_origin
        if not origin:
            return
        theme = self._resolve_theme_by_name(origin)
        if theme is None:
            return
        self._apply_theme(origin, theme)
        self._refresh_dashboard_panel()

    def _apply_theme_modal_result(self: Any, result: dict[str, str] | None) -> None:
        if result is None:
            self._theme_preview_active = False
            self._theme_commit_inflight = False
            self._restore_preview_origin()
            self._theme_preview_origin = None
            self.status = "Theme unchanged"
            self._append_activity("Theme picker closed without changes.")
            return

        selected_theme = str(result.get("theme", "")).strip()
        if not selected_theme:
            self._theme_preview_active = False
            self._theme_commit_inflight = False
            self._restore_preview_origin()
            self._theme_preview_origin = None
            self.status = "Theme unchanged"
            return

        self.status = f"Applying theme: {selected_theme}"
        self._append_activity(f"Applying theme '{selected_theme}'...")
        self._theme_commit_inflight = True
        self._run_backend_command(
            f"theme set {shlex.quote(selected_theme)}",
            bypass_local=True,
        )

    def _finalize_theme_commit(self: Any, *, success: bool) -> None:
        if not self._theme_commit_inflight and not self._theme_preview_active:
            return
        if success:
            self._theme_commit_inflight = False
            self._theme_preview_active = False
            self._theme_preview_origin = None
            return
        self._theme_commit_inflight = False
        self._theme_preview_active = False
        self._restore_preview_origin()
        self._theme_preview_origin = None
