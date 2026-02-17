from __future__ import annotations

import time
from typing import Any

from textual import events
from textual.widgets import Log, OptionList
from textual.widgets.option_list import Option

from agent_recall.cli.tui.constants import _LOADING_FRAMES
from agent_recall.cli.tui.logic.debug_log import _write_debug_log
from agent_recall.cli.tui.logic.text_sanitizers import _sanitize_activity_fragment


class ActivityMixin:
    def _refresh_activity_panel(self: Any) -> None:
        subtitle = f"{self.current_view} · {self.refresh_seconds:.1f}s · Ctrl+P commands"
        title = self.status
        if self._knowledge_run_workers:
            frame_index = int(time.time() * 6) % len(_LOADING_FRAMES)
            frame = _LOADING_FRAMES[frame_index]
            title = f"Knowledge run in progress {frame} Loading"
            subtitle = f"{subtitle} · running synthesis"
        if self._worker_context:
            title = f"{title} …"
        render_key = (title, subtitle)
        if render_key == self._last_activity_render:
            return

        activity_widget = self.query_one("#activity_log", Log)
        was_at_bottom = activity_widget.is_vertical_scroll_end
        focused_widget = self.focused
        focused_widget_id = focused_widget.id if focused_widget is not None else ""

        activity_widget.border_title = title
        activity_widget.border_subtitle = subtitle
        self._last_activity_render = render_key
        if self._worker_context and self._debug_scroll_sample_count < 8:
            self._debug_scroll_sample_count += 1
            # region agent log
            _write_debug_log(
                hypothesis_id="H6",
                location="textual_tui.py:_refresh_activity_panel",
                message="Activity panel refreshed while worker running",
                data={
                    "sample": self._debug_scroll_sample_count,
                    "was_at_bottom": bool(was_at_bottom),
                    "prev_scroll_x": int(activity_widget.scroll_x),
                    "prev_scroll_y": int(activity_widget.scroll_y),
                    "max_scroll_y": int(activity_widget.max_scroll_y),
                    "focused_widget_id": focused_widget_id,
                    "worker_context_size": len(self._worker_context),
                    "line_count": len(self.activity),
                },
            )
            # endregion

        if was_at_bottom:
            self._activity_follow_tail = True

        if self._activity_follow_tail:
            activity_widget.scroll_end(animate=False)

    def _apply_activity_scroll_key(self: Any, key: str) -> None:
        activity_widget = self.query_one("#activity_log", Log)
        if key in {"down", "pagedown", "end"}:
            if key == "end":
                activity_widget.scroll_end(animate=False)
                self._activity_follow_tail = True
                return
            delta = 20 if key == "pagedown" else 3
            activity_widget.scroll_to(
                x=int(activity_widget.scroll_x),
                y=int(activity_widget.scroll_y) + delta,
                animate=False,
                force=True,
            )
            self._activity_follow_tail = activity_widget.is_vertical_scroll_end
            return

        if key in {"up", "pageup", "home"}:
            target_y = (
                0
                if key == "home"
                else int(activity_widget.scroll_y) - (20 if key == "pageup" else 3)
            )
            activity_widget.scroll_to(
                x=int(activity_widget.scroll_x),
                y=target_y,
                animate=False,
                force=True,
            )
            self._activity_follow_tail = False

    def on_key(self: Any, event: events.Key) -> None:
        if event.key not in {"up", "down", "pageup", "pagedown", "home", "end"}:
            return
        try:
            focused_widget = self.focused
        except Exception:  # noqa: BLE001
            focused_widget = None
        focused_widget_id = focused_widget.id if focused_widget is not None else ""
        activity_widget = self.query_one("#activity_log", Log)
        if focused_widget_id == "activity_result_list":
            self._close_inline_result_list(announce=False)
            # region agent log
            _write_debug_log(
                hypothesis_id="H8",
                location="textual_tui.py:on_key",
                message="Auto-closed stale result list on scroll key during active worker",
                data={"key": str(event.key)},
            )
            # endregion

        self._apply_activity_scroll_key(event.key)
        event.prevent_default()
        event.stop()
        # region agent log
        _write_debug_log(
            hypothesis_id="H5",
            location="textual_tui.py:on_key",
            message="Scroll/navigation key observed",
            data={
                "key": str(event.key),
                "focused_widget_id": focused_widget_id,
                "scroll_y": int(activity_widget.scroll_y),
                "max_scroll_y": int(activity_widget.max_scroll_y),
            },
        )
        # endregion

    def on_mouse_scroll_up(self: Any, _event: events.MouseScrollUp) -> None:
        activity_widget = self.query_one("#activity_log", Log)
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
            # region agent log
            _write_debug_log(
                hypothesis_id="H8",
                location="textual_tui.py:on_mouse_scroll_up",
                message="Auto-closed stale result list on mouse scroll up",
                data={},
            )
            # endregion
        activity_widget.scroll_to(
            x=int(activity_widget.scroll_x),
            y=int(activity_widget.scroll_y) - 3,
            animate=False,
            force=True,
        )
        self._activity_follow_tail = False
        # region agent log
        _write_debug_log(
            hypothesis_id="H5",
            location="textual_tui.py:on_mouse_scroll_up",
            message="Mouse scroll up observed",
            data={
                "scroll_y": int(activity_widget.scroll_y),
                "max_scroll_y": int(activity_widget.max_scroll_y),
            },
        )
        # endregion

    def on_mouse_scroll_down(self: Any, _event: events.MouseScrollDown) -> None:
        activity_widget = self.query_one("#activity_log", Log)
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
            # region agent log
            _write_debug_log(
                hypothesis_id="H8",
                location="textual_tui.py:on_mouse_scroll_down",
                message="Auto-closed stale result list on mouse scroll down",
                data={},
            )
            # endregion
        activity_widget.scroll_to(
            x=int(activity_widget.scroll_x),
            y=int(activity_widget.scroll_y) + 3,
            animate=False,
            force=True,
        )
        self._activity_follow_tail = activity_widget.is_vertical_scroll_end
        # region agent log
        _write_debug_log(
            hypothesis_id="H5",
            location="textual_tui.py:on_mouse_scroll_down",
            message="Mouse scroll down observed",
            data={
                "scroll_y": int(activity_widget.scroll_y),
                "max_scroll_y": int(activity_widget.max_scroll_y),
            },
        )
        # endregion

    def _append_activity(self: Any, line: str) -> None:
        clean = _sanitize_activity_fragment(line)
        if not clean:
            return
        self.activity.append(clean)
        payload = clean if clean.endswith(("\n", "\r")) else f"{clean}\n"
        try:
            activity_widget = self.query_one("#activity_log", Log)
            activity_widget.write(payload, scroll_end=self._activity_follow_tail)
        except Exception:  # noqa: BLE001
            pass
        self._refresh_activity_panel()

    def _show_inline_result_list(self: Any, lines: list[str]) -> None:
        if not lines:
            return

        picker = self.query_one("#activity_result_list", OptionList)
        picker.set_options(
            [Option(line, id=f"output:{index}") for index, line in enumerate(lines, start=1)]
        )
        picker.highlighted = 0
        picker.display = True
        self.query_one("#activity_log", Log).display = False
        self._result_list_open = True
        self.status = "Command output list"
        picker.focus()

    def _close_inline_result_list(self: Any, announce: bool = True) -> None:
        if not self._result_list_open:
            return
        picker = self.query_one("#activity_result_list", OptionList)
        picker.display = False
        self.query_one("#activity_log", Log).display = True
        self._result_list_open = False
        if announce:
            self._append_activity("Closed command output list.")
