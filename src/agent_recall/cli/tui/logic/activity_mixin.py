from __future__ import annotations

import re
import time
from typing import Any

from rich.text import Text
from textual import events
from textual.widgets import Input, Log, OptionList
from textual.widgets.option_list import Option

from agent_recall.cli.tui.constants import _LOADING_FRAMES
from agent_recall.cli.tui.logic.debug_log import _write_debug_log
from agent_recall.cli.tui.logic.text_sanitizers import (
    _activity_line_theme_style,
    _sanitize_activity_fragment,
)


class ActivityMixin:
    def _refresh_activity_panel(self: Any) -> None:
        subtitle = f"{self.current_view} · event-driven · Ctrl+P commands"
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

    def action_activity_search(self: Any) -> None:
        if self._activity_search_active:
            return
        self._open_activity_search()

    def _open_activity_search(self: Any) -> None:
        self._activity_search_active = True
        self._activity_search_query = ""
        self._activity_search_matches = []
        self._activity_search_current_index = -1
        try:
            activity_log = self.query_one("#activity_log", Log)
            self._activity_search_saved_scroll_y = int(activity_log.scroll_y)
        except Exception:
            self._activity_search_saved_scroll_y = 0
        try:
            search_input = self.query_one("#activity_search_input", Input)
            search_input.display = True
            search_input.value = ""
            search_input.focus()
        except Exception:
            pass
        self._refresh_activity_panel()

    def _close_activity_search(self: Any) -> None:
        if not self._activity_search_active:
            return
        self._activity_search_active = False
        self._activity_search_query = ""
        self._activity_search_matches = []
        self._activity_search_current_index = -1
        try:
            search_input = self.query_one("#activity_search_input", Input)
            search_input.display = False
            search_input.value = ""
        except Exception:
            pass
        self._restore_activity_log()
        try:
            activity_log = self.query_one("#activity_log", Log)
            activity_log.scroll_to(
                x=0, y=self._activity_search_saved_scroll_y, animate=False, force=True
            )
        except Exception:
            pass
        self._refresh_activity_panel()

    def _restore_activity_log(self: Any) -> None:
        try:
            activity_log = self.query_one("#activity_log", Log)
            activity_log.clear()
            for line in self.activity:
                payload = line if line.endswith(("\n", "\r")) else f"{line}\n"
                for fragment in payload.splitlines(keepends=True):
                    style_name = _activity_line_theme_style(fragment)
                    if style_name:
                        activity_log.write(
                            Text(fragment, style=style_name),
                            scroll_end=False,
                        )
                    else:
                        activity_log.write(fragment, scroll_end=False)
        except Exception:
            pass

    def _perform_activity_search(self: Any, query: str) -> None:
        self._activity_search_query = query
        if not query:
            self._activity_search_matches = []
            self._activity_search_current_index = -1
            self._restore_activity_log()
            self._update_search_input_placeholder()
            return
        query_lower = query.lower()
        self._activity_search_matches = []
        for i, line in enumerate(self.activity):
            if query_lower in line.lower():
                self._activity_search_matches.append(i)
        self._activity_search_current_index = 0 if self._activity_search_matches else -1
        self._render_search_results()
        self._update_search_input_placeholder()

    def _render_search_results(self: Any) -> None:
        if not self._activity_search_query:
            self._restore_activity_log()
            return
        try:
            activity_log = self.query_one("#activity_log", Log)
            activity_log.clear()
            query_lower = self._activity_search_query.lower()
            for line in self.activity:
                payload = line if line.endswith(("\n", "\r")) else f"{line}\n"
                for fragment in payload.splitlines(keepends=True):
                    style_name = _activity_line_theme_style(fragment)
                    if query_lower in fragment.lower():
                        highlighted = self._highlight_match(fragment, self._activity_search_query)
                        if style_name:
                            activity_log.write(
                                Text.assemble(Text(highlighted, style=style_name)),
                                scroll_end=False,
                            )
                        else:
                            activity_log.write(highlighted, scroll_end=False)
                    else:
                        if style_name:
                            activity_log.write(
                                Text(fragment, style=style_name),
                                scroll_end=False,
                            )
                        else:
                            activity_log.write(fragment, scroll_end=False)
            if self._activity_search_matches and self._activity_search_current_index >= 0:
                self._scroll_to_match(self._activity_search_current_index)
        except Exception:
            pass

    def _highlight_match(self: Any, text: str, query: str) -> str:
        if not query:
            return text
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        def replace_match(m: re.Match[str]) -> str:
            return f"[bold reverse]{m.group(0)}[/bold reverse]"

        return pattern.sub(replace_match, text)

    def _scroll_to_match(self: Any, match_index: int) -> None:
        if not self._activity_search_matches or match_index < 0:
            return
        if match_index >= len(self._activity_search_matches):
            return
        try:
            activity_log = self.query_one("#activity_log", Log)
            line_index = self._activity_search_matches[match_index]
            activity_log.scroll_to(x=0, y=line_index, animate=False, force=True)
        except Exception:
            pass

    def _cycle_search_match_forward(self: Any) -> None:
        if not self._activity_search_matches:
            return
        if len(self._activity_search_matches) == 0:
            return
        self._activity_search_current_index = (self._activity_search_current_index + 1) % len(
            self._activity_search_matches
        )
        self._scroll_to_match(self._activity_search_current_index)
        self._update_search_input_placeholder()

    def _cycle_search_match_backward(self: Any) -> None:
        if not self._activity_search_matches:
            return
        if len(self._activity_search_matches) == 0:
            return
        self._activity_search_current_index = (self._activity_search_current_index - 1) % len(
            self._activity_search_matches
        )
        self._scroll_to_match(self._activity_search_current_index)
        self._update_search_input_placeholder()

    def _update_search_input_placeholder(self: Any) -> None:
        try:
            search_input = self.query_one("#activity_search_input", Input)
            match_count = len(self._activity_search_matches)
            if match_count == 0:
                if self._activity_search_query:
                    search_input.placeholder = "No matches"
                else:
                    search_input.placeholder = "Search activity log..."
            else:
                current = self._activity_search_current_index + 1
                search_input.placeholder = f"{current}/{match_count} matches"
        except Exception:
            pass

    def on_input_changed_activity_search_input(self: Any, event: Input.Changed) -> None:
        if not self._activity_search_active:
            return
        self._perform_activity_search(event.value)

    def on_key_activity_search_input(self: Any, event: events.Key) -> None:
        if not self._activity_search_active:
            return
        if event.key == "escape":
            self._close_activity_search()
            event.prevent_default()
            event.stop()
        elif event.key in ("enter", "n"):
            self._cycle_search_match_forward()
            event.prevent_default()
            event.stop()
        elif event.key == "N":
            self._cycle_search_match_backward()
            event.prevent_default()
            event.stop()

    def close_activity_search_if_open(self: Any) -> None:
        if self._activity_search_active:
            self._close_activity_search()

    def on_key(self: Any, event: events.Key) -> None:
        if self._activity_search_active:
            return
        if event.key not in {"up", "down", "pageup", "pagedown", "home", "end"}:
            return
        try:
            focused_widget = self.focused
        except Exception:  # noqa: BLE001
            focused_widget = None
        focused_widget_id = focused_widget.id if focused_widget is not None else ""
        if focused_widget_id == "dashboard_timeline_interactive":
            return
        activity_widget = self.query_one("#activity_log", Log)
        if focused_widget_id == "activity_result_list":
            self._close_inline_result_list(announce=False)
            _write_debug_log(
                hypothesis_id="H8",
                location="textual_tui.py:on_key",
                message="Auto-closed stale result list on scroll key during active worker",
                data={"key": str(event.key)},
            )

        self._apply_activity_scroll_key(event.key)
        event.prevent_default()
        event.stop()
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
            for fragment in payload.splitlines(keepends=True):
                style_name = _activity_line_theme_style(fragment)
                if style_name:
                    activity_widget.write(
                        Text(fragment, style=style_name),
                        scroll_end=self._activity_follow_tail,
                    )
                else:
                    activity_widget.write(fragment, scroll_end=self._activity_follow_tail)
        except Exception:  # noqa: BLE001
            pass
        self._refresh_activity_panel()

    def _show_inline_result_list(self: Any, lines: list[str], *, focus: bool = True) -> None:
        if not lines:
            return
        options = [Option(line, id=f"output:{index}") for index, line in enumerate(lines, start=1)]
        self._set_activity_result_options(options)
        self.status = "Command output list"
        if focus:
            self.query_one("#activity_result_list", OptionList).focus()

    def _set_activity_result_options(self: Any, options: list[Option]) -> None:
        picker = self.query_one("#activity_result_list", OptionList)
        picker.set_options(options)
        for index, option in enumerate(options):
            if not option.disabled:
                picker.highlighted = index
                break
        picker.display = True
        self.query_one("#activity_log", Log).display = False
        self._result_list_open = True

    def _close_inline_result_list(self: Any, announce: bool = True) -> None:
        if not self._result_list_open:
            return
        picker = self.query_one("#activity_result_list", OptionList)
        picker.display = False
        self.query_one("#activity_log", Log).display = True
        self._result_list_open = False
        if announce:
            self._append_activity("Closed command output list.")
