from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


class SessionRunModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
        Binding("space", "toggle_selection", "Toggle"),
        Binding("enter", "apply", "Run"),
    ]

    def __init__(self, sessions: list[dict[str, Any]]):
        super().__init__()
        self.sessions = sessions
        self.filter_query = ""
        self.selected_session_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card"):
                yield Static("Run Knowledge Update", classes="modal_title")
                yield Input(
                    placeholder="Filter conversations...",
                    id="run_sessions_filter",
                    classes="field_input",
                )
                yield OptionList(id="run_sessions_list")
                yield Static("", id="run_sessions_status")
                with Horizontal(classes="modal_actions"):
                    yield Button("Run Selected", variant="primary", id="run_sessions_apply")
                    yield Button("Cancel", id="run_sessions_cancel")

    def on_mount(self) -> None:
        self.query_one("#run_sessions_filter", Input).focus()
        self._rebuild_options()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "run_sessions_filter":
            return
        self.filter_query = event.value
        self._rebuild_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "run_sessions_list":
            return
        self._toggle_highlighted()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_sessions_cancel":
            self.dismiss(None)
            return
        if event.button.id == "run_sessions_apply":
            self._submit()

    def action_toggle_selection(self) -> None:
        self._toggle_highlighted()

    def action_apply(self) -> None:
        self._submit()

    def _toggle_highlighted(self) -> None:
        option_list = self.query_one("#run_sessions_list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            return
        option = option_list.get_option_at_index(highlighted)
        option_id = option.id or ""
        if not option_id.startswith("session:"):
            return
        session_id = option_id.split(":", 1)[1]
        if not session_id:
            return
        if session_id in self.selected_session_ids:
            self.selected_session_ids.remove(session_id)
        else:
            self.selected_session_ids.add(session_id)
        self._rebuild_options()

    def _submit(self) -> None:
        ordered_selection = [
            str(session.get("session_id") or "")
            for session in self.sessions
            if str(session.get("session_id") or "") in self.selected_session_ids
        ]
        ordered_selection = [item for item in ordered_selection if item]
        if not ordered_selection:
            self.query_one("#run_sessions_status", Static).update(
                "[red]Select at least one conversation.[/red]"
            )
            return
        self.dismiss({"session_ids": ordered_selection})

    def _rebuild_options(self) -> None:
        query = self.filter_query.strip().lower()
        visible_sessions: list[dict[str, Any]] = []
        for session in self.sessions:
            title = str(session.get("title") or "").lower()
            started = str(session.get("started") or "").lower()
            source = str(session.get("source") or "").lower()
            if query and query not in f"{title} {started} {source}":
                continue
            visible_sessions.append(session)

        option_list = self.query_one("#run_sessions_list", OptionList)
        options: list[Option] = []
        for session in visible_sessions:
            session_id = str(session.get("session_id") or "")
            if not session_id:
                continue
            selected = session_id in self.selected_session_ids
            options.append(
                Option(
                    self._line_for_session(session, selected),
                    id=f"session:{session_id}",
                )
            )

        if not options:
            options = [Option("No matching conversations.", id="session:", disabled=True)]

        option_list.set_options(options)
        for index, option in enumerate(options):
            if option.disabled:
                continue
            option_list.highlighted = index
            break

        self.query_one("#run_sessions_status", Static).update(
            f"[dim]{len(visible_sessions)} shown · {len(self.selected_session_ids)} selected"
            " · Space toggles, Enter runs[/dim]"
        )

    def _line_for_session(self, session: dict[str, Any], selected: bool) -> str:
        marker = "[green]✓[/green]" if selected else "[dim]○[/dim]"
        title = str(session.get("title") or "Untitled conversation")
        started = str(session.get("started") or "-")
        message_count = int(session.get("message_count", 0))
        processed = "processed" if bool(session.get("processed")) else "new"
        return f"{marker} {title} [dim]({started} · {message_count} msg · {processed})[/dim]"
