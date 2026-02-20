from __future__ import annotations

from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option


class SessionsViewModal(ModalScreen[dict[str, object] | None]):
    """An interactive modal replacing the text-based console output for Sessions/Sources."""

    class SourceSyncRequested(Message):
        """Emitted when a source sync is requested."""

        def __init__(self, source_name: str) -> None:
            super().__init__()
            self.source_name = source_name

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def __init__(
        self,
        sessions: list[dict[str, object]],
        sources: list[dict[str, object]] | None = None,
        initial_filter: str = "",
    ):
        super().__init__()
        self.sessions = sessions
        self.sources = sources or []
        self.initial_filter = initial_filter
        self.filter_query = initial_filter
        self._syncing_sources: set[str] = set()

    def compose(self) -> ComposeResult:
        with Container(id="modal_overlay"):
            with Vertical(id="modal_card", classes="modal_wide"):
                yield Static("Sessions & Sources", classes="modal_title")
                yield Static(
                    "Browse all discovered agent conversations across enabled sources",
                    classes="modal_subtitle",
                )

                if self.sources:
                    yield Static("[bold]Sources[/bold]", classes="section_header")
                    yield DataTable(id="sources_table")
                    yield Static(id="sources_status")

                yield Static("[bold]Conversations[/bold]", classes="section_header")
                yield Input(
                    placeholder="Filter by title, source, or session ID...",
                    id="sessions_view_filter",
                    classes="field_input",
                )
                yield OptionList(id="sessions_view_list")
                with Horizontal(classes="modal_actions"):
                    yield Button("Close", id="sessions_view_close")

    def on_mount(self) -> None:
        filter_input = self.query_one("#sessions_view_filter", Input)
        if self.initial_filter:
            filter_input.value = self.initial_filter
        filter_input.focus()
        self._rebuild_options()
        self._build_sources_table()

    def _build_sources_table(self) -> None:
        if not self.sources:
            return
        try:
            table = self.query_one("#sources_table", DataTable)
            table.add_columns("Source", "Status", "Sessions", "Action")

            for source in self.sources:
                name = str(source.get("name", "unknown"))
                available = bool(source.get("available", False))
                sessions = int(source.get("count", 0))  # type: ignore[arg-type]
                error = source.get("error")

                if error:
                    status_text = "[error]✗ Error[/error]"
                elif available:
                    status_text = "[success]✓ Available[/success]"
                else:
                    status_text = "[dim]- No sessions[/dim]"

                button_label = "Syncing..." if name in self._syncing_sources else "Sync"
                button_disabled = name in self._syncing_sources

                sync_button = Button(
                    button_label,
                    id=f"sync_{name}",
                    disabled=button_disabled,
                    classes="sync-button",
                )

                table.add_row(
                    name,
                    status_text,
                    str(sessions),
                    sync_button,
                    key=name,
                )

            status_widget = self.query_one("#sources_status", Static)
            status_widget.update("[dim]Select a source to sync or browse conversations below[/dim]")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sessions_view_close":
            self.dismiss(None)
            return

        button_id = event.button.id or ""
        if button_id.startswith("sync_"):
            source_name = button_id.replace("sync_", "")
            if source_name in self._syncing_sources:
                return
            self._syncing_sources.add(source_name)
            event.button.label = "Syncing..."
            event.button.disabled = True
            self.post_message(self.SourceSyncRequested(source_name))

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            option_list = self.query_one("#sessions_view_list", OptionList)

            if option_list.highlighted is not None:
                index_to_select = option_list.highlighted
            elif option_list.options:
                index_to_select = 0
            else:
                return

            option = option_list.get_option_at_index(index_to_select)
            option_id = str(option.id)
            if option_id.startswith("session:"):
                session_id = option_id.replace("session:", "")
                if session_id != "none":
                    for s in self.sessions:
                        if str(s.get("session_id")) == session_id:
                            self.dismiss(s)
                            return
            return

        if event.key not in {"up", "down"}:
            return
        event.prevent_default()
        event.stop()
        direction = -1 if event.key == "up" else 1
        self._move_highlight(direction)

    def _move_highlight(self, direction: int) -> None:
        option_list = self.query_one("#sessions_view_list", OptionList)
        options = option_list.options
        if not options:
            return

        highlighted = option_list.highlighted
        if highlighted is None:
            index = 0 if direction > 0 else len(options) - 1
        else:
            index = highlighted + direction

        while 0 <= index < len(options):
            if not options[index].disabled:
                option_list.highlighted = index
                break
            index += direction

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "sessions_view_filter":
            return
        self.filter_query = event.value
        self._rebuild_options()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "sessions_view_list":
            option_id = str(event.option.id)
            if option_id.startswith("session:"):
                session_id = option_id.replace("session:", "")
                if session_id == "none":
                    return
                for s in self.sessions:
                    if str(s.get("session_id")) == session_id:
                        self.dismiss(s)
                        return

    def _format_session_label(self, session: dict[str, Any]) -> str:
        """Format a session into a rich Textual string."""
        title = str(session.get("title") or "Untitled conversation")
        source = str(session.get("source") or "")
        session_id = str(session.get("session_id") or "")
        messages = session.get("message_count", 0)
        started = session.get("started", "")
        processed = session.get("processed", False)

        # Pad title for column-like alignment
        padded_title = f"{title:<50}"[:50]

        source_str = f"{source:<10}"[:10]
        if source:
            source_str = source_str.replace(source, f"[accent]{source}[/accent]")

        status_word = "synced" if processed else "not synced"
        status_str = f"{status_word:<12}"
        if processed:
            status_str = status_str.replace(status_word, f"[success]{status_word}[/success]")
        else:
            status_str = status_str.replace(status_word, f"[warning]{status_word}[/warning]")

        if source and session_id.startswith(f"{source}-"):
            session_id = session_id[len(source) + 1 :]
        session_id_display = session_id[-8:]

        first_line = f"{session_id_display}    {padded_title} {messages} msgs"
        second_line = f"[dim]  └─ {source_str} {started}         {status_str}[/dim]"

        return f"{first_line}\n{second_line}"

    def _rebuild_options(self) -> None:
        query = self.filter_query.strip().lower()
        visible_items: list[dict[str, object]] = []

        for session in self.sessions:
            title = str(session.get("title") or "").lower()
            source = str(session.get("source") or "").lower()
            session_id = str(session.get("session_id") or "").lower()

            if query and query not in f"{title} {source} {session_id}":
                continue
            visible_items.append(session)

        option_list = self.query_one("#sessions_view_list", OptionList)
        options: list[Option] = []

        if not visible_items:
            options = [Option("No matching conversations found.", id="session:none", disabled=True)]
        else:
            for session in visible_items:
                session_id = str(session.get("session_id") or "")
                options.append(
                    Option(self._format_session_label(session), id=f"session:{session_id}")
                )

        option_list.set_options(options)
