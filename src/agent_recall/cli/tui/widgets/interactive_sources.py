from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Static

from agent_recall.ingest.health import HealthStatus


class InteractiveSourcesWidget(Vertical):
    """Interactive sources widget with DataTable, sync buttons, and health indicators."""

    class SourceSelected(Message):
        """Emitted when a source row is selected."""

        def __init__(self, source_name: str) -> None:
            super().__init__()
            self.source_name = source_name

    class HealthCheckRequested(Message):
        """Emitted when Ctrl+R is pressed to re-probe all sources."""

        def __init__(self) -> None:
            super().__init__()

    DEFAULT_CSS = """
    InteractiveSourcesWidget {
        height: auto;
        min-height: 10;
        padding: 0;
    }
    InteractiveSourcesWidget #sources_table {
        height: auto;
        min-height: 6;
        border: none;
    }
    InteractiveSourcesWidget #sources_status {
        height: auto;
        margin-top: 1;
        text-align: center;
    }
    InteractiveSourcesWidget .sync-row {
        height: auto;
        margin: 0;
        padding: 0;
    }
    InteractiveSourcesWidget .source-name {
        width: 20;
        content-align: left middle;
    }
    InteractiveSourcesWidget .source-status {
        width: 20;
        content-align: left middle;
    }
    InteractiveSourcesWidget .source-sessions {
        width: 12;
        content-align: right middle;
    }
    InteractiveSourcesWidget .sync-button {
        width: 10;
        min-width: 8;
    }
    """

    def __init__(
        self,
        sources: list[dict[str, Any]],
        on_sync: Callable[[str], None],
        last_synced: str = "Never",
        on_health_check: Callable[[], None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.sources = sources
        self.on_sync = on_sync
        self.on_health_check = on_health_check
        self.last_synced = last_synced
        self._syncing_sources: set[str] = set()
        self._checking_health: bool = False

    @staticmethod
    def _health_dot(health_status: str | None) -> str:
        if health_status == HealthStatus.OK.value:
            return "[success]●[/success]"
        if health_status == HealthStatus.DEGRADED.value:
            return "[warning]●[/warning]"
        if health_status == HealthStatus.UNAVAILABLE.value:
            return "[error]●[/error]"
        return "[dim]●[/dim]"

    def compose(self):
        table = DataTable(id="sources_table")
        table.add_columns("Health", "Source", "Status", "Sessions", "Action")

        for source in self.sources:
            name = source.get("name", "unknown")
            sessions = source.get("sessions", 0)
            available = source.get("available", False)
            health_status = source.get("health_status")

            health_dot = self._health_dot(health_status)
            status_text = "✓ Available" if available else "- No sessions"
            status_style = "success" if available else "dim"

            button_label = "Syncing..." if name in self._syncing_sources else "Sync"
            button_disabled = name in self._syncing_sources

            sync_button = Button(
                button_label,
                id=f"sync_{name}",
                disabled=button_disabled,
                classes="sync-button",
            )

            table.add_row(
                health_dot,
                name,
                f"[{status_style}]{status_text}[/{status_style}]",
                str(sessions),
                sync_button,
                key=name,
            )

        yield table
        checking_indicator = " [warning]Checking...[/warning]" if self._checking_health else ""
        yield Static(
            f"[dim]Last Synced:[/dim] {self.last_synced}{checking_indicator} "
            "[dim]Ctrl+R to re-probe[/dim]",
            id="sources_status",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sync button presses."""
        event.stop()
        button_id = event.button.id
        if not button_id or not button_id.startswith("sync_"):
            return

        source_name = button_id.replace("sync_", "")
        if source_name in self._syncing_sources:
            return

        self._syncing_sources.add(source_name)
        self._update_button_state(source_name, syncing=True)

        self.on_sync(source_name)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle clicking a row in the sources table."""
        if not event.row_key or not event.row_key.value:
            return
        self.post_message(self.SourceSelected(str(event.row_key.value)))

    def _update_button_state(self, source_name: str, syncing: bool) -> None:
        """Update the button state for a source."""
        try:
            button = self.query_one(f"#sync_{source_name}", Button)
            if syncing:
                button.label = "Syncing..."
                button.disabled = True
            else:
                button.label = "Sync"
                button.disabled = False
        except Exception:
            pass

    def mark_sync_complete(self, source_name: str, success: bool = True) -> None:
        """Mark a sync operation as complete."""
        self._syncing_sources.discard(source_name)
        self._update_button_state(source_name, syncing=False)

        try:
            status = self.query_one("#sources_status", Static)
            checking_indicator = " [warning]Checking...[/warning]" if self._checking_health else ""
            if success:
                msg = f"[success]✓ {source_name} synced[/success]"
                status.update(
                    f"{msg} | [dim]Last Synced:[/dim] {self.last_synced}"
                    f"{checking_indicator} [dim]Ctrl+R to re-probe[/dim]"
                )
            else:
                msg = f"[error]✗ {source_name} sync failed[/error]"
                status.update(
                    f"{msg} | [dim]Last Synced:[/dim] {self.last_synced}"
                    f"{checking_indicator} [dim]Ctrl+R to re-probe[/dim]"
                )
        except Exception:
            pass

    def set_checking_health(self, checking: bool) -> None:
        """Set the checking health indicator state."""
        self._checking_health = checking
        try:
            status = self.query_one("#sources_status", Static)
            checking_indicator = " [warning]Checking...[/warning]" if checking else ""
            status.update(
                f"[dim]Last Synced:[/dim] {self.last_synced}{checking_indicator} "
                "[dim]Ctrl+R to re-probe[/dim]"
            )
        except Exception:
            pass

    def update_sources(self, sources: list[dict[str, Any]], last_synced: str | None = None) -> None:
        """Update the sources data and refresh the table."""
        self.sources = sources
        if last_synced:
            self.last_synced = last_synced

        try:
            table = self.query_one("#sources_table", DataTable)
            status = self.query_one("#sources_status", Static)
            table.remove()
            status.remove()

            new_table = DataTable(id="sources_table")
            new_table.add_columns("Health", "Source", "Status", "Sessions", "Action")

            for source in self.sources:
                name = source.get("name", "unknown")
                available = source.get("available", False)
                sessions = source.get("sessions", 0)
                health_status = source.get("health_status")

                health_dot = self._health_dot(health_status)
                status_text = "✓ Available" if available else "- No sessions"
                status_style = "success" if available else "dim"

                button_label = "Syncing..." if name in self._syncing_sources else "Sync"
                button_disabled = name in self._syncing_sources

                sync_button = Button(
                    button_label,
                    id=f"sync_{name}",
                    disabled=button_disabled,
                    classes="sync-button",
                )

                new_table.add_row(
                    health_dot,
                    name,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    str(sessions),
                    sync_button,
                    key=name,
                )

            self.mount(new_table)
            checking_indicator = " [warning]Checking...[/warning]" if self._checking_health else ""
            self.mount(
                Static(
                    f"[dim]Last Synced:[/dim] {self.last_synced}{checking_indicator} "
                    "[dim]Ctrl+R to re-probe[/dim]",
                    id="sources_status",
                )
            )
        except Exception:
            pass

    def request_health_check(self) -> None:
        """Request a health check of all sources."""
        if self.on_health_check:
            self.on_health_check()
        else:
            self.post_message(self.HealthCheckRequested())
