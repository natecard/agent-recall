from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.widgets import DataTable

from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore


class InteractiveTimelineWidget(DataTable):
    """Interactive timeline widget with clickable entries."""

    DEFAULT_CSS = """
    InteractiveTimelineWidget {
        height: 100%;
        border: solid $accent;
    }
    InteractiveTimelineWidget > .datatable--header {
        background: $surface-darken-1;
    }
    InteractiveTimelineWidget > .datatable--cursor {
        background: $accent 30%;
    }
    """

    def __init__(
        self,
        report_store: IterationReportStore,
        on_select: Callable[[IterationReport], None] | None = None,
        max_entries: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._report_store = report_store
        self._on_select = on_select
        self._max_entries = max_entries
        self._reports: list[IterationReport] = []

    def on_mount(self) -> None:
        """Load and display timeline entries when mounted."""
        self._load_timeline()
        self.focus()

    def _load_timeline(self) -> None:
        """Load timeline entries from the report store."""
        self.clear(columns=True)
        self._reports = self._report_store.load_recent(count=self._max_entries)

        # Add columns
        self.add_column("Iteration", width=8)
        self.add_column("Item", width=20)
        self.add_column("Outcome", width=12)
        self.add_column("When", width=20)

        # Add rows
        for report in self._reports:
            outcome_str = report.outcome.value if report.outcome else "In Progress"
            when_str = self._format_when(report)
            self.add_row(
                str(report.iteration),
                report.item_id or "—",
                outcome_str,
                when_str,
                key=str(report.iteration),
            )
        if self._reports:
            self.cursor_type = "row"

    def _format_when(self, report: IterationReport) -> str:
        """Format the timestamp for display."""
        from datetime import UTC

        timestamp = report.completed_at or report.started_at
        if not timestamp:
            return "Unknown"

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)

        return timestamp.strftime("%m-%d %H:%M")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection - show detail view."""
        if self._on_select is None:
            return

        row_key = event.row_key.value
        if row_key is None:
            return

        # Find the report for this row
        iteration = int(row_key)
        for report in self._reports:
            if report.iteration == iteration:
                self._on_select(report)
                break

    def refresh_timeline(self) -> None:
        """Reload and refresh the timeline display."""
        self._load_timeline()
