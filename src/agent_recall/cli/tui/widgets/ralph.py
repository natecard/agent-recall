from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table

from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore


@dataclass(frozen=True)
class RalphStatusWidget:
    agent_dir: Path
    max_iterations: int | None = None

    def render(self) -> Panel:
        current_report = self._load_current_report()
        state_payload = self._load_state_payload()
        if current_report is not None:
            return self._render_running(current_report, state_payload)
        return self._render_idle(state_payload)

    def _load_current_report(self) -> IterationReport | None:
        store = IterationReportStore(self.agent_dir / "ralph")
        return store.load_current()

    def _load_state_payload(self) -> dict[str, Any]:
        path = self.agent_dir / "ralph" / "ralph_state.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _render_running(self, report: IterationReport, state_payload: dict[str, Any]) -> Panel:
        table = self._build_table()
        table.add_row("Status", "[success]Running[/success]")
        table.add_row("Iteration", str(report.iteration))
        table.add_row("Max Iterations", self._format_max_iterations())
        table.add_row("PRD Item", self._format_prd_item(report.item_id, report.item_title))
        started_at = self._format_timestamp(report.started_at)
        if started_at:
            table.add_row("Started", started_at)
        loop_status = self._format_state_value(state_payload.get("status"))
        if loop_status:
            table.add_row("Loop", loop_status)
        return Panel(table, title="Ralph Status", border_style="accent")

    def _render_idle(self, state_payload: dict[str, Any]) -> Panel:
        table = self._build_table()
        table.add_row("Status", "[dim]Idle[/dim]")
        loop_status = self._format_state_value(state_payload.get("status"))
        if loop_status:
            table.add_row("Loop", loop_status)
        last_run = self._format_timestamp(state_payload.get("last_run_at"))
        table.add_row("Last Run", last_run or "Never")
        last_outcome = self._format_state_value(state_payload.get("last_outcome"))
        table.add_row("Last Outcome", last_outcome or "Unknown")
        total_iterations = state_payload.get("total_iterations")
        table.add_row("Total Iterations", self._format_count(total_iterations))
        return Panel(table, title="Ralph Status", border_style="accent")

    @staticmethod
    def _build_table() -> Table:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Item", style="table_header", width=14, no_wrap=True)
        table.add_column("Value", overflow="fold")
        return table

    def _format_max_iterations(self) -> str:
        if isinstance(self.max_iterations, int) and self.max_iterations > 0:
            return str(self.max_iterations)
        return "Unknown"

    @staticmethod
    def _format_prd_item(item_id: str, title: str) -> str:
        item_id = item_id.strip()
        title = title.strip()
        if item_id and title:
            return f"{item_id} - {title}"
        return item_id or title or "Unknown"

    @staticmethod
    def _format_count(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "0"

    @staticmethod
    def _format_state_value(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _format_timestamp(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            normalized = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
            return normalized.strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return value
            normalized = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            return normalized.strftime("%Y-%m-%d %H:%M UTC")
        return None
