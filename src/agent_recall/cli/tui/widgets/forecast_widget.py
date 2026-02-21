from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from agent_recall.ralph.costs import CostBreakdown, CostSummary, summarize_costs
from agent_recall.ralph.iteration_store import (
    IterationOutcome,
    IterationReport,
    IterationReportStore,
)


@dataclass(frozen=True)
class ForecastWidget:
    agent_dir: Path
    prd_path: Path | None = None
    max_iterations: int | None = None
    cost_budget_usd: float | None = None
    format_usd: Callable[[float], str] = lambda x: f"${x:.2f}"

    def render(self) -> Panel:
        store = IterationReportStore(self.agent_dir / "ralph")
        reports = store.load_all()
        cost_summary = summarize_costs(reports)
        prd_data = self._load_prd_data()
        risk_items = self._identify_risk_items(reports)

        sections = [
            self._render_queue_summary(prd_data, reports),
            self._render_cost_tracker(cost_summary),
            self._render_item_cost_table(cost_summary),
            self._render_risk_table(risk_items),
        ]
        return Panel(
            Group(*sections),
            title="Ralph Forecast",
            border_style="accent",
        )

    def _load_prd_data(self) -> dict[str, Any]:
        if self.prd_path is None or not self.prd_path.exists():
            return {}
        try:
            payload = json.loads(self.prd_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _render_queue_summary(
        self, prd_data: dict[str, Any], reports: list[IterationReport]
    ) -> Group:
        items = prd_data.get("items", [])
        if not isinstance(items, list):
            items = []

        total_items = len(items)
        passed_items = sum(1 for item in items if item.get("passes", False))
        remaining_items = total_items - passed_items

        completed_iterations = len(reports)
        avg_iterations_per_item = 1.5
        if total_items > 0 and completed_iterations > 0:
            unique_items = len(set(r.item_id for r in reports if r.item_id))
            if unique_items > 0:
                avg_iterations_per_item = completed_iterations / unique_items

        estimated_remaining = max(1, int(remaining_items * avg_iterations_per_item))
        if self.max_iterations is not None and estimated_remaining > self.max_iterations:
            estimated_remaining = self.max_iterations

        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Metric", style="table_header", width=20, no_wrap=True)
        table.add_column("Value", overflow="fold")

        table.add_row("PRD Items", f"{total_items} total, {passed_items} passed")
        table.add_row("Remaining Items", str(remaining_items))
        table.add_row("Completed Iterations", str(completed_iterations))
        table.add_row("Est. Iterations to Completion", str(estimated_remaining))

        return Group(Rule("Queue Summary"), table)

    def _render_cost_tracker(self, cost_summary: CostSummary) -> Group:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Metric", style="table_header", width=20, no_wrap=True)
        table.add_column("Value", overflow="fold")

        total_cost = cost_summary.total_cost_usd
        total_tokens = cost_summary.total_tokens

        budget_text = "-"
        overage_text = ""
        if self.cost_budget_usd is not None:
            budget_text = self.format_usd(self.cost_budget_usd)
            overage = total_cost - self.cost_budget_usd
            if overage > 0:
                overage_text = f" [error](+{self.format_usd(overage)} over)[/error]"

        table.add_row("Total Tokens", f"{total_tokens:,}")
        table.add_row("Total Cost", self.format_usd(total_cost))
        table.add_row("Cost Budget", f"{budget_text}{overage_text}")

        return Group(Rule("Cost Tracker"), table)

    def _render_item_cost_table(self, cost_summary: CostSummary) -> Group:
        if not cost_summary.items:
            return Group(Rule("Per-Item Cost History"), "[dim]No iteration data yet.[/dim]")

        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Item", style="table_header", no_wrap=True, max_width=30)
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Outcome", justify="center")

        for item in cost_summary.items[:10]:
            label = item.item_id
            if item.item_title:
                truncated_title = (
                    item.item_title[:20] + "..." if len(item.item_title) > 20 else item.item_title
                )
                label = f"{item.item_id} · {truncated_title}"
            outcome = self._get_item_outcome(item)
            outcome_style = self._outcome_style(outcome)
            table.add_row(
                label[:30],
                f"{item.total_tokens:,}",
                self.format_usd(item.cost_usd),
                f"[{outcome_style}]{outcome}[/{outcome_style}]",
            )

        if len(cost_summary.items) > 10:
            remaining = len(cost_summary.items) - 10
            table.add_row(f"... and {remaining} more", "", "", "")

        return Group(Rule("Per-Item Cost History"), table)

    def _get_item_outcome(self, item: CostBreakdown) -> str:
        store = IterationReportStore(self.agent_dir / "ralph")
        reports = store.load_all()
        for report in reversed(reports):
            if report.item_id == item.item_id:
                if report.outcome is None:
                    return "running"
                return report.outcome.value.lower()
        return "unknown"

    def _outcome_style(self, outcome: str) -> str:
        if outcome == "completed":
            return "success"
        if outcome in {"validation_failed", "blocked", "timeout"}:
            return "error"
        if outcome == "scope_reduced":
            return "warning"
        return "dim"

    def _identify_risk_items(self, reports: list[IterationReport]) -> list[dict[str, Any]]:
        if not reports:
            return []

        failure_streaks: dict[str, int] = defaultdict(int)
        for report in reports:
            if report.outcome in {
                IterationOutcome.VALIDATION_FAILED,
                IterationOutcome.BLOCKED,
            }:
                failure_streaks[report.item_id] += 1
            elif report.outcome == IterationOutcome.COMPLETED:
                failure_streaks[report.item_id] = 0

        risk_items = []
        for item_id, streak in failure_streaks.items():
            if streak >= 2:
                title = ""
                for report in reports:
                    if report.item_id == item_id:
                        title = report.item_title
                        break
                risk_items.append(
                    {
                        "item_id": item_id,
                        "item_title": title,
                        "consecutive_failures": streak,
                    }
                )

        risk_items.sort(key=lambda x: x["consecutive_failures"], reverse=True)
        return risk_items[:5]

    def _render_risk_table(self, risk_items: list[dict[str, Any]]) -> Group:
        if not risk_items:
            return Group(Rule("Risk Indicators"), "[dim]No items with repeated failures.[/dim]")

        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Item", style="table_header", no_wrap=True, max_width=30)
        table.add_column("Failures", justify="right")
        table.add_column("Risk", justify="center")

        for item in risk_items:
            label = item["item_id"]
            if item["item_title"]:
                truncated = (
                    item["item_title"][:15] + "..."
                    if len(item["item_title"]) > 15
                    else item["item_title"]
                )
                label = f"{item['item_id']} · {truncated}"
            failures = item["consecutive_failures"]
            if failures >= 3:
                risk_indicator = "[error]● HIGH[/error]"
            else:
                risk_indicator = "[warning]● MED[/warning]"
            table.add_row(label[:30], str(failures), risk_indicator)

        return Group(Rule("Risk Indicators"), table)
