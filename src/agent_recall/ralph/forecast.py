from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_recall.llm import Message
from agent_recall.llm.base import LLMProvider
from agent_recall.ralph.iteration_store import (
    IterationOutcome,
    IterationReport,
    IterationReportStore,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier

FORECAST_LLM_PROMPT = (
    "You are generating a concise Ralph forecast from iteration summaries. "
    "Respond with Trajectory, Current Status, Watch For, and Emerging Pattern. "
    "Keep the response under 150 words.\n\n"
    "Iteration summaries:\n{iteration_summaries}\n\n"
    "Current item: {current_item}"
)


@dataclass
class ForecastConfig:
    window: int = 5
    use_llm: bool = False
    llm_on_consecutive_failures: int = 2
    llm_model: str | None = None


class ForecastGenerator:
    def __init__(
        self,
        ralph_dir: Path,
        files: FileStorage,
        config: ForecastConfig | None = None,
    ) -> None:
        self.ralph_dir = ralph_dir
        self.files = files
        self.config = config or ForecastConfig()

    def _empty_forecast(self) -> str:
        timestamp = _format_timestamp(datetime.now(UTC))
        lines = [
            "# Current Situation",
            f"Generated at {timestamp}",
            "",
            "## Trajectory",
            "- No iterations yet.",
            "",
            "## Current Status",
            "No iteration history available.",
            "",
            "## Watch For",
            "No watch items yet.",
        ]
        return "\n".join(lines)

    def _generate_heuristic(self, reports: list[IterationReport]) -> str:
        if not reports:
            return self._empty_forecast()

        timestamp = _format_timestamp(datetime.now(UTC))
        ordered = list(reversed(reports))
        latest = reports[0]

        trajectory_lines = ["## Trajectory"]
        for report in ordered:
            outcome = report.outcome
            symbol = "✓" if outcome == IterationOutcome.COMPLETED else "✗"
            hint = _truncate(_trajectory_hint(report), 50) or "no hint"
            trajectory_lines.append(f"- {report.iteration:03d} {symbol} {hint}")

        lines = [
            "# Current Situation",
            f"Generated at {timestamp}",
            "",
            *trajectory_lines,
            "",
            "## Current Status",
            f"Latest outcome: {_outcome_label(latest)}",
            "",
            "## Watch For",
            _watch_for_text(latest),
        ]

        pattern = latest.pattern_that_worked
        if pattern:
            lines.extend(["", "## Emerging Pattern", pattern])

        return "\n".join(lines)

    def generate(self, llm: LLMProvider | None = None) -> str:
        window = max(self.config.window, 0)
        store = IterationReportStore(self.ralph_dir)
        reports = store.load_recent(count=window)
        if not reports:
            return self._empty_forecast()
        if self.config.use_llm and llm is not None:
            consecutive_failures = 0
            for report in reports:
                if report.outcome == IterationOutcome.COMPLETED:
                    break
                consecutive_failures += 1
            if consecutive_failures >= self.config.llm_on_consecutive_failures:
                try:
                    return asyncio.run(self._generate_with_llm(reports, llm))
                except Exception:  # noqa: BLE001
                    return self._generate_heuristic(reports)
        return self._generate_heuristic(reports)

    def write_forecast(self, llm: LLMProvider | None = None) -> str:
        content = self.generate(llm=llm)
        self.files.write_tier(KnowledgeTier.RECENT, content)
        return content

    async def _generate_with_llm(self, reports: list[IterationReport], llm: LLMProvider) -> str:
        summaries: list[str] = []
        for report in reversed(reports):
            status = "PASS" if report.outcome == IterationOutcome.COMPLETED else "FAIL"
            hint = report.failure_reason or report.gotcha_discovered or report.validation_hint
            hint_text = f" - {hint}" if hint else ""
            summaries.append(f"{status} {report.iteration:03d}{hint_text}")
        prompt = FORECAST_LLM_PROMPT.format(
            iteration_summaries="\n".join(summaries),
            current_item=reports[0].item_id or "unknown",
        )
        message = Message(role="user", content=prompt)
        response = await llm.generate([message], temperature=0.3, max_tokens=400)
        header = "# Current Situation\n"
        return header + response.content.strip()


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _truncate(value: str | None, limit: int) -> str:
    if not value:
        return ""
    trimmed = value.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[:limit]


def _trajectory_hint(report: IterationReport) -> str:
    if report.validation_hint:
        return report.validation_hint
    if report.failure_reason:
        return report.failure_reason
    if report.summary:
        return report.summary
    return ""


def _outcome_label(report: IterationReport) -> str:
    if report.outcome is None:
        return "UNKNOWN"
    return report.outcome.value


def _watch_for_text(report: IterationReport) -> str:
    if report.gotcha_discovered:
        return report.gotcha_discovered
    if report.validation_hint:
        return _truncate(report.validation_hint, 100)
    if report.scope_change:
        return report.scope_change
    return "No immediate risks identified."
