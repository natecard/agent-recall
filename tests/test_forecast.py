from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from agent_recall.llm.base import LLMProvider, LLMResponse
from agent_recall.ralph.forecast import ForecastConfig, ForecastGenerator
from agent_recall.ralph.iteration_store import (
    IterationOutcome,
    IterationReport,
    IterationReportStore,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier


def _write_report(store: IterationReportStore, report: IterationReport) -> None:
    store.iterations_dir.mkdir(parents=True, exist_ok=True)
    path = store.iterations_dir / f"{report.iteration:03d}.json"
    path.write_text(json_dumps(report), encoding="utf-8")


def json_dumps(report: IterationReport) -> str:
    import json

    return json.dumps(report.to_dict(), indent=2)


class StubLLM(LLMProvider):
    def __init__(self, content: str = "LLM output", raise_error: bool = False) -> None:
        self._content = content
        self._raise_error = raise_error
        self.last_messages: list | None = None
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    async def generate(
        self,
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.last_messages = list(messages)
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        if self._raise_error:
            raise RuntimeError("LLM failed")
        return LLMResponse(content=self._content, model=self.model_name)

    def validate(self) -> tuple[bool, str]:
        return (True, "")


def test_generate_empty_returns_placeholder(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)

    content = generator.generate()

    assert "No iterations yet" in content
    assert "# Current Situation" in content


def test_generate_single_report_includes_sections(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-003",
        item_title="Forecast",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        outcome=IterationOutcome.COMPLETED,
        validation_hint="all good",
        pattern_that_worked="keep scope small",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)

    content = generator.generate()

    assert "## Trajectory" in content
    assert "001 ✓" in content
    assert "## Current Status" in content
    assert "COMPLETED" in content
    assert "## Watch For" in content
    assert "all good" in content
    assert "## Emerging Pattern" in content
    assert "keep scope small" in content


def test_generate_trajectory_order_and_symbols(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    for iteration, outcome in [
        (1, IterationOutcome.COMPLETED),
        (2, IterationOutcome.VALIDATION_FAILED),
        (3, IterationOutcome.COMPLETED),
        (4, IterationOutcome.BLOCKED),
        (5, IterationOutcome.TIMEOUT),
    ]:
        report = IterationReport(
            iteration=iteration,
            item_id=f"WM-{iteration:03d}",
            item_title="Forecast",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            outcome=outcome,
            validation_hint=f"hint {iteration}",
        )
        _write_report(store, report)

    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)
    content = generator.generate()

    trajectory = _extract_section(content, "## Trajectory")
    assert trajectory[0].startswith("- 001 ✓")
    assert trajectory[1].startswith("- 002 ✗")
    assert trajectory[2].startswith("- 003 ✓")
    assert trajectory[3].startswith("- 004 ✗")
    assert trajectory[4].startswith("- 005 ✗")


def test_generate_missing_optional_fields_omits_emerging_pattern(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-003",
        item_title="Forecast",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        outcome=IterationOutcome.VALIDATION_FAILED,
        validation_hint="",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)

    content = generator.generate()

    assert "## Emerging Pattern" not in content
    assert "No immediate risks identified" in content


def test_write_forecast_overwrites_recent(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-003",
        item_title="Forecast",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        outcome=IterationOutcome.COMPLETED,
        validation_hint="ok",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)

    content = generator.write_forecast()

    recent_path = tmp_path / "RECENT.md"
    assert recent_path.exists()
    assert recent_path.read_text() == content
    assert "## Trajectory" in files.read_tier(KnowledgeTier.RECENT)


def test_write_forecast_triggers_tier_compaction(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-003",
        item_title="Forecast",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        outcome=IterationOutcome.COMPLETED,
        validation_hint="ok",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    files.write_config({"compaction": {"max_tier_tokens": 1}})
    generator = ForecastGenerator(ralph_dir, files)

    content = generator.write_forecast()

    recent = files.read_tier(KnowledgeTier.RECENT)
    assert recent == content
    assert recent.startswith("# Recent")
    assert "Current Situation" not in recent


def test_generate_uses_llm_on_consecutive_failures(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    for iteration, outcome in [
        (1, IterationOutcome.COMPLETED),
        (2, IterationOutcome.VALIDATION_FAILED),
        (3, IterationOutcome.BLOCKED),
    ]:
        report = IterationReport(
            iteration=iteration,
            item_id="WM-004",
            item_title="Forecast",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            outcome=outcome,
            failure_reason=f"fail {iteration}",
        )
        _write_report(store, report)
    files = FileStorage(tmp_path)
    config = ForecastConfig(window=5, use_llm=True, llm_on_consecutive_failures=2)
    generator = ForecastGenerator(ralph_dir, files, config=config)
    llm = StubLLM(content="LLM forecast")

    content = generator.generate(llm=llm)

    assert content.startswith("# Current Situation\n")
    assert "LLM forecast" in content
    assert llm.last_messages is not None


def test_generate_skips_llm_below_threshold(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    for iteration, outcome in [
        (1, IterationOutcome.COMPLETED),
        (2, IterationOutcome.VALIDATION_FAILED),
    ]:
        report = IterationReport(
            iteration=iteration,
            item_id="WM-004",
            item_title="Forecast",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            outcome=outcome,
            validation_hint=f"hint {iteration}",
        )
        _write_report(store, report)
    files = FileStorage(tmp_path)
    config = ForecastConfig(window=5, use_llm=True, llm_on_consecutive_failures=2)
    generator = ForecastGenerator(ralph_dir, files, config=config)
    llm = StubLLM(content="LLM forecast")

    content = generator.generate(llm=llm)

    assert "## Trajectory" in content
    assert "LLM forecast" not in content


def test_generate_llm_fallback_on_error(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    store = IterationReportStore(ralph_dir)
    report = IterationReport(
        iteration=1,
        item_id="WM-004",
        item_title="Forecast",
        started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
        outcome=IterationOutcome.VALIDATION_FAILED,
        validation_hint="failed",
    )
    _write_report(store, report)
    files = FileStorage(tmp_path)
    config = ForecastConfig(window=5, use_llm=True, llm_on_consecutive_failures=1)
    generator = ForecastGenerator(ralph_dir, files, config=config)
    llm = StubLLM(raise_error=True)

    content = generator.generate(llm=llm)

    assert "## Trajectory" in content


def test_generate_with_llm_prompt_includes_pass_fail(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "ralph"
    files = FileStorage(tmp_path)
    generator = ForecastGenerator(ralph_dir, files)
    reports = [
        IterationReport(
            iteration=2,
            item_id="WM-004",
            item_title="Forecast",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            outcome=IterationOutcome.VALIDATION_FAILED,
            failure_reason="boom",
        ),
        IterationReport(
            iteration=1,
            item_id="WM-004",
            item_title="Forecast",
            started_at=datetime(2026, 2, 14, 12, 0, 0, tzinfo=UTC),
            outcome=IterationOutcome.COMPLETED,
        ),
    ]
    llm = StubLLM(content="LLM forecast")

    asyncio.run(generator._generate_with_llm(reports, llm))

    assert llm.last_messages is not None
    prompt = llm.last_messages[0].content
    assert "PASS 001" in prompt
    assert "FAIL 002 - boom" in prompt
    assert "Current item: WM-004" in prompt
    assert llm.last_temperature == 0.3
    assert llm.last_max_tokens == 400


def _extract_section(content: str, header: str) -> list[str]:
    lines = content.splitlines()
    try:
        start = lines.index(header)
    except ValueError:
        return []
    collected: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("-"):
            collected.append(line)
    return collected
