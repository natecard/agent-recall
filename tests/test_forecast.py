from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_recall.ralph.forecast import ForecastGenerator
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
