from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent_recall.ralph.iteration_store import (
    IterationOutcome,
    IterationReport,
    IterationReportStore,
)


def test_iteration_report_round_trip() -> None:
    started_at = datetime(2026, 2, 13, 12, 0, 0, tzinfo=UTC)
    completed_at = datetime(2026, 2, 13, 12, 30, 0, tzinfo=UTC)
    report = IterationReport(
        iteration=3,
        item_id="WM-001",
        item_title="Iteration Record Store",
        started_at=started_at,
        outcome=IterationOutcome.COMPLETED,
        summary="done",
        failure_reason=None,
        gotcha_discovered="none",
        pattern_that_worked="keep it simple",
        scope_change=None,
        validation_exit_code=0,
        validation_hint="",
        files_changed=["src/foo.py", "tests/test_foo.py"],
        commit_hash="deadbeef",
        completed_at=completed_at,
        duration_seconds=1800.0,
    )

    payload = report.to_dict()
    restored = IterationReport.from_dict(payload)

    assert restored.iteration == report.iteration
    assert restored.item_id == report.item_id
    assert restored.item_title == report.item_title
    assert restored.started_at == report.started_at
    assert restored.outcome == report.outcome
    assert restored.summary == report.summary
    assert restored.failure_reason == report.failure_reason
    assert restored.gotcha_discovered == report.gotcha_discovered
    assert restored.pattern_that_worked == report.pattern_that_worked
    assert restored.scope_change == report.scope_change
    assert restored.validation_exit_code == report.validation_exit_code
    assert restored.validation_hint == report.validation_hint
    assert restored.files_changed == report.files_changed
    assert restored.commit_hash == report.commit_hash
    assert restored.completed_at == report.completed_at
    assert restored.duration_seconds == report.duration_seconds


def test_iteration_report_store_create_and_finalize(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "agent_recall" / "ralph"
    store = IterationReportStore(ralph_dir)

    store.create_for_iteration(1, "WM-001", "Iteration Record Store")
    assert store.current_path.exists()
    loaded = store.load_current()
    assert loaded is not None
    assert loaded.item_id == "WM-001"

    finalized = store.finalize_current(0, "")
    assert finalized is not None
    assert finalized.outcome == IterationOutcome.COMPLETED
    assert (store.iterations_dir / "001.json").exists()
    assert not store.current_path.exists()


def test_iteration_report_store_finalize_missing_returns_none(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "agent_recall" / "ralph"
    store = IterationReportStore(ralph_dir)

    assert store.finalize_current(1, "failed") is None


def test_iteration_report_store_load_current_corrupt_returns_none(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "agent_recall" / "ralph"
    store = IterationReportStore(ralph_dir)
    store.iterations_dir.mkdir(parents=True, exist_ok=True)
    store.current_path.write_text("{not json", encoding="utf-8")

    assert store.load_current() is None


def test_iteration_report_store_load_recent_orders_newest_first(tmp_path: Path) -> None:
    ralph_dir = tmp_path / "agent_recall" / "ralph"
    store = IterationReportStore(ralph_dir)
    store.iterations_dir.mkdir(parents=True, exist_ok=True)

    for iteration in [1, 2, 3]:
        report = IterationReport(
            iteration=iteration,
            item_id=f"WM-{iteration:03d}",
            item_title="Test",
            started_at=datetime(2026, 2, 13, 12, 0, 0, tzinfo=UTC),
            outcome=IterationOutcome.COMPLETED,
        )
        path = store.iterations_dir / f"{iteration:03d}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    recent = store.load_recent(count=2)
    assert [report.iteration for report in recent] == [3, 2]
