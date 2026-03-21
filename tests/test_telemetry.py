from __future__ import annotations

from pathlib import Path

from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.storage.models import PipelineEventAction, PipelineStage


def test_pipeline_telemetry_records_snapshot_and_recent_runs(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".agent"
    telemetry = PipelineTelemetry(agent_dir=agent_dir, enabled=True)
    run_id = telemetry.create_run_id("sync")

    telemetry.record_event(
        run_id=run_id,
        stage=PipelineStage.EXTRACT,
        action=PipelineEventAction.COMPLETE,
        success=True,
        duration_ms=48.0,
        metadata={"source": "cursor", "session_id": "cursor-1"},
    )
    telemetry.record_event(
        run_id=run_id,
        stage=PipelineStage.INGEST,
        action=PipelineEventAction.COMPLETE,
        success=True,
        duration_ms=12.0,
        metadata={"entries_written": 1},
    )

    snapshot = telemetry.read_snapshot()
    assert snapshot["counters"]["events_total"] == 2
    assert snapshot["counters"]["by_stage"]["extract"]["complete"] == 1
    assert snapshot["counters"]["by_stage"]["ingest"]["complete"] == 1
    assert snapshot["duration_histograms_ms"]["extract"]["<= 50ms"] == 1
    assert snapshot["duration_histograms_ms"]["ingest"]["<= 25ms"] == 1

    runs = telemetry.list_recent_runs(limit=1)
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == run_id
    assert run["events_total"] == 2
    assert run["by_stage"]["extract"]["success"] == 1
    assert run["by_stage"]["ingest"]["success"] == 1


def test_pipeline_telemetry_disabled_does_not_write_files(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".agent"
    telemetry = PipelineTelemetry(agent_dir=agent_dir, enabled=False)
    run_id = telemetry.create_run_id("sync")
    telemetry.record_event(
        run_id=run_id,
        stage=PipelineStage.COMPACT,
        action=PipelineEventAction.COMPLETE,
        success=True,
        duration_ms=88.0,
    )

    assert not telemetry.events_path.exists()
    assert telemetry.read_snapshot()["counters"]["events_total"] == 0
