from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_recall.storage.models import PipelineEvent, PipelineEventAction, PipelineStage

_DEFAULT_BUCKETS_MS = (25, 50, 100, 250, 500, 1000, 5000)


def _to_stage(value: PipelineStage | str) -> PipelineStage:
    if isinstance(value, PipelineStage):
        return value
    return PipelineStage(str(value).strip().lower())


def _to_action(value: PipelineEventAction | str) -> PipelineEventAction:
    if isinstance(value, PipelineEventAction):
        return value
    return PipelineEventAction(str(value).strip().lower())


def _bucket_key(duration_ms: float, *, buckets: tuple[int, ...] = _DEFAULT_BUCKETS_MS) -> str:
    for bucket in buckets:
        if duration_ms <= bucket:
            return f"<= {bucket}ms"
    return f"> {buckets[-1]}ms"


class PipelineTelemetry:
    """Small local telemetry sink for pipeline lifecycle events."""

    def __init__(self, agent_dir: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.agent_dir = agent_dir
        self.metrics_dir = agent_dir / "metrics"
        self.events_path = self.metrics_dir / "pipeline-events.jsonl"
        self.snapshot_path = self.metrics_dir / "pipeline-metrics.json"

    @classmethod
    def from_config(
        cls, *, agent_dir: Path, config: dict[str, Any] | None = None
    ) -> PipelineTelemetry:
        telemetry_cfg = (config or {}).get("telemetry")
        enabled = True
        if isinstance(telemetry_cfg, dict):
            enabled = bool(telemetry_cfg.get("enabled", True))
        return cls(agent_dir=agent_dir, enabled=enabled)

    @staticmethod
    def create_run_id(prefix: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return f"{prefix}-{stamp}-{uuid4().hex[:8]}"

    def record_event(
        self,
        *,
        run_id: str,
        stage: PipelineStage | str,
        action: PipelineEventAction | str,
        success: bool | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineEvent:
        event = PipelineEvent(
            run_id=run_id,
            stage=_to_stage(stage),
            action=_to_action(action),
            success=success,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        if not self.enabled:
            return event

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(event.model_dump_json() + "\n")
        self._update_snapshot(event)
        return event

    def read_snapshot(self) -> dict[str, Any]:
        if not self.snapshot_path.exists():
            return self._empty_snapshot()
        try:
            loaded = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_snapshot()
        if not isinstance(loaded, dict):
            return self._empty_snapshot()
        return loaded

    def list_recent_runs(self, *, limit: int = 5) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []

        runs: dict[str, dict[str, Any]] = {}
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        for raw_line in lines:
            try:
                event = PipelineEvent.model_validate_json(raw_line)
            except Exception:  # noqa: BLE001
                continue

            run = runs.setdefault(
                event.run_id,
                {
                    "run_id": event.run_id,
                    "started_at": event.created_at,
                    "last_event_at": event.created_at,
                    "events_total": 0,
                    "by_stage": defaultdict(
                        lambda: {
                            "start": 0,
                            "complete": 0,
                            "error": 0,
                            "success": 0,
                            "failure": 0,
                            "duration_ms_total": 0.0,
                            "duration_samples": 0,
                        }
                    ),
                },
            )
            run["events_total"] += 1
            if event.created_at < run["started_at"]:
                run["started_at"] = event.created_at
            if event.created_at > run["last_event_at"]:
                run["last_event_at"] = event.created_at

            stage = str(event.stage.value)
            action = str(event.action.value)
            stage_stats = run["by_stage"][stage]
            if action in {"start", "complete", "error"}:
                stage_stats[action] += 1
            if event.success is True:
                stage_stats["success"] += 1
            elif event.success is False:
                stage_stats["failure"] += 1
            if isinstance(event.duration_ms, int | float) and event.duration_ms >= 0:
                stage_stats["duration_ms_total"] += float(event.duration_ms)
                stage_stats["duration_samples"] += 1

        sorted_runs = sorted(
            runs.values(),
            key=lambda item: item.get("last_event_at") or item.get("started_at"),
            reverse=True,
        )

        rendered: list[dict[str, Any]] = []
        for run in sorted_runs[: max(1, int(limit))]:
            started_at = run["started_at"]
            last_event_at = run["last_event_at"]
            duration_ms = max(
                0.0,
                (last_event_at - started_at).total_seconds() * 1000,
            )
            by_stage: dict[str, Any] = {}
            for stage, stage_stats in run["by_stage"].items():
                avg_duration_ms: float | None = None
                if stage_stats["duration_samples"] > 0:
                    avg_duration_ms = (
                        stage_stats["duration_ms_total"] / stage_stats["duration_samples"]
                    )
                by_stage[stage] = {
                    "start": stage_stats["start"],
                    "complete": stage_stats["complete"],
                    "error": stage_stats["error"],
                    "success": stage_stats["success"],
                    "failure": stage_stats["failure"],
                    "avg_duration_ms": avg_duration_ms,
                }
            rendered.append(
                {
                    "run_id": run["run_id"],
                    "started_at": started_at.isoformat(),
                    "last_event_at": last_event_at.isoformat(),
                    "duration_ms": round(duration_ms, 2),
                    "events_total": run["events_total"],
                    "by_stage": by_stage,
                }
            )
        return rendered

    def _update_snapshot(self, event: PipelineEvent) -> None:
        snapshot = self.read_snapshot()
        counters = snapshot.setdefault("counters", {})
        counters["events_total"] = int(counters.get("events_total", 0)) + 1

        by_stage = counters.setdefault("by_stage", {})
        stage_key = event.stage.value
        stage_stats = by_stage.setdefault(
            stage_key,
            {"start": 0, "complete": 0, "error": 0, "success": 0, "failure": 0},
        )
        action_key = event.action.value
        if action_key in stage_stats:
            stage_stats[action_key] = int(stage_stats[action_key]) + 1
        if event.success is True:
            stage_stats["success"] = int(stage_stats.get("success", 0)) + 1
        elif event.success is False:
            stage_stats["failure"] = int(stage_stats.get("failure", 0)) + 1

        if isinstance(event.duration_ms, int | float) and event.duration_ms >= 0:
            hist_root = snapshot.setdefault("duration_histograms_ms", {})
            stage_hist = hist_root.setdefault(stage_key, {})
            bucket = _bucket_key(float(event.duration_ms))
            stage_hist[bucket] = int(stage_hist.get(bucket, 0)) + 1

        snapshot["schema_version"] = 1
        snapshot["updated_at"] = datetime.now(UTC).isoformat()
        self.snapshot_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _empty_snapshot() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "updated_at": None,
            "counters": {"events_total": 0, "by_stage": {}},
            "duration_histograms_ms": {},
        }
