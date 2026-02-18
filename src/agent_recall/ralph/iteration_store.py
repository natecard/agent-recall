from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, overload


class IterationOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    SCOPE_REDUCED = "SCOPE_REDUCED"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"


@dataclass
class IterationReport:
    # Harness-set before agent
    iteration: int = 0
    item_id: str = ""
    item_title: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Agent-set during work
    outcome: IterationOutcome | None = None
    summary: str | None = None
    failure_reason: str | None = None
    gotcha_discovered: str | None = None
    pattern_that_worked: str | None = None
    scope_change: str | None = None
    token_usage: dict[str, int] | None = None
    token_model: str | None = None

    # Harness-set after agent
    validation_exit_code: int | None = None
    validation_hint: str | None = None
    files_changed: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "item_id": self.item_id,
            "item_title": self.item_title,
            "started_at": _format_datetime(self.started_at),
            "outcome": self.outcome.value if self.outcome is not None else None,
            "summary": self.summary,
            "failure_reason": self.failure_reason,
            "gotcha_discovered": self.gotcha_discovered,
            "pattern_that_worked": self.pattern_that_worked,
            "scope_change": self.scope_change,
            "token_usage": dict(self.token_usage) if self.token_usage else None,
            "token_model": self.token_model,
            "validation_exit_code": self.validation_exit_code,
            "validation_hint": self.validation_hint,
            "files_changed": list(self.files_changed),
            "commit_hash": self.commit_hash,
            "completed_at": _format_datetime(self.completed_at),
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationReport:
        outcome_value = data.get("outcome")
        outcome = None
        if outcome_value:
            try:
                outcome = IterationOutcome(str(outcome_value))
            except ValueError:
                outcome = None

        files_changed_raw = data.get("files_changed")
        files_changed = list(files_changed_raw) if isinstance(files_changed_raw, list) else []
        started_at = _parse_datetime(data.get("started_at"))
        if started_at is None:
            started_at = datetime.now(UTC)

        return cls(
            iteration=int(data.get("iteration") or 0),
            item_id=str(data.get("item_id") or ""),
            item_title=str(data.get("item_title") or ""),
            started_at=started_at,
            outcome=outcome,
            summary=_optional_str(data.get("summary")),
            failure_reason=_optional_str(data.get("failure_reason")),
            gotcha_discovered=_optional_str(data.get("gotcha_discovered")),
            pattern_that_worked=_optional_str(data.get("pattern_that_worked")),
            scope_change=_optional_str(data.get("scope_change")),
            token_usage=_optional_usage(data.get("token_usage")),
            token_model=_optional_str(data.get("token_model")),
            validation_exit_code=_optional_int(data.get("validation_exit_code")),
            validation_hint=_optional_str(data.get("validation_hint")),
            files_changed=files_changed,
            commit_hash=_optional_str(data.get("commit_hash")),
            completed_at=_parse_datetime(data.get("completed_at"), allow_none=True),
            duration_seconds=_optional_float(data.get("duration_seconds")),
        )


class IterationReportStore:
    def __init__(self, ralph_dir: Path) -> None:
        self.ralph_dir = ralph_dir
        self.iterations_dir = ralph_dir / "iterations"
        self.current_path = self.iterations_dir / "current.json"

    def create_for_iteration(
        self,
        iteration: int,
        item_id: str,
        item_title: str,
    ) -> IterationReport:
        resolved_iteration = self._allocate_iteration_number(iteration)
        report = IterationReport(
            iteration=resolved_iteration,
            item_id=item_id,
            item_title=item_title,
            started_at=datetime.now(UTC),
        )
        self.save_current(report)
        return report

    def finalize_current(
        self, validation_exit: int, validation_hint: str | None
    ) -> IterationReport | None:
        report = self.load_current()
        if report is None:
            return None
        now = datetime.now(UTC)
        report.validation_exit_code = validation_exit
        report.validation_hint = validation_hint
        report.completed_at = now
        report.duration_seconds = self._compute_duration_seconds(report, now)
        if validation_exit == 0:
            report.outcome = IterationOutcome.COMPLETED
        self._archive_report(report)
        try:
            self.current_path.unlink()
        except FileNotFoundError:
            pass
        return report

    def load_current(self) -> IterationReport | None:
        return self._load_report_path(self.current_path)

    def save_current(self, report: IterationReport) -> None:
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        self.current_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_recent(self, count: int = 10) -> list[IterationReport]:
        if count <= 0:
            return []
        if not self.iterations_dir.exists():
            return []
        archived = self._iter_archive_paths()
        archived.sort(key=lambda path: path.name, reverse=True)
        reports: list[IterationReport] = []
        for path in archived:
            report = self._load_report_path(path)
            if report is None:
                continue
            reports.append(report)
            if len(reports) >= count:
                break
        return reports

    def load_all(self) -> list[IterationReport]:
        if not self.iterations_dir.exists():
            return []
        archived = self._iter_archive_paths()
        archived.sort(key=lambda path: path.name)
        reports: list[IterationReport] = []
        for path in archived:
            report = self._load_report_path(path)
            if report is None:
                continue
            reports.append(report)
        return reports

    def _archive_report(self, report: IterationReport) -> None:
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        archive_iteration = report.iteration if report.iteration > 0 else 1
        archive_path = self.iterations_dir / f"{archive_iteration:03d}.json"
        if archive_path.exists():
            archive_iteration = self._allocate_iteration_number(archive_iteration)
            archive_path = self.iterations_dir / f"{archive_iteration:03d}.json"
        report.iteration = archive_iteration
        archive_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    def _allocate_iteration_number(self, requested: int) -> int:
        preferred = requested if requested > 0 else 1
        used = self._used_iteration_numbers(include_current=True)
        if preferred not in used:
            return preferred

        candidate = max(used) if used else preferred
        if candidate < preferred:
            candidate = preferred
        while candidate in used:
            candidate += 1
        return candidate

    def _used_iteration_numbers(self, *, include_current: bool) -> set[int]:
        used: set[int] = set()
        for path in self._iter_archive_paths():
            parsed = self._parse_archive_iteration(path)
            if parsed is not None:
                used.add(parsed)

        if include_current:
            current = self.load_current()
            if current is not None and current.iteration > 0:
                used.add(current.iteration)

        return used

    def _parse_archive_iteration(self, path: Path) -> int | None:
        stem = path.stem
        if not stem.isdigit():
            return None
        value = int(stem)
        return value if value > 0 else None

    def save_current_diff(self, report: IterationReport, diff_text: str) -> None:
        if not diff_text:
            return
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        path = self.iterations_dir / f"{report.iteration:03d}.diff"
        path.write_text(diff_text, encoding="utf-8")

    def load_diff_for_iteration(self, iteration: int) -> str | None:
        path = self.iterations_dir / f"{iteration:03d}.diff"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _iter_archive_paths(self) -> list[Path]:
        return [
            path
            for path in self.iterations_dir.glob("*.json")
            if path.name != self.current_path.name
        ]

    def _compute_duration_seconds(self, report: IterationReport, completed_at: datetime) -> float:
        started_at = report.started_at
        if not isinstance(started_at, datetime):
            return 0.0
        duration = completed_at - started_at
        return max(duration.total_seconds(), 0.0)

    def _load_report_path(self, path: Path) -> IterationReport | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return IterationReport.from_dict(payload)
        except (TypeError, ValueError):
            return None


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat()


@overload
def _parse_datetime(value: Any, *, allow_none: Literal[True]) -> datetime | None: ...


@overload
def _parse_datetime(value: Any, *, allow_none: Literal[False] = False) -> datetime: ...


def _parse_datetime(value: Any, *, allow_none: bool = False) -> datetime | None:
    if value is None:
        if allow_none:
            return None
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = datetime.now(UTC)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    if allow_none:
        return None
    return datetime.now(UTC)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, int] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        parsed_value = _optional_int(raw)
        if parsed_value is None:
            continue
        parsed[key] = parsed_value
    return parsed or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
