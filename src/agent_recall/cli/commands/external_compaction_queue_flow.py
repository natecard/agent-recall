from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rich import box
from rich.table import Table


class ExternalCompactionQueueService(Protocol):
    def list_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def approve_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]: ...

    def reject_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]: ...


@dataclass(frozen=True)
class QueueListRequest:
    states: list[str] | None
    limit: int
    with_attribution: bool = False


@dataclass(frozen=True)
class QueueTransitionRequest:
    target_state: str
    ids: list[int]
    actor: str


class ExternalCompactionQueueAdapter:
    """Small adapter boundary around queue operations used by CLI handlers."""

    def __init__(self, service: ExternalCompactionQueueService) -> None:
        self._service = service

    def list_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._service.list_queue(states=states, limit=limit)

    def transition(
        self,
        *,
        target_state: str,
        ids: list[int],
        actor: str,
    ) -> dict[str, int]:
        if target_state == "approved":
            return self._service.approve_queue(ids=ids, actor=actor)
        if target_state == "rejected":
            return self._service.reject_queue(ids=ids, actor=actor)
        raise ValueError(f"Unsupported queue transition: {target_state}")


def load_queue_rows(
    adapter: ExternalCompactionQueueAdapter,
    *,
    request: QueueListRequest,
    resolve_attribution: Callable[[list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    rows = adapter.list_queue(
        states=[item for item in (request.states or []) if item] or None,
        limit=request.limit,
    )
    if not request.with_attribution:
        return rows

    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        source_ids = [
            str(item) for item in row_copy.get("source_session_ids", []) if str(item).strip()
        ]
        attribution = "-"
        if callable(resolve_attribution):
            attribution = str(resolve_attribution(source_ids))
        row_copy["attribution"] = attribution
        resolved_rows.append(row_copy)
    return resolved_rows


def transition_queue_rows(
    adapter: ExternalCompactionQueueAdapter,
    *,
    request: QueueTransitionRequest,
) -> dict[str, int]:
    if not request.ids:
        raise ValueError("Provide at least one --id value.")
    return adapter.transition(
        target_state=request.target_state,
        ids=request.ids,
        actor=request.actor,
    )


def render_queue_table(rows: list[dict[str, Any]], *, with_attribution: bool) -> Table:
    table = Table(title="External Compaction Review Queue", box=box.SIMPLE)
    table.add_column("ID", justify="right")
    table.add_column("State")
    table.add_column("Tier")
    table.add_column("Line", overflow="fold")
    table.add_column("Source Sessions", overflow="fold")
    if with_attribution:
        table.add_column("Attribution", overflow="fold")
    table.add_column("Actor")
    table.add_column("Timestamp", overflow="fold")

    for row in rows:
        source_sessions = ", ".join(str(item) for item in row.get("source_session_ids", []))
        row_values = [
            str(row.get("id", "")),
            str(row.get("state", "")),
            str(row.get("tier", "")),
            str(row.get("line", "")),
            source_sessions,
        ]
        if with_attribution:
            row_values.append(str(row.get("attribution", "-")))
        row_values.extend(
            [
                str(row.get("actor", "")),
                str(row.get("timestamp", "")),
            ]
        )
        table.add_row(*row_values)

    return table
