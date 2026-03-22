from __future__ import annotations

from typing import Any

import pytest

from agent_recall.cli.commands.external_compaction_queue_flow import (
    ExternalCompactionQueueAdapter,
    QueueListRequest,
    QueueTransitionRequest,
    load_queue_rows,
    render_queue_table,
    transition_queue_rows,
)


class _FakeQueueService:
    def __init__(self) -> None:
        self.list_calls: list[tuple[list[str] | None, int]] = []
        self.approve_calls: list[tuple[list[int], str]] = []
        self.reject_calls: list[tuple[list[int], str]] = []
        self._rows: list[dict[str, Any]] = [
            {
                "id": 7,
                "state": "pending",
                "tier": "GUARDRAILS",
                "line": "- [GOTCHA] Keep queue behavior deterministic.",
                "source_session_ids": ["session-a"],
                "actor": "system",
                "timestamp": "2026-03-21T00:00:00+00:00",
            }
        ]

    def list_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.list_calls.append((states, limit))
        return list(self._rows)

    def approve_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]:
        self.approve_calls.append((list(ids), actor))
        return {"updated": len(ids), "skipped": 0}

    def reject_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]:
        self.reject_calls.append((list(ids), actor))
        return {"updated": len(ids), "skipped": 0}


def test_load_queue_rows_applies_optional_attribution() -> None:
    service = _FakeQueueService()
    adapter = ExternalCompactionQueueAdapter(service)
    rows = load_queue_rows(
        adapter,
        request=QueueListRequest(states=["pending"], limit=5, with_attribution=True),
        resolve_attribution=lambda source_ids: f"attr:{','.join(source_ids)}",
    )
    assert service.list_calls == [(["pending"], 5)]
    assert rows[0]["attribution"] == "attr:session-a"


def test_transition_queue_rows_routes_to_target_state() -> None:
    service = _FakeQueueService()
    adapter = ExternalCompactionQueueAdapter(service)
    approved = transition_queue_rows(
        adapter,
        request=QueueTransitionRequest(target_state="approved", ids=[1, 2], actor="alice"),
    )
    rejected = transition_queue_rows(
        adapter,
        request=QueueTransitionRequest(target_state="rejected", ids=[3], actor="bob"),
    )
    assert approved == {"updated": 2, "skipped": 0}
    assert rejected == {"updated": 1, "skipped": 0}
    assert service.approve_calls == [([1, 2], "alice")]
    assert service.reject_calls == [([3], "bob")]


def test_transition_queue_rows_requires_ids() -> None:
    service = _FakeQueueService()
    adapter = ExternalCompactionQueueAdapter(service)
    with pytest.raises(ValueError, match="Provide at least one --id value."):
        transition_queue_rows(
            adapter,
            request=QueueTransitionRequest(target_state="approved", ids=[], actor="system"),
        )


def test_render_queue_table_includes_attribution_column_when_requested() -> None:
    table = render_queue_table(
        [
            {
                "id": 1,
                "state": "pending",
                "tier": "STYLE",
                "line": "- [PATTERN] Prefer small helpers.",
                "source_session_ids": ["s1"],
                "actor": "system",
                "timestamp": "2026-03-21T00:00:00+00:00",
                "attribution": "cursor/openai:1",
            }
        ],
        with_attribution=True,
    )
    headers = [column.header for column in table.columns]
    assert "Attribution" in headers
