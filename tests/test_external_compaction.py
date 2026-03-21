from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agent_recall.external_compaction.service import ExternalCompactionService
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import LogEntry, LogSource, SemanticLabel


def test_external_compaction_service_builds_payload_and_applies_notes(storage, files) -> None:
    source_session_id = "codex-session-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Need safer migration rollback handling",
            label=SemanticLabel.GOTCHA,
        )
    )
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Prefer idempotent migration scripts",
            label=SemanticLabel.PATTERN,
        )
    )

    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    queue = service.list_imported_conversations(limit=10, pending_only=True)
    assert any(item["source_session_id"] == source_session_id for item in queue)

    payload = service.build_payload(
        limit=10,
        pending_only=True,
        entry_limit=50,
        write_target="runtime",
    )
    conversation_ids = [item["source_session_id"] for item in payload["conversations"]]
    assert source_session_id in conversation_ids

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Keep rollback paths idempotent for migrations.",
                    "source_session_ids": [source_session_id],
                },
                {
                    "tier": "STYLE",
                    "line": "- [PATTERN] Prefer reversible migrations with explicit validation.",
                    "source_session_ids": [source_session_id],
                },
            ]
        },
        write_target="runtime",
        dry_run=False,
        mark_processed=True,
    )

    assert result["notes_applied"] == 2
    assert result["sessions_marked"] == 1
    assert "GUARDRAILS" in result["tiers_changed"]
    assert "STYLE" in result["tiers_changed"]

    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    assert "Keep rollback paths idempotent for migrations." in guardrails
    assert "Prefer reversible migrations with explicit validation." in style

    queue_after = service.list_imported_conversations(limit=10, pending_only=True)
    assert all(item["source_session_id"] != source_session_id for item in queue_after)

    events_path = files.agent_dir / "metrics" / "pipeline-events.jsonl"
    assert events_path.exists()
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("stage") == "apply"
        and event.get("action") == "complete"
        and event.get("success") is True
        for event in events
    )


def test_external_compaction_service_dry_run_does_not_write(storage, files) -> None:
    source_session_id = "cursor-session-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Use deterministic sort for migrations list",
            label=SemanticLabel.PATTERN,
        )
    )

    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )
    before = files.read_tier(KnowledgeTier.GUARDRAILS)

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Sort migration files before execution.",
                    "source_session_ids": [source_session_id],
                }
            ]
        },
        dry_run=True,
    )

    after = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert result["dry_run"] is True
    assert result["notes_applied"] == 1
    assert before == after


def test_external_compaction_service_backfills_json_state_into_sqlite(storage, files) -> None:
    state_path = files.agent_dir / "external_compaction_state.json"
    state_path.write_text(
        json.dumps(
            {
                "sessions": {
                    "legacy-session-1": {
                        "source_hash": "abc123",
                        "processed_at": "2026-03-21T00:00:00+00:00",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
        state_path=state_path,
    )

    list_state = getattr(storage, "list_external_compaction_states")
    rows = list_state()
    assert any(row["source_session_id"] == "legacy-session-1" for row in rows)


def test_external_compaction_cleanup_state_removes_stale_rows(storage, files) -> None:
    valid_session_id = "valid-session-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=valid_session_id,
            content="Use explicit lock ordering in migrations",
            label=SemanticLabel.GOTCHA,
        )
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    upsert_state = getattr(storage, "upsert_external_compaction_state")
    now = datetime.now(UTC).isoformat()
    upsert_state(
        valid_session_id,
        source_hash=service._hash_source_session(valid_session_id),
        processed_at=now,
    )
    upsert_state(
        "stale-session-1",
        source_hash="deadbeef",
        processed_at=now,
    )

    result = service.cleanup_state()
    assert result["removed_stale"] >= 1
    rows = getattr(storage, "list_external_compaction_states")()
    ids = {row["source_session_id"] for row in rows}
    assert "stale-session-1" not in ids
    assert valid_session_id in ids
