from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.core.guardrail_enforcement import GuardrailSuppressionStore, parse_guardrail_rules
from agent_recall.external_compaction.models import external_notes_json_schema
from agent_recall.external_compaction.service import ExternalCompactionService
from agent_recall.external_compaction.write_guard import (
    ExternalWritePolicyError,
    ExternalWriteScopeGuard,
    WriteTarget,
)
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
    assert result["evidence_backlinks_written"] >= 2

    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    assert "Keep rollback paths idempotent for migrations." in guardrails
    assert "Prefer reversible migrations with explicit validation." in style

    evidence_rows = getattr(storage, "list_external_compaction_evidence")()
    assert any(
        row["tier"] == "GUARDRAILS" and row["source_session_id"] == source_session_id
        for row in evidence_rows
    )

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


def test_external_compaction_apply_partial_write_failure_does_not_mark_state(
    storage,
    files,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_session_id = "partial-failure-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Keep rollback checkpoints for every migration phase",
            label=SemanticLabel.GOTCHA,
        )
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    original_write = service._write_target_tier
    call_count = {"value": 0}

    def flaky_write(tier: KnowledgeTier, content: str, *, write_target: WriteTarget) -> None:
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise OSError("simulated disk write failure")
        original_write(tier, content, write_target=write_target)

    monkeypatch.setattr(service, "_write_target_tier", flaky_write)

    with pytest.raises(OSError, match="simulated disk write failure"):
        service.apply_notes_payload(
            {
                "notes": [
                    {
                        "tier": "GUARDRAILS",
                        "line": "- [GOTCHA] Validate migration inputs first.",
                        "source_session_ids": [source_session_id],
                    },
                    {
                        "tier": "STYLE",
                        "line": "- [PATTERN] Keep migration scripts idempotent.",
                        "source_session_ids": [source_session_id],
                    },
                ]
            },
            dry_run=False,
            mark_processed=True,
        )

    state_rows = getattr(storage, "list_external_compaction_states")()
    assert all(row["source_session_id"] != source_session_id for row in state_rows)


def test_external_compaction_merge_inserts_before_ralph_block(storage, files) -> None:
    source_session_id = "merge-order-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Capture migration ordering constraints",
            label=SemanticLabel.GOTCHA,
        )
    )
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        (
            "# Guardrails\n\n"
            "- [GOTCHA] Existing guardrail line.\n\n"
            "## 2026-03-21T00:00:00Z Iteration 1 (abc123)\n"
            "Prior Ralph synthesis block.\n"
        ),
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Insert before Ralph blocks.",
                    "source_session_ids": [source_session_id],
                }
            ]
        },
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 1
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert guardrails.index("Insert before Ralph blocks.") < guardrails.index(
        "## 2026-03-21T00:00:00Z Iteration 1 (abc123)"
    )


def test_external_compaction_semantic_dedupe_skips_equivalent_lines(storage, files) -> None:
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- [GOTCHA] Sort migration inputs before execution.\n",
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [PATTERN] sort migration inputs before execution",
                    "source_session_ids": ["session-1"],
                }
            ]
        },
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 0
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert guardrails.count("Sort migration inputs before execution.") == 1


def test_external_compaction_conflict_policy_prefer_newest_replaces_line(storage, files) -> None:
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- [GOTCHA] Always run migration scripts automatically.\n",
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Never run migration scripts automatically.",
                    "source_session_ids": ["session-1"],
                }
            ]
        },
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 1
    assert len(result["conflicts"]) == 1
    updated = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert "Never run migration scripts automatically." in updated
    assert "Always run migration scripts automatically." not in updated


def test_external_compaction_conflict_policy_queue_for_review_skips_write(storage, files) -> None:
    (files.agent_dir / "config.yaml").write_text(
        "compaction:\n  external:\n    conflict_policy: queue_for_review\n",
        encoding="utf-8",
    )
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- [GOTCHA] Always run migration scripts automatically.\n",
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Never run migration scripts automatically.",
                    "source_session_ids": ["session-1"],
                }
            ]
        },
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 0
    assert len(result["conflicts"]) == 1
    assert result["conflict_policy"] == "queue_for_review"
    unchanged = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert "Always run migration scripts automatically." in unchanged
    assert "Never run migration scripts automatically." not in unchanged


def test_external_compaction_guardrail_block_enforcement(storage, files) -> None:
    (files.agent_dir / "config.yaml").write_text(
        "guardrails:\n  enforcement:\n    enabled: true\n",
        encoding="utf-8",
    )
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- [BLOCK] regex: rm\\s+-rf\\s+/\n",
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    with pytest.raises(ValueError, match="Guardrail blocked external note apply"):
        service.apply_notes_payload(
            {
                "notes": [
                    {
                        "tier": "GUARDRAILS",
                        "line": "- [GOTCHA] Do not run rm -rf / during migrations.",
                        "source_session_ids": ["session-1"],
                    }
                ]
            },
            dry_run=False,
            mark_processed=False,
        )


def test_external_compaction_guardrail_suppression_allows_override(storage, files) -> None:
    (files.agent_dir / "config.yaml").write_text(
        "guardrails:\n  enforcement:\n    enabled: true\n",
        encoding="utf-8",
    )
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- [BLOCK] regex: rm\\s+-rf\\s+/\n",
    )
    rules = parse_guardrail_rules(files.read_tier(KnowledgeTier.GUARDRAILS))
    assert rules
    store = GuardrailSuppressionStore(files.agent_dir / "guardrail_suppressions.json")
    store.add(
        rule_id=rules[0].rule_id,
        reason="approved one-time recovery runbook",
        actor="reviewer",
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Do not run rm -rf / during migrations.",
                    "source_session_ids": ["session-1"],
                }
            ]
        },
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 1
    assert result["guardrail_blocks"] == []


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


def test_external_compaction_queue_flow_service(storage, files) -> None:
    source_session_id = "queue-flow-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Prefer deterministic migration ordering",
            label=SemanticLabel.GOTCHA,
        )
    )
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=Path.cwd(),
    )

    queued = service.queue_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Sort migration inputs before execution.",
                    "source_session_ids": [source_session_id],
                }
            ]
        },
        actor="tester",
    )
    assert queued["queued"] == 1
    queue_id = int(queued["items"][0]["id"])

    approve = service.approve_queue(ids=[queue_id], actor="reviewer")
    assert approve["updated"] == 1

    preview = service.patch_preview(states=["approved"])
    assert preview["notes_considered"] == 1
    assert "Sort migration inputs before execution." in preview["combined_diff"]

    applied = service.apply_approved_queue(actor="applier", dry_run=False)
    assert applied["queue_items_applied"] == 1
    assert applied["notes_applied"] >= 1
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    assert "Sort migration inputs before execution." in guardrails

    second = service.apply_approved_queue(actor="applier", dry_run=False)
    assert second["queue_items_considered"] == 0
    assert second["notes_applied"] == 0


def test_external_compaction_service_blocks_template_writes_without_policy_opt_in(
    storage,
    files,
    tmp_path: Path,
) -> None:
    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=tmp_path,
        write_guard=ExternalWriteScopeGuard(
            repo_root=tmp_path,
            allow_template_writes=False,
        ),
    )

    with pytest.raises(ExternalWritePolicyError, match="allow_template_writes"):
        service.read_tiers(write_target="templates")


def test_external_compaction_service_writes_templates_when_policy_enabled(
    storage,
    files,
    tmp_path: Path,
) -> None:
    source_session_id = "template-write-1"
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content="Prefer reversible migrations with dry-run checkpoints",
            label=SemanticLabel.GOTCHA,
        )
    )

    service = ExternalCompactionService(
        storage,
        files,
        agent_dir=files.agent_dir,
        repo_root=tmp_path,
        write_guard=ExternalWriteScopeGuard(
            repo_root=tmp_path,
            allow_template_writes=True,
        ),
    )

    result = service.apply_notes_payload(
        {
            "notes": [
                {
                    "tier": "GUARDRAILS",
                    "line": "- [GOTCHA] Always run migration dry-runs before commit.",
                    "source_session_ids": [source_session_id],
                }
            ]
        },
        write_target="templates",
        dry_run=False,
        mark_processed=False,
    )
    assert result["notes_applied"] == 1

    template_path = tmp_path / "src" / "agent_recall" / "templates" / "GUARDRAILS.md"
    assert template_path.exists()
    assert "Always run migration dry-runs before commit." in template_path.read_text(
        encoding="utf-8"
    )


def test_external_compaction_schema_artifact_matches_model() -> None:
    artifact = Path("docs/schemas/external-compaction-notes.schema.json")
    assert artifact.exists()
    loaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert loaded == external_notes_json_schema()
