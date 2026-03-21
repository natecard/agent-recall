from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_recall.core.rule_confidence import decay_confidence, reinforce_confidence, snapshot_rules
from agent_recall.storage.files import KnowledgeTier


def test_rule_confidence_model_helpers() -> None:
    reinforced = reinforce_confidence(0.5, reinforcement_factor=0.2)
    decayed = decay_confidence(1.0, elapsed_days=45, half_life_days=45)
    assert reinforced > 0.5
    assert 0.49 <= decayed <= 0.51


def test_snapshot_rules_extracts_bullets(files) -> None:
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- Keep tests deterministic\n- Keep tests deterministic\n",
    )
    files.write_tier(KnowledgeTier.STYLE, "# Style\n\n- Prefer small commits\n")

    rows = snapshot_rules(files)
    assert len(rows) == 2
    assert {row["tier"] for row in rows} == {"GUARDRAILS", "STYLE"}


def test_storage_rule_confidence_sync_decay_and_prune(storage) -> None:
    rules = [
        {"rule_id": "rule-1", "tier": "GUARDRAILS", "line": "Keep tests deterministic"},
        {"rule_id": "rule-2", "tier": "STYLE", "line": "Prefer small commits"},
    ]
    first = storage.sync_rule_confidence(rules)
    assert first["inserted"] == 2
    second = storage.sync_rule_confidence([rules[0]])
    assert second["updated"] == 1

    stale_anchor = (datetime.now(UTC) - timedelta(days=120)).isoformat()
    with storage._connect() as conn:
        conn.execute(
            """
            UPDATE rule_confidence
            SET last_reinforced_at = ?, confidence = ?
            WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
            """,
            (stale_anchor, 0.2, "rule-1", storage.tenant_id, storage.project_id),
        )

    decay_result = storage.decay_rule_confidence(half_life_days=30, stale_after_days=60)
    assert decay_result["decayed"] >= 1

    stale_rows = storage.list_rule_confidence(stale_only=True)
    assert any(row["rule_id"] == "rule-1" for row in stale_rows)

    dry_run = storage.archive_and_prune_rule_confidence(
        max_confidence=0.5,
        stale_only=True,
        dry_run=True,
    )
    assert any(row["rule_id"] == "rule-1" for row in dry_run)

    committed = storage.archive_and_prune_rule_confidence(
        max_confidence=0.5,
        stale_only=True,
        dry_run=False,
    )
    assert any(row["rule_id"] == "rule-1" for row in committed)
    remaining = storage.list_rule_confidence(limit=10)
    assert all(row["rule_id"] != "rule-1" for row in remaining)
