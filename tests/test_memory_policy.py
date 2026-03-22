from __future__ import annotations

from agent_recall.memory.policy import MemoryPolicy, normalize_memory_rows


def test_memory_policy_from_memory_config_parses_defaults_and_overrides() -> None:
    policy = MemoryPolicy.from_memory_config(
        {
            "cost": {"max_vector_records": 12},
            "privacy": {
                "retention_days": 14,
                "redaction_patterns": ["secret", "["],  # invalid regex is ignored
            },
        }
    )
    assert policy.max_vector_records == 12
    assert policy.retention_days == 14
    assert len(policy.redaction_patterns) == 1


def test_normalize_memory_rows_applies_redaction_dedupe_and_limit() -> None:
    policy = MemoryPolicy.from_memory_config(
        {
            "cost": {"max_vector_records": 2},
            "privacy": {"redaction_patterns": ["token-[0-9]+"]},
        }
    )
    rows = [
        {"id": "1", "text": "Use token-123 for auth"},
        {"id": "2", "text": "use TOKEN-999 for auth"},
        {"id": "3", "text": "Keep migration rollback safe"},
    ]

    normalized = normalize_memory_rows(rows, policy=policy)
    assert normalized.rows_discovered == 3
    assert normalized.redacted_rows == 2
    assert normalized.rows_deduplicated == 1
    assert normalized.rows_normalized == 2
    assert normalized.rows_capped == 0
    assert len(normalized.rows) == 2
    assert normalized.rows[0]["text"] == "Use [REDACTED] for auth"


def test_memory_policy_retention_days_uses_override_when_provided() -> None:
    policy = MemoryPolicy.from_memory_config({"privacy": {"retention_days": 30}})
    assert policy.resolve_retention_days() == 30
    assert policy.resolve_retention_days(override=7) == 7
