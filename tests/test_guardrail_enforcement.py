from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_recall.core.guardrail_enforcement import (
    GuardrailSuppressionStore,
    evaluate_guardrail_text,
    parse_guardrail_rules,
)


def test_parse_guardrail_rules_extracts_severity_and_patterns() -> None:
    rules = parse_guardrail_rules(
        """
# Guardrails
- [BLOCK] regex: rm\\s+-rf\\s+/
- [WARN] `force push`
- [GOTCHA] Avoid unsafe fallback scripts.
"""
    )
    block_rules = [rule for rule in rules if rule.severity == "block"]
    warn_rules = [rule for rule in rules if rule.severity == "warn"]

    assert len(block_rules) == 1
    assert block_rules[0].pattern == r"rm\s+-rf\s+/"
    assert len(warn_rules) >= 2
    assert any("force\\ push" in rule.pattern for rule in warn_rules)


def test_guardrail_suppression_store_honors_expiry(tmp_path: Path) -> None:
    rules = parse_guardrail_rules("- [BLOCK] regex: drop\\s+table")
    rule = rules[0]
    store = GuardrailSuppressionStore(tmp_path / "guardrail_suppressions.json")

    active = store.add(
        rule_id=rule.rule_id,
        reason="migration dry-run in disposable database",
        actor="tester",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    assert active.actor == "tester"
    assert store.is_suppressed(rule.rule_id) is True

    store.add(
        rule_id=rule.rule_id,
        reason="expired override",
        actor="tester",
        expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    )
    assert store.is_suppressed(rule.rule_id) is False


def test_guardrail_evaluation_skips_suppressed_rules(tmp_path: Path) -> None:
    rules = parse_guardrail_rules("- [BLOCK] regex: delete\\s+production")
    rule = rules[0]
    store = GuardrailSuppressionStore(tmp_path / "guardrail_suppressions.json")
    store.add(
        rule_id=rule.rule_id,
        reason="approved emergency rollback runbook",
        actor="oncall",
    )

    violations = evaluate_guardrail_text(
        "delete production table now",
        rules,
        suppression_store=store,
    )
    assert violations == []
