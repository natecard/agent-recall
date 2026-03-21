from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

GuardrailSeverity = Literal["warn", "block"]
_RULE_LINE_RE = re.compile(r"^\s*-\s*(?:\[(?P<tag>[A-Z_]+)\])?\s*(?P<body>.+)\s*$")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class GuardrailRule:
    rule_id: str
    severity: GuardrailSeverity
    pattern: str
    description: str
    source_line: str


@dataclass(frozen=True)
class GuardrailViolation:
    rule_id: str
    severity: GuardrailSeverity
    pattern: str
    description: str
    matched_text: str


@dataclass(frozen=True)
class GuardrailSuppression:
    rule_id: str
    reason: str
    actor: str
    created_at: str
    expires_at: str | None = None

    def is_active(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return True
        try:
            expires = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        effective_now = now or datetime.now(UTC)
        if effective_now.tzinfo is None:
            effective_now = effective_now.replace(tzinfo=UTC)
        return expires > effective_now

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "reason": self.reason,
            "actor": self.actor,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GuardrailSuppression | None:
        if not isinstance(payload, dict):
            return None
        rule_id = str(payload.get("rule_id", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        actor = str(payload.get("actor", "")).strip() or "system"
        created_at = str(payload.get("created_at", "")).strip()
        expires_at_raw = payload.get("expires_at")
        expires_at = str(expires_at_raw).strip() if expires_at_raw else None
        if not rule_id or not reason:
            return None
        if not created_at:
            created_at = datetime.now(UTC).isoformat()
        return cls(
            rule_id=rule_id,
            reason=reason,
            actor=actor,
            created_at=created_at,
            expires_at=expires_at or None,
        )


def parse_guardrail_rules(guardrails_text: str) -> list[GuardrailRule]:
    rules: list[GuardrailRule] = []
    seen_ids: set[str] = set()

    for raw_line in guardrails_text.splitlines():
        match = _RULE_LINE_RE.match(raw_line)
        if not match:
            continue
        tag = str(match.group("tag") or "").strip().upper()
        body = str(match.group("body") or "").strip()
        if not body:
            continue

        severity: GuardrailSeverity = "block" if tag == "BLOCK" else "warn"
        patterns = _extract_patterns(body)
        for pattern in patterns:
            if not pattern:
                continue
            rule_id = _rule_id_for(severity, pattern)
            if rule_id in seen_ids:
                continue
            seen_ids.add(rule_id)
            rules.append(
                GuardrailRule(
                    rule_id=rule_id,
                    severity=severity,
                    pattern=pattern,
                    description=body,
                    source_line=raw_line,
                )
            )

    return rules


def evaluate_guardrail_text(
    text: str,
    rules: list[GuardrailRule],
    *,
    suppression_store: GuardrailSuppressionStore | None = None,
) -> list[GuardrailViolation]:
    violations: list[GuardrailViolation] = []
    if not text.strip():
        return violations

    for rule in rules:
        if suppression_store and suppression_store.is_suppressed(rule.rule_id):
            continue
        try:
            match = re.search(rule.pattern, text, re.IGNORECASE)
        except re.error:
            continue
        if not match:
            continue
        violations.append(
            GuardrailViolation(
                rule_id=rule.rule_id,
                severity=rule.severity,
                pattern=rule.pattern,
                description=rule.description,
                matched_text=match.group(0),
            )
        )

    return violations


def is_guardrail_enforcement_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    guardrails_cfg = config.get("guardrails")
    if not isinstance(guardrails_cfg, dict):
        return False
    enforcement_cfg = guardrails_cfg.get("enforcement")
    if isinstance(enforcement_cfg, dict):
        return bool(enforcement_cfg.get("enabled", False))
    if isinstance(guardrails_cfg.get("enforcement_enabled"), bool):
        return bool(guardrails_cfg["enforcement_enabled"])
    return False


class GuardrailSuppressionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_suppressions(self) -> list[GuardrailSuppression]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = payload.get("suppressions") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        suppressions: list[GuardrailSuppression] = []
        for item in items:
            suppression = GuardrailSuppression.from_dict(item)
            if suppression is None:
                continue
            suppressions.append(suppression)
        return suppressions

    def is_suppressed(self, rule_id: str, *, now: datetime | None = None) -> bool:
        target = str(rule_id).strip()
        if not target:
            return False
        for suppression in self.list_suppressions():
            if suppression.rule_id != target:
                continue
            if suppression.is_active(now=now):
                return True
        return False

    def add(
        self,
        *,
        rule_id: str,
        reason: str,
        actor: str = "system",
        expires_at: str | None = None,
    ) -> GuardrailSuppression:
        normalized_rule_id = str(rule_id).strip()
        normalized_reason = str(reason).strip()
        if not normalized_rule_id:
            raise ValueError("rule_id is required")
        if not normalized_reason:
            raise ValueError("reason is required")

        created_at = datetime.now(UTC).isoformat()
        suppression = GuardrailSuppression(
            rule_id=normalized_rule_id,
            reason=normalized_reason,
            actor=str(actor).strip() or "system",
            created_at=created_at,
            expires_at=str(expires_at).strip() if expires_at else None,
        )
        suppressions = self.list_suppressions()
        by_rule = {item.rule_id: item for item in suppressions}
        by_rule[normalized_rule_id] = suppression
        updated = [item.to_dict() for item in by_rule.values()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"suppressions": updated}, indent=2),
            encoding="utf-8",
        )
        return suppression


def _extract_patterns(body: str) -> list[str]:
    regex_match = re.match(r"(?i)^regex:\s*(.+)$", body)
    if regex_match:
        pattern = regex_match.group(1).strip()
        return [pattern] if pattern else []

    literal_patterns = [match.strip() for match in _BACKTICK_RE.findall(body) if match.strip()]
    if literal_patterns:
        return [re.escape(item) for item in literal_patterns]

    return [re.escape(body.strip())]


def _rule_id_for(severity: GuardrailSeverity, pattern: str) -> str:
    digest = sha256(f"{severity}|{pattern}".encode()).hexdigest()
    return f"gr-{digest[:12]}"
