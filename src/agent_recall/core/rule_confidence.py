from __future__ import annotations

import hashlib
import re
from typing import Any

from agent_recall.storage.files import FileStorage, KnowledgeTier

_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s*$")


def rule_id(tier: str, line: str) -> str:
    payload = f"{tier.strip().upper()}::{line.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def reinforce_confidence(
    current: float,
    *,
    reinforcement_factor: float = 0.15,
    steps: int = 1,
) -> float:
    baseline = max(0.0, min(1.0, float(current)))
    factor = max(0.0, min(1.0, float(reinforcement_factor)))
    amount = 1.0 - ((1.0 - factor) ** max(1, int(steps)))
    return max(0.0, min(1.0, baseline + ((1.0 - baseline) * amount)))


def decay_confidence(
    current: float,
    *,
    elapsed_days: float,
    half_life_days: float = 45.0,
) -> float:
    baseline = max(0.0, min(1.0, float(current)))
    life = max(1.0, float(half_life_days))
    elapsed = max(0.0, float(elapsed_days))
    multiplier = 0.5 ** (elapsed / life)
    return max(0.0, min(1.0, baseline * multiplier))


def extract_rule_lines(content: str) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in content.splitlines():
        match = _BULLET_RE.match(raw)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(value)
    return lines


def snapshot_rules(files: FileStorage) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in (KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE):
        content = files.read_tier(tier)
        for line in extract_rule_lines(content):
            rows.append(
                {
                    "rule_id": rule_id(tier.value, line),
                    "tier": tier.value,
                    "line": line,
                }
            )
    return rows
