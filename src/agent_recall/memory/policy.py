from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _as_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _coerce_int(value: object, *, default: int, minimum: int) -> int:
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            parsed = default
    else:
        parsed = default
    return max(minimum, parsed)


def compile_redaction_patterns(patterns: object) -> tuple[re.Pattern[str], ...]:
    if not isinstance(patterns, list):
        return ()
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        try:
            compiled.append(re.compile(text, flags=re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


@dataclass(frozen=True)
class MemoryPolicy:
    redaction_patterns: tuple[re.Pattern[str], ...]
    max_vector_records: int
    retention_days: int

    @classmethod
    def from_memory_config(cls, memory_cfg: object) -> MemoryPolicy:
        cfg = _as_mapping(memory_cfg)
        privacy_cfg = _as_mapping(cfg.get("privacy"))
        cost_cfg = _as_mapping(cfg.get("cost"))
        return cls(
            redaction_patterns=compile_redaction_patterns(
                privacy_cfg.get("redaction_patterns", [])
            ),
            max_vector_records=_coerce_int(
                cost_cfg.get("max_vector_records", 20_000),
                default=20_000,
                minimum=1,
            ),
            retention_days=_coerce_int(
                privacy_cfg.get("retention_days", 90),
                default=90,
                minimum=1,
            ),
        )

    @staticmethod
    def normalize_text(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def dedupe_key(text: str) -> str:
        return " ".join(text.lower().split())

    def redact_text(self, text: str) -> tuple[str, bool]:
        redacted = text
        changed = False
        for pattern in self.redaction_patterns:
            redacted_next = pattern.sub("[REDACTED]", redacted)
            if redacted_next != redacted:
                changed = True
                redacted = redacted_next
        return redacted, changed

    def resolve_record_limit(self, override: int | None = None) -> int:
        if override is None:
            return self.max_vector_records
        return max(1, int(override))

    def resolve_retention_days(self, override: int | None = None) -> int:
        if override is None:
            return self.retention_days
        return max(1, int(override))


@dataclass(frozen=True)
class NormalizedRows:
    rows: list[dict[str, Any]]
    rows_discovered: int
    rows_normalized: int
    rows_capped: int
    rows_deduplicated: int
    redacted_rows: int


def normalize_memory_rows(
    rows: list[dict[str, Any]],
    *,
    policy: MemoryPolicy,
    limit_override: int | None = None,
) -> NormalizedRows:
    normalized_rows: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    redacted_rows = 0
    rows_deduplicated = 0

    for row in rows:
        text = MemoryPolicy.normalize_text(row.get("text", ""))
        if not text:
            continue
        redacted_text, changed = policy.redact_text(text)
        if changed:
            redacted_rows += 1
        dedupe_key = policy.dedupe_key(redacted_text)
        if dedupe_key in seen_text:
            rows_deduplicated += 1
            continue
        seen_text.add(dedupe_key)
        normalized = dict(row)
        normalized["text"] = redacted_text
        normalized_rows.append(normalized)

    limit = policy.resolve_record_limit(limit_override)
    capped_rows = normalized_rows[:limit]
    rows_capped = max(0, len(normalized_rows) - len(capped_rows))
    return NormalizedRows(
        rows=capped_rows,
        rows_discovered=len(rows),
        rows_normalized=len(normalized_rows),
        rows_capped=rows_capped,
        rows_deduplicated=rows_deduplicated,
        redacted_rows=redacted_rows,
    )
