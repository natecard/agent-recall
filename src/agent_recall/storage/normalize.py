from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def normalize_non_empty_text(value: object) -> str | None:
    text = normalize_text(value)
    return text or None


def normalize_limit(value: object, *, minimum: int = 1, default: int = 1) -> int:
    try:
        if isinstance(value, (str, bytes, bytearray, int, float)):
            parsed = int(value)
        else:
            parsed = int(str(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), parsed)


def normalize_uuid_text(value: object) -> str | None:
    if isinstance(value, UUID):
        return str(value)
    text = normalize_non_empty_text(value)
    if not text:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        return None


def parse_iso_datetime(value: object) -> datetime | None:
    text = normalize_non_empty_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    raw = str(value)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def parse_json_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        source = value
    elif value is None:
        source = []
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            parsed = []
        source = parsed if isinstance(parsed, list) else []
    return [str(item).strip() for item in source if str(item).strip()]


def dump_json_compact(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"))
