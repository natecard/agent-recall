from __future__ import annotations

from uuid import UUID, uuid4

from agent_recall.storage.normalize import (
    dump_json_compact,
    normalize_limit,
    normalize_non_empty_text,
    normalize_text,
    normalize_uuid_text,
    parse_iso_datetime,
    parse_json_object,
    parse_json_string_list,
    utc_now_iso,
)


def test_normalize_text_helpers() -> None:
    assert normalize_text("  abc  ") == "abc"
    assert normalize_text(None, default="fallback") == "fallback"
    assert normalize_non_empty_text("  ") is None
    assert normalize_non_empty_text("  value ") == "value"


def test_normalize_limit_coerces_to_minimum() -> None:
    assert normalize_limit("5") == 5
    assert normalize_limit(0, minimum=1) == 1
    assert normalize_limit("nope", default=3) == 3


def test_normalize_uuid_text_supports_uuid_instances_and_strings() -> None:
    value = uuid4()
    assert normalize_uuid_text(value) == str(value)
    assert normalize_uuid_text(str(value)) == str(value)
    assert normalize_uuid_text("not-a-uuid") is None


def test_parse_iso_datetime_parses_valid_timestamp() -> None:
    now_text = utc_now_iso()
    parsed = parse_iso_datetime(now_text)
    assert parsed is not None
    assert parsed.isoformat() == now_text
    assert parse_iso_datetime("invalid") is None


def test_parse_json_helpers_are_permissive() -> None:
    assert parse_json_object({"a": 1}) == {"a": 1}
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object("[]") == {}

    assert parse_json_string_list(["a", " ", 2]) == ["a", "2"]
    assert parse_json_string_list('["x", " ", "y"]') == ["x", "y"]
    assert parse_json_string_list('{"not":"a-list"}') == []


def test_dump_json_compact_round_trip() -> None:
    payload = {"ids": [str(UUID(int=1)), str(UUID(int=2))]}
    encoded = dump_json_compact(payload)
    assert " " not in encoded
    assert parse_json_object(encoded)["ids"] == payload["ids"]
