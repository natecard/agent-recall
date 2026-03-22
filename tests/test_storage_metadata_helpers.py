from __future__ import annotations

from agent_recall.storage.metadata import (
    AttributionMetadata,
    EntryMetadata,
    FeedbackMetadata,
    attribution_fields,
    build_entry_metadata,
)


def test_attribution_metadata_parses_known_fields_and_preserves_unknown() -> None:
    parsed = AttributionMetadata.from_value(
        {
            "agent_source": " cursor ",
            "provider": "openai",
            "model": "gpt-5",
            "custom": {"team": "core"},
        }
    )
    assert parsed.agent_source == "cursor"
    assert parsed.provider == "openai"
    assert parsed.model == "gpt-5"
    assert parsed.extra == {"custom": {"team": "core"}}

    encoded = parsed.to_dict()
    assert encoded["agent_source"] == "cursor"
    assert encoded["provider"] == "openai"
    assert encoded["model"] == "gpt-5"
    assert encoded["custom"] == {"team": "core"}


def test_entry_metadata_is_permissive_for_invalid_input() -> None:
    parsed = EntryMetadata.from_value("not-json")
    assert parsed.attribution is None
    assert parsed.extra == {}


def test_build_entry_metadata_preserves_existing_unknown_keys() -> None:
    merged = build_entry_metadata(
        base={
            "source_tool": "cursor",
            "unknown": {"a": 1},
        },
        ingested_from="tests.jsonl",
    )
    assert merged["source_tool"] == "cursor"
    assert merged["ingested_from"] == "tests.jsonl"
    assert merged["unknown"] == {"a": 1}


def test_feedback_metadata_parses_and_keeps_unknown_keys() -> None:
    parsed = FeedbackMetadata.from_value(
        {
            "surface": "slash",
            "source_session_id": "session-123",
            "custom": 7,
        }
    )
    assert parsed.surface == "slash"
    assert parsed.source_session_id == "session-123"
    assert parsed.extra == {"custom": 7}
    assert parsed.to_dict()["custom"] == 7


def test_attribution_fields_applies_fallbacks() -> None:
    agent_source, provider, model = attribution_fields(
        metadata={"attribution": {"agent_source": "codex"}},
        fallback_agent_source="fallback-agent",
        fallback_provider="fallback-provider",
        fallback_model="fallback-model",
    )
    assert agent_source == "codex"
    assert provider == "fallback-provider"
    assert model == "fallback-model"
