from __future__ import annotations

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.tier_notes import (
    normalize_tier_content,
    normalize_tier_line,
    polarity,
    semantic_key,
    topic_key,
)
from agent_recall.core.tier_writer import _normalize_for_dedup
from agent_recall.external_compaction import service as external_service


def test_tier_note_normalization_parity_across_modules() -> None:
    text = "  Keep   Validation  Green  "
    line = "- [PATTERN] Keep   Validation  Green"
    assert _normalize_for_dedup(text) == normalize_tier_content(text)
    assert CompactionEngine._normalize_content(text) == normalize_tier_content(text)
    assert CompactionEngine._normalize_line(line) == normalize_tier_line(line)


def test_external_compaction_semantic_helpers_match_shared_utility() -> None:
    line = "- [GOTCHA] Sort migration inputs before execution."
    assert external_service._semantic_key(line) == semantic_key(line)
    assert external_service._topic_key(line) == topic_key(line)
    assert external_service._polarity("Never auto-apply migrations") == polarity(
        "Never auto-apply migrations"
    )
