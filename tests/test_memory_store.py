from __future__ import annotations

from agent_recall.memory.store import MarkdownMemoryStore, infer_tier_from_label
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def test_markdown_memory_store_preserves_markdown_behavior(storage, files) -> None:
    store = MarkdownMemoryStore(storage, files)
    written = store.write_tier_note(
        tier=KnowledgeTier.GUARDRAILS,
        note_line="Keep tests deterministic",
    )
    assert written is True
    duplicate = store.write_tier_note(
        tier=KnowledgeTier.GUARDRAILS,
        note_line="- Keep tests deterministic",
    )
    assert duplicate is False
    spacing_duplicate = store.write_tier_note(
        tier=KnowledgeTier.GUARDRAILS,
        note_line="-   KEEP   tests     deterministic",
    )
    assert spacing_duplicate is False
    assert "Keep tests deterministic" in store.read_tier(KnowledgeTier.GUARDRAILS)


def test_markdown_memory_store_search_and_export(storage, files) -> None:
    store = MarkdownMemoryStore(storage, files)
    chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Use hybrid retrieval mode for semantic fallback",
        label=SemanticLabel.PATTERN,
        tags=["retrieval"],
    )
    storage.store_chunk(chunk)
    results = store.search(query="semantic fallback", top_k=5)
    assert any(item.id == chunk.id for item in results)

    snapshot = store.export_snapshot()
    assert "tiers" in snapshot
    assert "chunks" in snapshot


def test_infer_tier_from_label_maps_categories() -> None:
    assert infer_tier_from_label(SemanticLabel.GOTCHA) == KnowledgeTier.GUARDRAILS
    assert infer_tier_from_label(SemanticLabel.PREFERENCE) == KnowledgeTier.STYLE
    assert infer_tier_from_label(SemanticLabel.NARRATIVE) == KnowledgeTier.RECENT
