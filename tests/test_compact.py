from __future__ import annotations

import pytest

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.log import LogWriter
from agent_recall.core.session import SessionManager
from agent_recall.llm.base import LLMProvider, LLMResponse, Message
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import CurationStatus, SemanticLabel


class FakeLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (temperature, max_tokens)
        prompt = messages[-1].content
        if "Current GUARDRAILS.md" in prompt:
            return LLMResponse(
                content="- [GOTCHA] Verify lock ordering to avoid deadlocks",
                model="fake",
            )
        if "Current STYLE.md" in prompt:
            return LLMResponse(
                content="- [PATTERN] Use repository pattern in service layer",
                model="fake",
            )
        return LLMResponse(
            content="**2026-02-09**: Finished feature and validated behavior.",
            model="fake",
        )

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


@pytest.mark.asyncio
async def test_compaction_updates_tiers_and_indexes(storage, files) -> None:
    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)

    session = session_mgr.start("Implement auth")
    log_writer.log(content="Avoid weak password hashing rounds", label=SemanticLabel.GOTCHA)
    log_writer.log(content="Use DTO mapping at API boundaries", label=SemanticLabel.PATTERN)
    session_mgr.end(session.id, "Completed auth flow")

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert results["guardrails_updated"] is True
    assert results["style_updated"] is True
    assert results["recent_updated"] is True
    assert int(results["chunks_indexed"]) == 2

    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)

    assert "[GOTCHA] Verify lock ordering" in guardrails
    assert "[PATTERN] Use repository pattern" in style
    assert "2026-02-09" in recent


@pytest.mark.asyncio
async def test_compaction_indexes_decision_and_exploration_by_default(storage, files) -> None:
    log_writer = LogWriter(storage)
    log_writer.log(
        content="Chose event-sourcing boundaries for audit consistency",
        label=SemanticLabel.DECISION_RATIONALE,
    )
    log_writer.log(
        content="Explored optimistic locking but write contention remained high",
        label=SemanticLabel.EXPLORATION,
    )
    log_writer.log(
        content="General retrospective notes from the session",
        label=SemanticLabel.NARRATIVE,
    )

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 2

    decision_chunks = storage.search_chunks_fts("audit consistency", top_k=5)
    exploration_chunks = storage.search_chunks_fts("write contention", top_k=5)
    narrative_chunks = storage.search_chunks_fts("retrospective notes", top_k=5)

    assert any(chunk.label == SemanticLabel.DECISION_RATIONALE for chunk in decision_chunks)
    assert any(chunk.label == SemanticLabel.EXPLORATION for chunk in exploration_chunks)
    assert all(chunk.label != SemanticLabel.NARRATIVE for chunk in narrative_chunks)


@pytest.mark.asyncio
async def test_compaction_filters_non_style_indexing_by_confidence_thresholds(
    storage,
    files,
) -> None:
    files.write_config(
        {
            "compaction": {
                "index_decision_entries": True,
                "index_decision_min_confidence": 0.9,
                "index_exploration_entries": True,
                "index_exploration_min_confidence": 0.85,
            }
        }
    )

    log_writer = LogWriter(storage)
    log_writer.log(
        content="Decision confidence is too low for indexing",
        label=SemanticLabel.DECISION_RATIONALE,
        confidence=0.8,
    )
    log_writer.log(
        content="Exploration confidence is high enough for indexing",
        label=SemanticLabel.EXPLORATION,
        confidence=0.9,
    )
    log_writer.log(
        content="Exploration confidence is too low for indexing",
        label=SemanticLabel.EXPLORATION,
        confidence=0.6,
    )

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 1
    assert storage.has_chunk(
        "Exploration confidence is high enough for indexing",
        SemanticLabel.EXPLORATION,
    )
    assert not storage.has_chunk(
        "Decision confidence is too low for indexing",
        SemanticLabel.DECISION_RATIONALE,
    )
    assert not storage.has_chunk(
        "Exploration confidence is too low for indexing",
        SemanticLabel.EXPLORATION,
    )


@pytest.mark.asyncio
async def test_compaction_generates_embeddings_when_enabled(storage, files) -> None:
    files.write_config(
        {
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "retrieval": {"embedding_enabled": True, "embedding_dimensions": 16},
        }
    )

    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)
    session = session_mgr.start("Improve auth")
    log_writer.log(content="Avoid weak password hashing rounds", label=SemanticLabel.GOTCHA)
    session_mgr.end(session.id, "Completed")

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    await engine.compact(force=True)

    chunks = storage.search_chunks_fts("password", top_k=5)
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding or []) == 16


@pytest.mark.asyncio
async def test_compaction_ignores_pending_entries(storage, files) -> None:
    log_writer = LogWriter(storage)
    log_writer.log(content="Approved entry", label=SemanticLabel.GOTCHA)
    pending_entry = log_writer.log(
        content="Pending entry",
        label=SemanticLabel.GOTCHA,
    )
    storage.update_entry_curation_status(pending_entry.id, CurationStatus.PENDING)

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 1
    assert storage.has_chunk("Approved entry", SemanticLabel.GOTCHA)
    assert not storage.has_chunk("Pending entry", SemanticLabel.GOTCHA)


@pytest.mark.asyncio
async def test_compaction_can_target_pending_entries(storage, files) -> None:
    files.write_config({"compaction": {"curation_status": "pending"}})

    log_writer = LogWriter(storage)
    pending_entry = log_writer.log(
        content="Pending entry",
        label=SemanticLabel.GOTCHA,
    )
    storage.update_entry_curation_status(pending_entry.id, CurationStatus.PENDING)

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 1
    assert storage.has_chunk("Pending entry", SemanticLabel.GOTCHA)


def test_curation_status_transitions_update_list(storage) -> None:
    log_writer = LogWriter(storage)
    entry = log_writer.log(
        content="Lifecycle entry",
        label=SemanticLabel.GOTCHA,
    )

    storage.update_entry_curation_status(entry.id, CurationStatus.PENDING)
    pending = storage.list_entries_by_curation_status(CurationStatus.PENDING)
    assert [item.id for item in pending] == [entry.id]

    storage.update_entry_curation_status(entry.id, CurationStatus.REJECTED)
    assert storage.list_entries_by_curation_status(CurationStatus.PENDING) == []
    rejected = storage.list_entries_by_curation_status(CurationStatus.REJECTED)
    assert [item.id for item in rejected] == [entry.id]

    storage.update_entry_curation_status(entry.id, CurationStatus.APPROVED)
    assert storage.list_entries_by_curation_status(CurationStatus.REJECTED) == []
    approved = storage.list_entries_by_curation_status(CurationStatus.APPROVED)
    assert [item.id for item in approved] == [entry.id]


@pytest.mark.asyncio
async def test_compaction_promotes_only_approved_entries(storage, files) -> None:
    log_writer = LogWriter(storage)
    entry = log_writer.log(content="Curation gating", label=SemanticLabel.GOTCHA)
    storage.update_entry_curation_status(entry.id, CurationStatus.PENDING)
    storage.update_entry_curation_status(entry.id, CurationStatus.REJECTED)

    engine = CompactionEngine(storage=storage, files=files, llm=FakeLLMProvider())
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 0
    assert not storage.has_chunk("Curation gating", SemanticLabel.GOTCHA)

    storage.update_entry_curation_status(entry.id, CurationStatus.APPROVED)
    results = await engine.compact(force=True)

    assert int(results["chunks_indexed"]) == 1
    assert storage.has_chunk("Curation gating", SemanticLabel.GOTCHA)
