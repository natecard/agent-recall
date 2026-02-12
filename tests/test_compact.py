from __future__ import annotations

import pytest

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.log import LogWriter
from agent_recall.core.session import SessionManager
from agent_recall.llm.base import LLMProvider, LLMResponse, Message
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import SemanticLabel


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
