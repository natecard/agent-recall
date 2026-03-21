from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.core.extract import TranscriptExtractor
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.ingest.base import RawMessage, RawSession
from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class _AttributionLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock-provider"

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        return LLMResponse(
            content=(
                '[{"label":"pattern","content":"Attribution pattern","tags":[],"confidence":0.8}]'
            ),
            model="mock-model",
        )

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


@pytest.mark.asyncio
async def test_extractor_writes_attribution_metadata() -> None:
    extractor = TranscriptExtractor(_AttributionLLM())
    session = RawSession(
        source="cursor",
        session_id="cursor-session-1",
        started_at=datetime.now(UTC),
        messages=[
            RawMessage(
                role="user",
                content=(
                    "Investigate retries in API client with enough transcript detail "
                    "to trigger extraction and retain attribution metadata in the result."
                ),
            ),
            RawMessage(
                role="assistant",
                content=(
                    "Added deterministic retries, timeout guards, and fallback behavior "
                    "for long-running API calls with explicit validation notes."
                ),
            ),
        ],
    )
    entries = await extractor.extract(session)
    assert len(entries) == 1
    metadata = entries[0].metadata
    assert metadata["attribution"]["agent_source"] == "cursor"
    assert metadata["attribution"]["provider"] == "mock-provider"
    assert metadata["attribution"]["model"] == "mock-model"


def test_transcript_ingestor_adds_attribution_from_payload(storage, tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        '{"content":"Imported message","source":"codex","provider":"openai","model":"gpt-5"}\n',
        encoding="utf-8",
    )
    ingestor = TranscriptIngestor(storage)
    count = ingestor.ingest_jsonl(transcript, source_session_id="codex-123")
    assert count == 1

    entries = storage.get_entries_by_source_session("codex-123", limit=10)
    assert len(entries) == 1
    attribution = entries[0].metadata["attribution"]
    assert attribution["agent_source"] == "codex"
    assert attribution["provider"] == "openai"
    assert attribution["model"] == "gpt-5"
