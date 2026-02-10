from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.core.sync import AutoSync
from agent_recall.ingest.base import RawMessage, RawSession, SessionIngester
from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class AdaptiveLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "adaptive"

    @property
    def model_name(self) -> str:
        return "mock"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (temperature, max_tokens)
        prompt = messages[-1].content

        if "Analyze this development session transcript" in prompt:
            return LLMResponse(
                content=(
                    "[{\"label\": \"pattern\", \"content\": "
                    "\"Use explicit transactions around writes\", "
                    "\"tags\": [\"db\"], \"confidence\": 0.8}]"
                ),
                model="mock",
            )

        if "Current STYLE.md" in prompt or "Current GUARDRAILS.md" in prompt:
            return LLMResponse(content="NONE", model="mock")

        return LLMResponse(content="", model="mock")

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


class FailingLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str:
        return "failing-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        raise RuntimeError("provider unavailable")

    def validate(self) -> tuple[bool, str]:
        return False, "provider unavailable"


class SlowLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "slow"

    @property
    def model_name(self) -> str:
        return "slow-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        await asyncio.sleep(0.2)
        return LLMResponse(content="[]", model="slow")

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


class FakeIngester(SessionIngester):
    def __init__(self, source_name: str, sessions: list[Path]):
        self._source_name = source_name
        self._sessions = sessions

    @property
    def source_name(self) -> str:
        return self._source_name

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        if since is None:
            return self._sessions

        normalized_since = (
            since.replace(tzinfo=UTC) if since.tzinfo is None else since.astimezone(UTC)
        )
        kept: list[Path] = []
        for session in self._sessions:
            mtime = datetime.fromtimestamp(session.stat().st_mtime, tz=UTC)
            if mtime >= normalized_since:
                kept.append(session)
        return kept

    def parse_session(self, path: Path) -> RawSession:
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            project_path=path.parent,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            messages=[
                RawMessage(
                    role="user",
                    content=(
                        "Please fix the migration ordering and make sure rollbacks are safe "
                        "for partial writes."
                    ),
                ),
                RawMessage(
                    role="assistant",
                    content=(
                        "I wrapped writes in explicit transactions, added rollback behavior, "
                        "and validated the migration sequence."
                    ),
                ),
            ],
        )

    def get_session_id(self, path: Path) -> str:
        return f"{self.source_name}-{path.stem}"


@pytest.mark.asyncio
async def test_auto_sync_processes_then_skips_duplicates(storage, files, tmp_path: Path) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    first = await sync.sync()
    assert first["sessions_discovered"] == 1
    assert first["sessions_processed"] == 1
    assert first["learnings_extracted"] == 1

    second = await sync.sync()
    assert second["sessions_discovered"] == 1
    assert second["sessions_processed"] == 0
    assert second["sessions_skipped"] == 1


@pytest.mark.asyncio
async def test_auto_sync_filters_by_source(storage, files, tmp_path: Path) -> None:
    cursor_session = tmp_path / "cursor-session"
    claude_session = tmp_path / "claude-session"
    cursor_session.write_text("cursor")
    claude_session.write_text("claude")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[
            FakeIngester("cursor", [cursor_session]),
            FakeIngester("claude-code", [claude_session]),
        ],
    )

    results = await sync.sync(sources=["cursor"])
    by_source = results["by_source"]

    assert "cursor" in by_source
    assert "claude-code" not in by_source
    assert results["sessions_processed"] == 1


@pytest.mark.asyncio
async def test_sync_and_compact_includes_compaction_results(storage, files, tmp_path: Path) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync_and_compact()

    assert "compaction" in results
    assert results["sessions_processed"] == 1


@pytest.mark.asyncio
async def test_auto_sync_since_filter(storage, files, tmp_path: Path) -> None:
    old_session = tmp_path / "old-session"
    new_session = tmp_path / "new-session"
    old_session.write_text("old")
    new_session.write_text("new")

    now = datetime.now(UTC).timestamp()
    old_time = now - 86_400
    new_time = now

    import os

    os.utime(old_session, (old_time, old_time))
    os.utime(new_session, (new_time, new_time))

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [old_session, new_session])],
    )

    results = await sync.sync(since=datetime.fromtimestamp(now - 60, tz=UTC))
    assert results["sessions_discovered"] == 1
    assert results["sessions_processed"] == 1


@pytest.mark.asyncio
async def test_auto_sync_marks_processed_when_extraction_fails(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=FailingLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    first = await sync.sync()
    assert first["sessions_discovered"] == 1
    assert first["sessions_processed"] == 1
    assert first["learnings_extracted"] == 0
    assert len(first["errors"]) == 1
    assert "extraction failed" in first["errors"][0]

    second = await sync.sync()
    assert second["sessions_processed"] == 0
    assert second["sessions_skipped"] == 1


@pytest.mark.asyncio
async def test_auto_sync_timeout_is_reported(storage, files, tmp_path: Path) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=SlowLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )
    sync.extract_timeout_seconds = 0.01

    results = await sync.sync()
    assert results["sessions_processed"] == 1
    assert results["learnings_extracted"] == 0
    assert len(results["errors"]) == 1
    assert "timed out" in results["errors"][0]
