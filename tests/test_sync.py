from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.core.sync import AutoSync, SyncDiscoverStage, SyncFilterStage
from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.ingest.base import RawMessage, RawSession, SessionIngester
from agent_recall.llm.base import LLMProvider, LLMRateLimitError, LLMResponse, Message
from agent_recall.storage.models import Chunk, ChunkSource, CurationStatus, SemanticLabel


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
                    '[{"label": "pattern", "content": '
                    '"Use explicit transactions around writes", '
                    '"tags": ["db"], "confidence": 0.8}]'
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


class EmptyLearningLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "empty-learning"

    @property
    def model_name(self) -> str:
        return "empty-learning-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (temperature, max_tokens)
        prompt = messages[-1].content
        if "Analyze this development session transcript" in prompt:
            return LLMResponse(content="[]", model="empty-learning-model")
        if "Current STYLE.md" in prompt or "Current GUARDRAILS.md" in prompt:
            return LLMResponse(content="NONE", model="empty-learning-model")
        return LLMResponse(content="", model="empty-learning-model")

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


class FlakyRateLimitedLLM(LLMProvider):
    def __init__(self, fail_attempts: int = 2):
        self.fail_attempts = fail_attempts
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "flaky-rate-limit"

    @property
    def model_name(self) -> str:
        return "flaky-rate-limit-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        self.calls += 1
        if self.calls <= self.fail_attempts:
            raise LLMRateLimitError("retry later")
        return LLMResponse(
            content=(
                '[{"label": "pattern", "content": "Use checkpoints in migrations", '
                '"tags": ["db"], "confidence": 0.8}]'
            ),
            model="flaky-rate-limit-model",
        )

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
    assert second["sessions_already_processed"] == 1
    assert second["session_diagnostics"][0]["status"] == "skipped_already_processed"


@pytest.mark.asyncio
async def test_auto_sync_auto_approves_extracted_entries_when_curation_mode_disabled(
    storage,
    files,
    tmp_path: Path,
) -> None:
    files.write_config({"storage": {"curation_mode": False}})
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync()
    assert results["sessions_processed"] == 1
    assert results["learnings_extracted"] == 1
    assert storage.list_entries_by_curation_status(CurationStatus.PENDING) == []
    approved = storage.list_entries_by_curation_status(CurationStatus.APPROVED)
    assert len(approved) == 1


@pytest.mark.asyncio
async def test_auto_sync_keeps_extracted_entries_pending_when_curation_mode_enabled(
    storage,
    files,
    tmp_path: Path,
) -> None:
    files.write_config({"storage": {"curation_mode": True}})
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync()
    assert results["sessions_processed"] == 1
    assert results["learnings_extracted"] == 1
    pending = storage.list_entries_by_curation_status(CurationStatus.PENDING)
    assert len(pending) == 1
    assert storage.list_entries_by_curation_status(CurationStatus.APPROVED) == []


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
async def test_auto_sync_filters_by_session_id(storage, files, tmp_path: Path) -> None:
    first = tmp_path / "first-session"
    second = tmp_path / "second-session"
    first.write_text("one")
    second.write_text("two")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [first, second])],
    )

    results = await sync.sync(session_ids=["cursor-first-session"])
    assert results["sessions_discovered"] == 1
    assert results["sessions_processed"] == 1
    assert storage.is_session_processed("cursor-first-session") is True
    assert storage.is_session_processed("cursor-second-session") is False


@pytest.mark.asyncio
async def test_auto_sync_max_sessions_limits_to_most_recent(
    storage,
    files,
    tmp_path: Path,
) -> None:
    old_session = tmp_path / "old-session"
    new_session = tmp_path / "new-session"
    old_session.write_text("old")
    new_session.write_text("new")

    import os

    now = datetime.now(UTC).timestamp()
    os.utime(old_session, (now - 120, now - 120))
    os.utime(new_session, (now, now))

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [old_session, new_session])],
    )

    results = await sync.sync(max_sessions=1)
    assert results["sessions_discovered"] == 1
    assert results["sessions_processed"] == 1
    assert storage.is_session_processed("cursor-new-session") is True
    assert storage.is_session_processed("cursor-old-session") is False


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

    results = await sync.sync_and_compact(force_compact=True)

    assert "compaction" in results
    assert results["sessions_processed"] == 1
    events_path = files.agent_dir / "metrics" / "pipeline-events.jsonl"
    assert events_path.exists()
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = {event.get("stage") for event in events}
    assert "extract" in stages
    assert "ingest" in stages
    assert "compact" in stages


@pytest.mark.asyncio
async def test_compact_after_sync_runs_as_standalone_phase(storage, files, tmp_path: Path) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    sync_results = await sync.sync()
    results = await sync.compact_after_sync(sync_results, force_compact=True)

    assert "compaction" in results
    assert results["compaction"]["chunks_indexed"] >= 0


@pytest.mark.asyncio
async def test_sync_and_compact_external_backend_reports_pending(
    storage,
    files,
    tmp_path: Path,
) -> None:
    files.write_config(
        {
            "compaction": {
                "backend": "mcp_external",
                "external": {
                    "pending_limit": 10,
                },
            }
        }
    )

    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync_and_compact(force_compact=True)

    compaction = results.get("compaction", {})
    assert compaction.get("backend") == "mcp_external"
    assert compaction.get("external_required") is True
    assert "pending_external_conversations" in compaction


@pytest.mark.asyncio
async def test_sync_and_compact_defers_until_thresholds(storage, files, tmp_path: Path) -> None:
    files.write_config(
        {
            "compaction": {
                "max_sessions_before_compact": 5,
                "max_recent_tokens": 5000,
                "max_hours_before_compact": 9999,
            }
        }
    )

    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync_and_compact()

    compaction = results.get("compaction", {})
    assert compaction.get("deferred") is True
    assert compaction.get("deferred_reason") == "below_thresholds"
    assert compaction.get("llm_requests") == 0
    assert compaction.get("recent_updated") is False


@pytest.mark.asyncio
async def test_compact_after_sync_skips_when_no_learnings_and_not_forced(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.compact_after_sync(
        {
            "sessions_discovered": 1,
            "sessions_processed": 0,
            "sessions_skipped": 1,
            "learnings_extracted": 0,
            "by_source": {},
            "errors": [],
        }
    )

    assert "compaction" not in results


@pytest.mark.asyncio
async def test_sync_and_compact_backfills_missing_embeddings_when_enabled(
    storage, files, tmp_path: Path, monkeypatch
) -> None:
    files.write_config({"retrieval": {"semantic_index_enabled": True, "embedding_dimensions": 384}})

    preexisting = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="legacy chunk without embedding",
        label=SemanticLabel.PATTERN,
        embedding=None,
    )
    storage.store_chunk(preexisting)

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.33] * 384 for _ in texts],
    )

    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync_and_compact()

    assert results.get("embedding_indexing") == {"indexed": 1, "skipped": 0}
    embedded_chunks = storage.list_chunks_with_embeddings()
    assert any(chunk.id == preexisting.id for chunk in embedded_chunks)


@pytest.mark.asyncio
async def test_index_chunk_embeddings_runs_as_standalone_phase(storage, files, monkeypatch) -> None:
    files.write_config({"retrieval": {"semantic_index_enabled": True, "embedding_dimensions": 384}})

    preexisting = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="legacy chunk without embedding",
        label=SemanticLabel.PATTERN,
        embedding=None,
    )
    storage.store_chunk(preexisting)

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.33] * 384 for _ in texts],
    )

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[],
    )

    results = sync.index_chunk_embeddings(
        {
            "sessions_discovered": 0,
            "sessions_processed": 0,
            "sessions_skipped": 0,
            "learnings_extracted": 0,
            "by_source": {},
            "errors": [],
        }
    )

    assert results.get("embedding_indexing") == {"indexed": 1, "skipped": 0}
    embedded_chunks = storage.list_chunks_with_embeddings()
    assert any(chunk.id == preexisting.id for chunk in embedded_chunks)


@pytest.mark.asyncio
async def test_sync_and_compact_skip_embeddings_does_not_backfill(
    storage, files, tmp_path: Path
) -> None:
    files.write_config({"retrieval": {"semantic_index_enabled": True, "embedding_dimensions": 384}})

    preexisting = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="legacy chunk without embedding",
        label=SemanticLabel.PATTERN,
        embedding=None,
    )
    storage.store_chunk(preexisting)

    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[FakeIngester("cursor", [session_path])],
    )

    results = await sync.sync_and_compact(skip_embeddings=True)

    assert "embedding_indexing" not in results
    still_missing = [chunk for chunk in storage.list_chunks() if chunk.id == preexisting.id]
    assert len(still_missing) == 1
    assert still_missing[0].embedding is None


@pytest.mark.asyncio
async def test_sync_and_compact_orchestrates_sync_compact_and_embedding(
    storage, files, monkeypatch
) -> None:
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[],
    )
    calls: list[str] = []

    async def _fake_sync(**kwargs: object) -> dict[str, object]:
        _ = kwargs
        calls.append("sync")
        return {"learnings_extracted": 1, "by_source": {}, "errors": []}

    async def _fake_compact_after_sync(
        sync_results: dict[str, object],
        *,
        force_compact: bool = False,
        telemetry: object | None = None,
        telemetry_run_id: str | None = None,
    ) -> dict[str, object]:
        assert calls == ["sync"]
        assert force_compact is True
        assert telemetry is not None
        assert telemetry_run_id is not None
        calls.append("compact")
        sync_results["compaction"] = {"recent_updated": True}
        return sync_results

    def _fake_index_chunk_embeddings(
        sync_results: dict[str, object],
        *,
        skip_embeddings: bool = False,
    ) -> dict[str, object]:
        assert calls == ["sync", "compact"]
        assert skip_embeddings is True
        calls.append("embed")
        sync_results["embedding_indexing"] = {"indexed": 0, "skipped": 0}
        return sync_results

    def _fake_sync_vector_memory(
        sync_results: dict[str, object],
        *,
        trigger: str,
    ) -> dict[str, object]:
        assert calls == ["sync", "compact", "embed"]
        assert trigger == "sync_and_compact"
        calls.append("vectors")
        sync_results["vector_memory_sync"] = {"status": "ready"}
        return sync_results

    monkeypatch.setattr(sync, "sync", _fake_sync)
    monkeypatch.setattr(sync, "compact_after_sync", _fake_compact_after_sync)
    monkeypatch.setattr(sync, "index_chunk_embeddings", _fake_index_chunk_embeddings)
    monkeypatch.setattr(sync, "sync_vector_memory", _fake_sync_vector_memory)

    results = await sync.sync_and_compact(force_compact=True, skip_embeddings=True)

    assert calls == ["sync", "compact", "embed", "vectors"]
    assert results["compaction"]["recent_updated"] is True
    assert results["embedding_indexing"] == {"indexed": 0, "skipped": 0}
    assert results["vector_memory_sync"] == {"status": "ready"}


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
    assert first["sessions_processed"] == 0
    assert first["sessions_skipped"] == 1
    assert first["learnings_extracted"] == 0
    assert len(first["errors"]) == 1
    assert "extraction failed" in first["errors"][0]
    assert first["session_diagnostics"][0]["status"] == "failed_extraction"
    assert storage.is_session_processed("cursor-cursor-session") is False

    second = await sync.sync()
    assert second["sessions_processed"] == 0
    assert second["sessions_skipped"] == 1
    assert second["session_diagnostics"][0]["status"] == "failed_extraction"


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
    sync.extract_retry_attempts = 1
    sync.extract_retry_backoff_seconds = 0.0

    results = await sync.sync()
    assert results["sessions_processed"] == 0
    assert results["sessions_skipped"] == 1
    assert results["learnings_extracted"] == 0
    assert len(results["errors"]) == 1
    assert "timed out" in results["errors"][0]
    assert results["session_diagnostics"][0]["status"] == "failed_extraction"
    assert storage.is_session_processed("cursor-cursor-session") is False


@pytest.mark.asyncio
async def test_auto_sync_retries_rate_limit_then_processes(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    llm = FlakyRateLimitedLLM(fail_attempts=2)
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=llm,
        ingesters=[FakeIngester("cursor", [session_path])],
    )
    sync.extract_retry_attempts = 3
    sync.extract_retry_backoff_seconds = 0.0

    results = await sync.sync()
    assert llm.calls == 3
    assert results["sessions_processed"] == 1
    assert results["sessions_skipped"] == 0
    assert results["learnings_extracted"] == 1
    assert results["errors"] == []
    assert results["session_diagnostics"][0]["status"] == "processed"
    assert storage.is_session_processed("cursor-cursor-session") is True


@pytest.mark.asyncio
async def test_auto_sync_rate_limit_honors_retry_after_and_reduces_batch_size(
    storage,
    files,
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    class RetryAfterLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def provider_name(self) -> str:
            return "retry-after"

        @property
        def model_name(self) -> str:
            return "retry-after-model"

        async def generate(
            self,
            messages: list[Message],
            temperature: float = 0.3,
            max_tokens: int = 4096,
        ) -> LLMResponse:
            _ = (messages, temperature, max_tokens)
            self.calls += 1
            if self.calls == 1:
                raise LLMRateLimitError("retry later", retry_after_seconds=0.25)
            return LLMResponse(
                content=(
                    '[{"label": "pattern", "content": "Use stable retry pacing", '
                    '"tags": ["sync"], "confidence": 0.8}]'
                ),
                model="retry-after-model",
            )

        def validate(self) -> tuple[bool, str]:
            return True, "ok"

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(float(seconds))

    monkeypatch.setattr("agent_recall.core.sync.asyncio.sleep", _fake_sleep)

    progress_events: list[dict[str, object]] = []
    llm = RetryAfterLLM()
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=llm,
        ingesters=[FakeIngester("cursor", [session_path])],
        progress_callback=progress_events.append,
    )
    sync.extract_retry_attempts = 2
    sync.extract_retry_backoff_seconds = 0.0

    results = await sync.sync()

    assert llm.calls == 2
    assert results["sessions_processed"] == 1
    assert sleep_calls == [pytest.approx(0.25)]
    assert sync.extractor is not None
    assert sync.extractor.messages_per_batch == 50

    retry_events = [
        event for event in progress_events if event.get("event") == "extraction_retry_scheduled"
    ]
    assert len(retry_events) == 1
    assert retry_events[0]["reason"] == "rate_limit"
    assert retry_events[0]["delay_seconds"] == pytest.approx(0.25)

    batch_resize_events = [
        event for event in progress_events if event.get("event") == "extraction_batch_size_adjusted"
    ]
    assert len(batch_resize_events) == 1
    assert batch_resize_events[0]["old_messages_per_batch"] == 50
    assert batch_resize_events[0]["new_messages_per_batch"] == 25


@pytest.mark.asyncio
async def test_auto_sync_warns_when_long_session_yields_no_learnings(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    class LongConversationIngester(FakeIngester):
        def parse_session(self, path: Path) -> RawSession:
            messages = []
            for index in range(60):
                role = "user" if index % 2 == 0 else "assistant"
                messages.append(
                    RawMessage(
                        role=role,
                        content=f"Message {index + 1} about debugging parser behavior",
                    )
                )
            return RawSession(
                source=self.source_name,
                session_id=self.get_session_id(path),
                project_path=path.parent,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                messages=messages,
            )

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=EmptyLearningLLM(),
        ingesters=[LongConversationIngester("cursor", [session_path])],
    )

    results = await sync.sync()
    assert results["sessions_processed"] == 1
    assert results["learnings_extracted"] == 0
    assert any("60 messages but yielded 0 learnings" in error for error in results["errors"])
    diagnostics = results["session_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["message_count"] == 60
    assert diagnostics[0]["learnings_extracted"] == 0
    assert diagnostics[0]["status"] == "processed"
    assert "warning" in diagnostics[0]


@pytest.mark.asyncio
async def test_auto_sync_recovery_pass_extracts_learning_and_counts_request(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    class RecoveryIngester(FakeIngester):
        def parse_session(self, path: Path) -> RawSession:
            messages = [
                RawMessage(
                    role="user" if index % 2 == 0 else "assistant",
                    content=(
                        "Progress update on the browser engine migration and traversal "
                        f"work. Message {index + 1}."
                    ),
                )
                for index in range(58)
            ]
            messages.extend(
                [
                    RawMessage(
                        role="assistant",
                        content=(
                            "Implemented the decision to keep Playwright as the default "
                            "browser engine while preserving agent-browser as the explicit "
                            "fallback path."
                        ),
                    ),
                    RawMessage(
                        role="assistant",
                        content=(
                            "Final result: Playwright is the default, agent-browser remains "
                            "available as a compatibility fallback, and click traversal uses "
                            "explicit wait-before-click guards."
                        ),
                    ),
                ]
            )
            return RawSession(
                source=self.source_name,
                session_id=self.get_session_id(path),
                project_path=path.parent,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                messages=messages,
            )

    class RecoveryLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def provider_name(self) -> str:
            return "recovery"

        @property
        def model_name(self) -> str:
            return "recovery-model"

        async def generate(
            self,
            messages: list[Message],
            temperature: float = 0.3,
            max_tokens: int = 4096,
        ) -> LLMResponse:
            _ = (temperature, max_tokens)
            self.calls += 1
            prompt = messages[-1].content
            if "HIGH-SIGNAL SNIPPETS START" in prompt:
                return LLMResponse(
                    content=(
                        '[{"label":"decision","content":"Keep Playwright as the default '
                        'browser engine and retain agent-browser as the explicit fallback",'
                        '"tags":["playwright","fallback"],"confidence":0.9}]'
                    ),
                    model="recovery-model",
                )
            return LLMResponse(content="[]", model="recovery-model")

        def validate(self) -> tuple[bool, str]:
            return True, "ok"

    llm = RecoveryLLM()
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=llm,
        ingesters=[RecoveryIngester("cursor", [session_path])],
    )

    results = await sync.sync()

    assert results["sessions_processed"] == 1
    assert results["learnings_extracted"] == 1
    assert results["llm_requests"] == 3
    assert results["by_source"]["cursor"]["llm_batches"] == 3
    assert not any("yielded 0 learnings" in error for error in results["errors"])
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_auto_sync_batches_extraction_and_emits_progress(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    class VeryLongConversationIngester(FakeIngester):
        def parse_session(self, path: Path) -> RawSession:
            messages = []
            for index in range(205):
                role = "user" if index % 2 == 0 else "assistant"
                messages.append(
                    RawMessage(
                        role=role,
                        content=(
                            f"Message {index + 1}: harden migration sequencing, retry handling, "
                            "and rollback guarantees for partial failures."
                        ),
                    )
                )
            return RawSession(
                source=self.source_name,
                session_id=self.get_session_id(path),
                project_path=path.parent,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                messages=messages,
            )

    progress_events: list[dict[str, object]] = []
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[VeryLongConversationIngester("cursor", [session_path])],
        progress_callback=progress_events.append,
    )

    results = await sync.sync()
    assert results["sessions_processed"] == 1
    assert results["llm_requests"] == 5
    assert results["by_source"]["cursor"]["llm_batches"] == 5

    session_events = [
        event for event in progress_events if event.get("event") == "extraction_session_started"
    ]
    batch_events = [
        event for event in progress_events if event.get("event") == "extraction_batch_complete"
    ]
    assert len(session_events) == 1
    assert len(batch_events) == 5
    assert batch_events[0]["messages_processed"] == 50
    assert batch_events[1]["messages_processed"] == 100
    assert batch_events[2]["messages_processed"] == 150
    assert batch_events[3]["messages_processed"] == 200
    assert batch_events[4]["messages_processed"] == 205


def test_auto_sync_list_sessions_includes_titles_and_processed_state(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session-one"
    session_path.write_text("session")

    class TitledIngester(FakeIngester):
        def parse_session(self, path: Path) -> RawSession:
            parsed = super().parse_session(path)
            return parsed.model_copy(update={"title": f"Title for {path.stem}"})

    session_id = "cursor-session-one"
    storage.mark_session_processed(session_id)

    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[TitledIngester("cursor", [session_path])],
    )

    listed = sync.list_sessions()
    assert listed["sessions_discovered"] == 1
    assert len(listed["sessions"]) == 1
    row = listed["sessions"][0]
    assert row["session_id"] == session_id
    assert row["title"] == "Title for session-one"
    assert row["processed"] is True


class GrowingSessionIngester(FakeIngester):
    """Ingester that simulates a growing session with new messages added each time."""

    def __init__(self, source_name: str, sessions: list[Path], message_count: int = 2):
        super().__init__(source_name, sessions)
        self._message_count = message_count

    def parse_session(self, path: Path) -> RawSession:
        messages = []
        for i in range(self._message_count):
            role = "user" if i % 2 == 0 else "assistant"
            # Use content that matches the AdaptiveLLM pattern for extracting learnings
            if i == 0:
                content = (
                    "Please fix the migration ordering and make sure rollbacks are safe "
                    "for partial writes."
                )
            else:
                content = (
                    "I wrapped writes in explicit transactions, added rollback behavior, "
                    "and validated the migration sequence."
                )
            messages.append(
                RawMessage(
                    role=role,
                    content=content,
                )
            )
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            project_path=path.parent,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            messages=messages,
        )


@pytest.mark.asyncio
async def test_incremental_sync_processes_only_new_messages(storage, files, tmp_path: Path) -> None:
    """Test that incremental sync only processes messages after the checkpoint."""
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    # First sync with 2 messages
    ingester_v1 = GrowingSessionIngester("cursor", [session_path], message_count=2)
    sync_v1 = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[ingester_v1],
    )

    first = await sync_v1.sync()
    assert first["sessions_processed"] == 1
    assert first["learnings_extracted"] == 1
    assert first["sessions_incremental"] == 0  # First sync is not incremental

    # Verify checkpoint was created
    checkpoint = storage.get_session_checkpoint("cursor-cursor-session")
    assert checkpoint is not None
    assert checkpoint.last_message_index == 1  # 0-indexed, 2 messages = index 1

    # Second sync with 4 messages (2 new)
    ingester_v2 = GrowingSessionIngester("cursor", [session_path], message_count=4)
    sync_v2 = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[ingester_v2],
    )

    second = await sync_v2.sync()
    assert second["sessions_processed"] == 1
    assert second["sessions_incremental"] == 1  # This is an incremental sync
    # The diagnostic should show we processed only the new messages
    assert second["session_diagnostics"][0]["message_count"] == 2  # Only 2 new messages
    assert second["session_diagnostics"][0]["original_message_count"] == 4


@pytest.mark.asyncio
async def test_incremental_sync_no_changes_skips(storage, files, tmp_path: Path) -> None:
    """Test that sync skips when content hasn't changed."""
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    ingester = GrowingSessionIngester("cursor", [session_path], message_count=2)
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[ingester],
    )

    # First sync
    first = await sync.sync()
    assert first["sessions_processed"] == 1

    # Second sync with same content
    second = await sync.sync()
    assert second["sessions_processed"] == 0
    assert second["sessions_skipped"] == 1
    assert second["sessions_already_processed"] == 1


@pytest.mark.asyncio
async def test_reset_checkpoints_clears_and_reprocesses(storage, files, tmp_path: Path) -> None:
    """Test that reset_checkpoints clears checkpoints and allows reprocessing."""
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    ingester = GrowingSessionIngester("cursor", [session_path], message_count=2)
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[ingester],
    )

    # First sync
    first = await sync.sync()
    assert first["sessions_processed"] == 1

    # Reset checkpoints and sync again - should reprocess
    second = await sync.sync(reset_checkpoints=True)
    assert second["sessions_processed"] == 1
    assert second["sessions_incremental"] == 0  # Full reprocess, not incremental


@pytest.mark.asyncio
async def test_reset_full_clears_everything(storage, files, tmp_path: Path) -> None:
    """Test that reset_full clears both checkpoints and processed markers."""
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")

    ingester = GrowingSessionIngester("cursor", [session_path], message_count=2)
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[ingester],
    )

    # First sync
    first = await sync.sync()
    assert first["sessions_processed"] == 1

    # Verify checkpoint and processed marker exist
    assert storage.get_session_checkpoint("cursor-cursor-session") is not None
    assert storage.is_session_processed("cursor-cursor-session") is True

    # Reset full and sync again
    second = await sync.sync(reset_full=True)
    assert second["sessions_processed"] == 1

    # Check that we have fresh checkpoint
    checkpoint = storage.get_session_checkpoint("cursor-cursor-session")
    assert checkpoint is not None


class BrokenParseIngester(FakeIngester):
    def parse_session(self, path: Path) -> RawSession:
        _ = path
        raise ValueError("broken parse")


class SingleMessageIngester(FakeIngester):
    def parse_session(self, path: Path) -> RawSession:
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            project_path=path.parent,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            messages=[RawMessage(role="user", content="only one message")],
        )


def test_sync_stage_discover_and_filter_contract(storage, files, tmp_path: Path) -> None:
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

    discover_stage = sync._run_discover_stage(since=None, sources=["cursor"])
    assert isinstance(discover_stage, SyncDiscoverStage)
    assert [ingester.source_name for ingester in discover_stage.active_ingesters] == ["cursor"]
    assert len(discover_stage.candidates) == 1

    filter_stage = sync._run_filter_stage(
        discover_stage=discover_stage,
        session_ids=None,
        max_sessions=1,
    )
    assert isinstance(filter_stage, SyncFilterStage)
    assert len(filter_stage.candidates) == 1
    assert not filter_stage.missing_session_ids


def test_sync_session_filter_stage_failed_parse(storage, files, tmp_path: Path) -> None:
    broken_session = tmp_path / "broken-session"
    broken_session.write_text("broken")
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[BrokenParseIngester("cursor", [broken_session])],
    )
    discover_stage = sync._run_discover_stage(since=None, sources=["cursor"])
    candidate = discover_stage.candidates[0]
    stage = sync._run_session_filter_stage(candidate, reset_checkpoints=False)
    assert stage.status == "failed_parse"
    assert stage.error == "broken parse"


def test_sync_session_filter_stage_checkpoint_branches(storage, files, tmp_path: Path) -> None:
    single = tmp_path / "single-session"
    single.write_text("single")
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=AdaptiveLLM(),
        ingesters=[SingleMessageIngester("cursor", [single])],
    )
    discover_stage = sync._run_discover_stage(since=None, sources=["cursor"])
    candidate = discover_stage.candidates[0]

    first = sync._run_session_filter_stage(candidate, reset_checkpoints=False)
    assert first.status == "skip_empty"
    assert first.raw_session is not None
    assert first.content_hash is not None
    sync._persist_empty_session(
        candidate.session_id,
        raw_session=first.raw_session,
        content_hash=first.content_hash,
        is_fully_processed=first.is_fully_processed,
    )

    second = sync._run_session_filter_stage(candidate, reset_checkpoints=False)
    assert second.status == "skip_already_processed"
    assert second.message_count == 1


@pytest.mark.asyncio
async def test_sync_extract_and_persist_stage_contracts(
    storage,
    files,
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "cursor-session"
    session_path.write_text("session")
    llm = FlakyRateLimitedLLM(fail_attempts=1)
    sync = AutoSync(
        storage=storage,
        files=files,
        llm=llm,
        ingesters=[FakeIngester("cursor", [session_path])],
    )
    sync.extract_retry_backoff_seconds = 0
    discover_stage = sync._run_discover_stage(since=None, sources=["cursor"])
    candidate = discover_stage.candidates[0]
    filter_stage = sync._run_session_filter_stage(candidate, reset_checkpoints=False)
    assert filter_stage.status == "process"
    assert filter_stage.raw_session is not None
    assert filter_stage.content_hash is not None

    telemetry = PipelineTelemetry.from_config(
        agent_dir=files.agent_dir,
        config=files.read_config(),
    )
    extract_stage = await sync._run_extract_stage(
        candidate,
        raw_session=filter_stage.raw_session,
        telemetry=telemetry,
        run_id="sync-stage-test",
    )
    assert extract_stage.success is True
    assert len(extract_stage.entries) >= 1

    persist_stage = sync._run_persist_stage(
        candidate,
        raw_session=filter_stage.raw_session,
        content_hash=filter_stage.content_hash,
        is_fully_processed=filter_stage.is_fully_processed,
        entries=extract_stage.entries,
    )
    assert persist_stage.entries_written == len(extract_stage.entries)
    assert storage.get_session_checkpoint(candidate.session_id) is not None
    assert storage.is_session_processed(candidate.session_id) is True
