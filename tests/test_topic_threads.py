from __future__ import annotations

from uuid import UUID

from agent_recall.core.topic_threads import build_topic_threads
from agent_recall.storage.models import Chunk, ChunkSource, LogEntry, LogSource, SemanticLabel


def _append_entry(storage, *, entry_id: UUID, content: str, source_session_id: str) -> None:
    storage.append_entry(
        LogEntry(
            id=entry_id,
            source=LogSource.EXTRACTED,
            source_session_id=source_session_id,
            content=content,
            label=SemanticLabel.PATTERN,
            tags=["auth"],
        )
    )


def test_build_topic_threads_and_storage_round_trip(storage) -> None:
    first_entry = UUID("00000000-0000-0000-0000-000000000061")
    second_entry = UUID("00000000-0000-0000-0000-000000000062")
    _append_entry(
        storage,
        entry_id=first_entry,
        content="auth timeout mitigation",
        source_session_id="cursor-auth-1",
    )
    _append_entry(
        storage,
        entry_id=second_entry,
        content="auth token refresh strategy",
        source_session_id="cursor-auth-2",
    )

    storage.store_chunk(
        Chunk(
            id=UUID("00000000-0000-0000-0000-000000000071"),
            source=ChunkSource.LOG_ENTRY,
            source_ids=[first_entry],
            content="auth timeout mitigation for api clients",
            label=SemanticLabel.PATTERN,
            tags=["src/auth/session.py"],
        )
    )
    storage.store_chunk(
        Chunk(
            id=UUID("00000000-0000-0000-0000-000000000072"),
            source=ChunkSource.LOG_ENTRY,
            source_ids=[second_entry],
            content="auth token refresh strategy for long sessions",
            label=SemanticLabel.PATTERN,
            tags=["src/auth/token.py"],
        )
    )

    threads = build_topic_threads(storage, min_cluster_size=2, max_threads=10)
    assert threads
    first_thread = threads[0]
    assert first_thread["entry_count"] >= 2
    assert first_thread["source_session_count"] >= 1

    inserted = storage.replace_topic_threads(threads)
    assert inserted == len(threads)

    listed = storage.list_topic_threads(limit=10)
    assert listed
    thread_id = str(listed[0]["thread_id"])
    detail = storage.get_topic_thread(thread_id, limit_links=10)
    assert detail is not None
    assert detail["thread_id"] == thread_id
    assert detail["links"]
