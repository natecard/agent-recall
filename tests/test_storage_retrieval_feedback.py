from __future__ import annotations

from uuid import UUID

import pytest

from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _create_chunk(storage, *, content: str) -> Chunk:
    chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=SemanticLabel.PATTERN,
    )
    storage.store_chunk(chunk)
    return chunk


def test_storage_retrieval_feedback_round_trip(storage) -> None:
    chunk = _create_chunk(storage, content="retry queue bounds")
    created = storage.record_retrieval_feedback(
        query="retry queue",
        chunk_id=chunk.id,
        score=1,
        actor="alice",
        source="tui",
        metadata={"surface": "slash"},
    )
    assert created["query_text"] == "retry queue"
    assert created["chunk_id"] == str(chunk.id)
    assert created["score"] == 1

    rows = storage.list_retrieval_feedback(limit=10, query="retry queue")
    assert len(rows) == 1
    assert rows[0]["actor"] == "alice"
    assert rows[0]["source"] == "tui"
    assert rows[0]["metadata"] == {"surface": "slash"}


def test_storage_retrieval_feedback_scores_use_query_specific_first(storage) -> None:
    chunk = _create_chunk(storage, content="vector fallback")
    storage.record_retrieval_feedback(query="alpha query", chunk_id=chunk.id, score=-1)
    storage.record_retrieval_feedback(query="alpha query", chunk_id=chunk.id, score=1)
    storage.record_retrieval_feedback(query="beta query", chunk_id=chunk.id, score=1)

    scores_alpha = storage.get_retrieval_feedback_scores(
        query="alpha query",
        chunk_ids=[chunk.id],
    )
    assert scores_alpha[chunk.id] == 0.0

    scores_gamma = storage.get_retrieval_feedback_scores(
        query="gamma query",
        chunk_ids=[chunk.id],
    )
    assert round(scores_gamma[chunk.id], 3) == round((1.0 / 3.0) * 0.5, 3)


def test_storage_retrieval_feedback_rejects_unknown_chunk(storage) -> None:
    with pytest.raises(ValueError, match="chunk not found"):
        storage.record_retrieval_feedback(
            query="no chunk",
            chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
            score=1,
        )


def test_storage_retrieval_feedback_rejects_invalid_score(storage) -> None:
    chunk = _create_chunk(storage, content="retry")
    with pytest.raises(ValueError, match="score must be -1"):
        storage.record_retrieval_feedback(query="retry", chunk_id=chunk.id, score=3)
