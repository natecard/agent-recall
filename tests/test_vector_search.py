from __future__ import annotations

import numpy as np

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _embedding(a: float, b: float) -> list[float]:
    values = [0.0] * 384
    values[0] = a
    values[1] = b
    return values


def test_vector_search_returns_results_in_similarity_order(storage, monkeypatch) -> None:
    query_vector = _embedding(1.0, 0.0)
    high = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="high",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    medium = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="medium",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.8, 0.2),
    )
    low = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="low",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    storage.store_chunk(high)
    storage.store_chunk(medium)
    storage.store_chunk(low)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    results = Retriever(storage).search_by_vector_similarity(
        "jwt auth",
        top_k=3,
        min_similarity=0.0,
    )

    assert [item.id for item in results] == [high.id, medium.id, low.id]


def test_vector_search_respects_similarity_threshold(storage, monkeypatch) -> None:
    query_vector = _embedding(1.0, 0.0)
    high = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="high",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    medium = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="medium",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.8, 0.2),
    )
    storage.store_chunk(high)
    storage.store_chunk(medium)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    results = Retriever(storage).search_by_vector_similarity(
        "jwt auth",
        top_k=5,
        min_similarity=0.99,
    )

    assert [item.id for item in results] == [high.id]


def test_vector_search_on_empty_database_returns_empty_list(storage) -> None:
    results = Retriever(storage).search_by_vector_similarity("hello world", top_k=5)
    assert results == []
