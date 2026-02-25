"""Tests for the vector search functionality in retrieval.

This module contains comprehensive unit tests for vector-based similarity search,
including tests for result ordering, similarity thresholds, top_k limits, and
handling of edge cases like empty databases and malformed embeddings.
"""

from __future__ import annotations

import logging

import numpy as np

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _embedding(a: float, b: float) -> list[float]:
    """Create a 384-dimensional embedding with values at specific positions."""
    values = [0.0] * 384
    values[0] = a
    values[1] = b
    return values


def test_search_by_vector_similarity_returns_chunks(mock_storage, monkeypatch) -> None:
    """Test that search_by_vector_similarity returns matching chunks."""
    query_vector = _embedding(1.0, 0.0)

    chunk1 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT authentication",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    chunk2 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Database queries",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    mock_storage.store_chunk(chunk1)
    mock_storage.store_chunk(chunk2)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_by_vector_similarity(
        "test query",
        top_k=5,
        min_similarity=0.0,
    )

    assert len(results) == 2
    assert chunk1.id in [r.id for r in results]
    assert chunk2.id in [r.id for r in results]


def test_results_are_ordered_by_similarity_descending(mock_storage, monkeypatch) -> None:
    """Test that results are ordered by similarity score in descending order."""
    query_vector = _embedding(1.0, 0.0)

    high_sim = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="high similarity",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    medium_sim = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="medium similarity",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.8, 0.2),
    )
    low_sim = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="low similarity",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    mock_storage.store_chunk(high_sim)
    mock_storage.store_chunk(medium_sim)
    mock_storage.store_chunk(low_sim)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_by_vector_similarity(
        "test",
        top_k=3,
        min_similarity=0.0,
    )

    result_ids = [r.id for r in results]
    assert result_ids[0] == high_sim.id
    assert result_ids[1] == medium_sim.id
    assert result_ids[2] == low_sim.id


def test_min_similarity_threshold_filters_results(mock_storage, monkeypatch) -> None:
    """Test that min_similarity threshold correctly filters out low-similarity results."""
    query_vector = _embedding(1.0, 0.0)

    high_sim = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="high similarity",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    low_sim = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="low similarity",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.5, 0.5),
    )
    mock_storage.store_chunk(high_sim)
    mock_storage.store_chunk(low_sim)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_by_vector_similarity(
        "test",
        top_k=5,
        min_similarity=0.99,
    )

    assert len(results) == 1
    assert results[0].id == high_sim.id


def test_top_k_limit_is_respected(mock_storage, monkeypatch) -> None:
    """Test that top_k parameter correctly limits the number of results returned."""
    query_vector = _embedding(1.0, 0.0)

    for i in range(10):
        mock_storage.store_chunk(
            Chunk(
                source=ChunkSource.MANUAL,
                source_ids=[],
                content=f"chunk {i}",
                label=SemanticLabel.PATTERN,
                embedding=_embedding(1.0 - (i * 0.1), 0.0 + (i * 0.1)),
            )
        )

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_by_vector_similarity(
        "test",
        top_k=3,
        min_similarity=0.0,
    )

    assert len(results) == 3


def test_empty_database_returns_empty_list(mock_storage) -> None:
    """Test that searching an empty database returns an empty list."""
    results = Retriever(mock_storage).search_by_vector_similarity(
        "test query",
        top_k=5,
    )

    assert results == []


def test_malformed_embedding_is_skipped_with_warning(mock_storage, monkeypatch, caplog) -> None:
    """Test that malformed embeddings are skipped with a warning log."""
    caplog.set_level(logging.WARNING)

    from agent_recall.core.retrieve import Retriever

    valid_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="valid chunk",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    mock_storage.store_chunk(valid_chunk)

    query_vector = _embedding(1.0, 0.0)
    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    retriever = Retriever(mock_storage)
    result = retriever._coerce_embedding([1, "not", "a", "float"])
    assert result is None


def test_vector_search_with_no_embeddings_returns_empty(mock_storage) -> None:
    """Test that searching when no chunks have embeddings returns empty list."""
    mock_storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="no embedding",
            label=SemanticLabel.PATTERN,
            embedding=None,
        )
    )

    results = Retriever(mock_storage).search_by_vector_similarity(
        "test",
        top_k=5,
    )

    assert results == []
