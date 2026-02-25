"""Tests for hybrid search functionality combining FTS and semantic search.

This module contains comprehensive unit tests for the hybrid search feature,
including tests for result deduplication, weight configurations, and ranking
by combined FTS and semantic scores.
"""

from __future__ import annotations

import numpy as np

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _embedding(a: float, b: float) -> list[float]:
    """Create a 384-dimensional embedding with values at specific positions."""
    vector = [0.0] * 384
    vector[0] = a
    vector[1] = b
    return vector


def test_hybrid_search_combines_fts_and_semantic(mock_storage, monkeypatch) -> None:
    """Test that hybrid search combines results from both FTS and semantic search."""
    query = "JWT token validation"
    query_vector = _embedding(1.0, 0.0)

    fts_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT token validation logic",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    semantic_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Credential refresh handling",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    mock_storage.store_chunk(fts_chunk)
    mock_storage.store_chunk(semantic_chunk)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_hybrid(query=query, top_k=5)

    assert len(results) == 2


def test_hybrid_deduplicates_overlapping_results(mock_storage, monkeypatch) -> None:
    """Test that hybrid search deduplicates chunks that appear in both FTS and semantic results."""
    query = "JWT auth"
    query_vector = _embedding(1.0, 0.0)

    overlapping_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT authentication token handling",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    mock_storage.store_chunk(overlapping_chunk)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_hybrid(query=query, top_k=5)

    ids = [item.id for item in results]
    assert ids.count(overlapping_chunk.id) == 1
    assert len(ids) == 1


def test_pure_fts_when_semantic_weight_is_zero(mock_storage, monkeypatch) -> None:
    """Test that setting semantic_weight to 0 results in pure FTS ranking."""
    chunk1 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="jwt first match",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    chunk2 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="jwt second match",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    mock_storage.store_chunk(chunk1)
    mock_storage.store_chunk(chunk2)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(_embedding(1.0, 0.0)),
    )

    retriever = Retriever(mock_storage)
    fts_results = retriever.storage.search_chunks_fts("jwt", top_k=2)
    hybrid_results = retriever.search_hybrid(
        query="jwt",
        top_k=2,
        fts_weight=1.0,
        semantic_weight=0.0,
    )

    assert [item.id for item in hybrid_results] == [item.id for item in fts_results]


def test_pure_semantic_when_fts_weight_is_zero(mock_storage, monkeypatch) -> None:
    """Test that setting fts_weight to 0 results in pure semantic ranking."""
    query_vector = _embedding(1.0, 0.0)

    chunk1 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="semantic first",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    chunk2 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="semantic second",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.8, 0.2),
    )
    mock_storage.store_chunk(chunk1)
    mock_storage.store_chunk(chunk2)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    retriever = Retriever(mock_storage)
    semantic_results = retriever.search_by_vector_similarity("auth", top_k=2, min_similarity=0.0)
    hybrid_results = retriever.search_hybrid(
        query="auth",
        top_k=2,
        fts_weight=0.0,
        semantic_weight=1.0,
    )

    assert [item.id for item in hybrid_results] == [item.id for item in semantic_results]


def test_results_ranked_by_hybrid_score(mock_storage, monkeypatch) -> None:
    """Test that results are ranked by the combined hybrid score."""
    query_vector = _embedding(1.0, 0.0)

    high_fts_high_semantic = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT auth token validation",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    other = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT handling code",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    mock_storage.store_chunk(high_fts_high_semantic)
    mock_storage.store_chunk(other)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_hybrid(
        query="JWT auth",
        top_k=2,
        fts_weight=0.4,
        semantic_weight=0.6,
    )

    assert len(results) >= 1
    assert results[0].id == high_fts_high_semantic.id


def test_hybrid_search_with_empty_fts_results(mock_storage, monkeypatch) -> None:
    """Test hybrid search when FTS returns no results but semantic does."""
    query_vector = _embedding(1.0, 0.0)

    semantic_only = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="some unrelated content",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    mock_storage.store_chunk(semantic_only)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    results = Retriever(mock_storage).search_hybrid(
        query="nonexistent term xyz123",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].id == semantic_only.id


def test_hybrid_search_with_empty_semantic_results(mock_storage, monkeypatch) -> None:
    """Test hybrid search when semantic returns no results but FTS does."""
    fts_only = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT authentication",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    mock_storage.store_chunk(fts_only)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(_embedding(0.5, 0.5)),
    )

    results = Retriever(mock_storage).search_hybrid(
        query="JWT auth",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].id == fts_only.id


def test_hybrid_search_deterministic_ranking(mock_storage, monkeypatch) -> None:
    """Test that hybrid search produces deterministic results across multiple runs."""
    query_vector = _embedding(1.0, 0.0)

    chunk1 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="alpha test content",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    chunk2 = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="beta test content",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    mock_storage.store_chunk(chunk1)
    mock_storage.store_chunk(chunk2)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(query_vector),
    )

    retriever = Retriever(mock_storage)
    run1 = retriever.search_hybrid(query="test", top_k=2)
    run2 = retriever.search_hybrid(query="test", top_k=2)

    assert [item.id for item in run1] == [item.id for item in run2]
