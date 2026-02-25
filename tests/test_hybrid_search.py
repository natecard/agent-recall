from __future__ import annotations

import numpy as np

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _embedding(a: float, b: float) -> list[float]:
    vector = [0.0] * 384
    vector[0] = a
    vector[1] = b
    return vector


def test_hybrid_search_combines_fts_and_semantic(storage, monkeypatch) -> None:
    query = "jwt"
    query_vector = _embedding(1.0, 0.0)

    fts_only = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT session token validation",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    semantic_only = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Credential refresh lifecycle",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    storage.store_chunk(fts_only)
    storage.store_chunk(semantic_only)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    results = Retriever(storage).search_hybrid(query=query, top_k=5)

    result_ids = {item.id for item in results}
    assert fts_only.id in result_ids
    assert semantic_only.id in result_ids


def test_hybrid_search_deduplicates_overlap(storage, monkeypatch) -> None:
    query = "jwt auth"
    query_vector = _embedding(1.0, 0.0)
    overlapping = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="JWT auth overlap chunk",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    storage.store_chunk(overlapping)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    results = Retriever(storage).search_hybrid(query=query, top_k=5)

    ids = [item.id for item in results]
    assert ids.count(overlapping.id) == 1


def test_hybrid_search_with_fts_only_weight_matches_fts_order(storage, monkeypatch) -> None:
    first = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="jwt first",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    second = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="jwt second",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.0, 1.0),
    )
    storage.store_chunk(first)
    storage.store_chunk(second)

    monkeypatch.setattr(
        "agent_recall.core.retrieve.embed_single",
        lambda _: np.array(_embedding(1.0, 0.0)),
    )

    retriever = Retriever(storage)
    fts = retriever.storage.search_chunks_fts("jwt", top_k=2)
    hybrid = retriever.search_hybrid(
        query="jwt",
        top_k=2,
        fts_weight=1.0,
        semantic_weight=0.0,
    )

    assert [item.id for item in hybrid] == [item.id for item in fts]


def test_hybrid_search_with_semantic_only_weight_matches_semantic_order(
    storage,
    monkeypatch,
) -> None:
    query_vector = _embedding(1.0, 0.0)
    first = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="semantic one",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    second = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="semantic two",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(0.8, 0.2),
    )
    storage.store_chunk(first)
    storage.store_chunk(second)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    retriever = Retriever(storage)
    semantic = retriever.search_by_vector_similarity("auth", top_k=2, min_similarity=0.0)
    hybrid = retriever.search_hybrid(
        query="auth",
        top_k=2,
        fts_weight=0.0,
        semantic_weight=1.0,
    )

    assert [item.id for item in hybrid] == [item.id for item in semantic]


def test_hybrid_search_order_is_deterministic(storage, monkeypatch) -> None:
    query_vector = _embedding(1.0, 0.0)
    first = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="deterministic alpha",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    second = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="deterministic beta",
        label=SemanticLabel.PATTERN,
        embedding=_embedding(1.0, 0.0),
    )
    storage.store_chunk(first)
    storage.store_chunk(second)

    monkeypatch.setattr("agent_recall.core.retrieve.embed_single", lambda _: np.array(query_vector))

    retriever = Retriever(storage)
    run_a = retriever.search_hybrid(query="deterministic", top_k=2)
    run_b = retriever.search_hybrid(query="deterministic", top_k=2)

    assert [item.id for item in run_a] == [item.id for item in run_b]
