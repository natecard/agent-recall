from __future__ import annotations

from uuid import UUID

from agent_recall.core.embeddings import generate_embedding
from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def test_retrieval_fts(storage) -> None:
    chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Use optimistic locking for account updates",
        label=SemanticLabel.PATTERN,
        tags=["db", "concurrency"],
    )
    storage.store_chunk(chunk)

    retriever = Retriever(storage)
    results = retriever.search("optimistic locking", top_k=5)

    assert len(results) == 1
    assert results[0].content == chunk.content


def test_retrieval_preserves_chunk_embedding(storage) -> None:
    chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="Use transactional outbox pattern for event publishing",
        label=SemanticLabel.PATTERN,
        tags=["architecture"],
        embedding=[0.2, -0.1, 0.5],
    )
    storage.store_chunk(chunk)

    retriever = Retriever(storage)
    results = retriever.search("transactional outbox", top_k=5)

    assert len(results) == 1
    assert results[0].embedding == [0.2, -0.1, 0.5]


def test_retrieval_hybrid_includes_vector_only_match(storage) -> None:
    query = "resilient sync retries"

    fts_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="resilient sync retries should remain bounded",
        label=SemanticLabel.PATTERN,
        tags=["ops"],
    )
    vector_chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="unrelated surface text",
        label=SemanticLabel.PATTERN,
        tags=["ops"],
        embedding=generate_embedding(query, dimensions=16),
    )
    storage.store_chunk(fts_chunk)
    storage.store_chunk(vector_chunk)

    retriever = Retriever(storage)
    results = retriever.search(query, top_k=5, backend="hybrid")

    assert any(chunk.id == fts_chunk.id for chunk in results)
    assert any(chunk.id == vector_chunk.id for chunk in results)


def test_retrieval_hybrid_tie_breaks_deterministically(storage) -> None:
    query = "deterministic ordering"
    shared_embedding = generate_embedding(query, dimensions=16)

    first = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="alpha",
        label=SemanticLabel.PATTERN,
        embedding=shared_embedding,
    )
    second = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000002"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="beta",
        label=SemanticLabel.PATTERN,
        embedding=shared_embedding,
    )
    storage.store_chunk(second)
    storage.store_chunk(first)

    retriever = Retriever(storage)
    results = retriever.search(query, top_k=2, backend="hybrid")

    assert [chunk.id for chunk in results] == [first.id, second.id]
