from __future__ import annotations

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
