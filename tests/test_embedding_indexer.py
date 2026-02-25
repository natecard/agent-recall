from __future__ import annotations

from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _make_chunk(content: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=SemanticLabel.PATTERN,
        embedding=embedding,
    )


def test_index_missing_embeddings_processes_all_missing(storage, monkeypatch) -> None:
    storage.store_chunk(_make_chunk("alpha"))
    storage.store_chunk(_make_chunk("beta"))

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.5] * 384 for _ in texts],
    )

    indexer = EmbeddingIndexer(storage, batch_size=10)
    result = indexer.index_missing_embeddings()

    assert result == {"indexed": 2, "skipped": 0}
    embedded = storage.list_chunks_with_embeddings()
    assert len(embedded) == 2
    assert all(len(chunk.embedding or []) == 384 for chunk in embedded)


def test_index_missing_embeddings_respects_batch_size(storage, monkeypatch) -> None:
    for index in range(5):
        storage.store_chunk(_make_chunk(f"chunk-{index}"))

    calls: list[list[str]] = []

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr("agent_recall.core.embedding_indexer.embed_batch_to_lists", _fake_embed)

    indexer = EmbeddingIndexer(storage, batch_size=2)
    result = indexer.index_missing_embeddings()

    assert result == {"indexed": 5, "skipped": 0}
    assert [len(batch) for batch in calls] == [2, 2, 1]


def test_index_missing_embeddings_respects_max_chunks(storage, monkeypatch) -> None:
    for index in range(6):
        storage.store_chunk(_make_chunk(f"chunk-{index}"))

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.2] * 384 for _ in texts],
    )

    indexer = EmbeddingIndexer(storage, batch_size=4)
    result = indexer.index_missing_embeddings(max_chunks=3)

    assert result == {"indexed": 3, "skipped": 0}
    assert len(storage.list_chunks_with_embeddings()) == 3


def test_index_missing_embeddings_noop_when_no_pending(storage) -> None:
    storage.store_chunk(_make_chunk("existing", embedding=[0.1, 0.2, 0.3]))

    indexer = EmbeddingIndexer(storage)
    result = indexer.index_missing_embeddings()

    assert result == {"indexed": 0, "skipped": 0}


def test_get_indexing_stats_reports_total_embedded_pending(storage) -> None:
    storage.store_chunk(_make_chunk("a", embedding=[0.1, 0.2, 0.3]))
    storage.store_chunk(_make_chunk("b"))
    storage.store_chunk(_make_chunk("c"))

    stats = EmbeddingIndexer(storage).get_indexing_stats()

    assert stats == {"total_chunks": 3, "embedded_chunks": 1, "pending": 2}
