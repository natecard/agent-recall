from __future__ import annotations

from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _chunk(content: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=SemanticLabel.PATTERN,
        embedding=embedding,
    )


def test_save_and_load_embedding_round_trip(storage) -> None:
    chunk = _chunk("auth retries")
    storage.store_chunk(chunk)

    storage.save_embedding(chunk.id, [0.1, 0.2, 0.3], version=2)
    loaded = storage.load_embedding(chunk.id)

    assert loaded == ([0.1, 0.2, 0.3], 2)


def test_get_chunks_without_embeddings_returns_only_pending(storage) -> None:
    pending = _chunk("pending")
    embedded = _chunk("embedded")
    storage.store_chunk(pending)
    storage.store_chunk(embedded)
    storage.save_embedding(embedded.id, [0.5, 0.6], version=1)

    chunks = storage.get_chunks_without_embeddings(limit=10)

    assert [item.id for item in chunks] == [pending.id]


def test_get_embedding_index_status_reports_counts(storage) -> None:
    storage.store_chunk(_chunk("a"))
    item_b = _chunk("b")
    storage.store_chunk(item_b)
    storage.save_embedding(item_b.id, [0.7, 0.8], version=1)

    status = storage.get_embedding_index_status()

    assert status == {"total_chunks": 2, "embedded_chunks": 1, "pending": 1}


def test_get_embedding_index_status_fresh_database(storage) -> None:
    status = storage.get_embedding_index_status()
    assert status == {"total_chunks": 0, "embedded_chunks": 0, "pending": 0}
