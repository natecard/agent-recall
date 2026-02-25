"""Tests for the embedding indexer module.

This module contains comprehensive unit tests for the EmbeddingIndexer class,
including tests for batch processing, max_chunks limits, progress bar display,
and various edge cases. Tests use mocked embedder to avoid external dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _make_chunk(
    content: str,
    embedding: list[float] | None = None,
    label: SemanticLabel = SemanticLabel.PATTERN,
) -> Chunk:
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=label,
        tags=[],
        embedding=embedding,
    )


def test_index_missing_embeddings_processes_unembedded_chunks(mock_storage, monkeypatch) -> None:
    """Test that index_missing_embeddings processes all chunks without embeddings."""
    mock_storage.store_chunk(_make_chunk("alpha"))
    mock_storage.store_chunk(_make_chunk("beta"))
    mock_storage.store_chunk(_make_chunk("gamma"))

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.5] * 384 for _ in texts],
    )

    indexer = EmbeddingIndexer(mock_storage, batch_size=10)
    result = indexer.index_missing_embeddings()

    assert result["indexed"] == 3
    assert result["skipped"] == 0
    embedded = mock_storage.list_chunks_with_embeddings()
    assert len(embedded) == 3
    assert all(len(chunk.embedding or []) == 384 for chunk in embedded)


def test_batch_size_is_respected(mock_storage, monkeypatch) -> None:
    """Test that the batch_size parameter is correctly respected during indexing."""
    for i in range(5):
        mock_storage.store_chunk(_make_chunk(f"chunk-{i}"))

    calls: list[list[str]] = []

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        _fake_embed,
    )

    indexer = EmbeddingIndexer(mock_storage, batch_size=2)
    result = indexer.index_missing_embeddings()

    assert result["indexed"] == 5
    assert result["skipped"] == 0
    assert [len(batch) for batch in calls] == [2, 2, 1]


def test_index_missing_embeddings_returns_correct_stats(mock_storage, monkeypatch) -> None:
    """Test that indexing returns correct statistics about indexed and skipped chunks."""
    mock_storage.store_chunk(_make_chunk("first"))
    mock_storage.store_chunk(_make_chunk("second", embedding=[0.1] * 384))
    mock_storage.store_chunk(_make_chunk("third"))

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.2] * 384 for _ in texts],
    )

    indexer = EmbeddingIndexer(mock_storage)
    result = indexer.index_missing_embeddings()

    assert result == {"indexed": 2, "skipped": 0}


def test_already_embedded_chunks_are_skipped(mock_storage, monkeypatch) -> None:
    """Test that chunks that already have embeddings are skipped during indexing."""
    mock_storage.store_chunk(_make_chunk("already_embedded", embedding=[0.1] * 384))
    mock_storage.store_chunk(_make_chunk("needs_embedding"))
    mock_storage.store_chunk(_make_chunk("also_embedded", embedding=[0.2] * 384))

    embed_calls: list[int] = []

    def _track_embed(texts: list[str]) -> list[list[float]]:
        embed_calls.append(len(texts))
        return [[0.5] * 384 for _ in texts]

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        _track_embed,
    )

    indexer = EmbeddingIndexer(mock_storage)
    result = indexer.index_missing_embeddings()

    assert result["indexed"] == 1
    assert result["skipped"] == 0
    assert sum(embed_calls) == 1


def test_max_chunks_limit_is_respected(mock_storage, monkeypatch) -> None:
    """Test that max_chunks parameter correctly limits the number of chunks indexed."""
    for i in range(10):
        mock_storage.store_chunk(_make_chunk(f"chunk-{i}"))

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.3] * 384 for _ in texts],
    )

    indexer = EmbeddingIndexer(mock_storage, batch_size=10)
    result = indexer.index_missing_embeddings(max_chunks=5)

    assert result["indexed"] == 5
    assert result["skipped"] == 0
    assert len(mock_storage.list_chunks_with_embeddings()) == 5


def test_progress_bar_displays(mock_storage, monkeypatch) -> None:
    """Test that progress bar is displayed when tqdm is available."""
    mock_storage.store_chunk(_make_chunk("chunk1"))
    mock_storage.store_chunk(_make_chunk("chunk2"))

    class MockProgress:
        def __init__(self, total: int, desc: str) -> None:
            self.total = total
            self.desc = desc
            self.updated = 0

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def update(self, amount: int = 1) -> None:
            self.updated += amount

    mock_progress = MockProgress(total=2, desc="Embedding chunks")

    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.5] * 384 for _ in texts],
    )

    with patch("tqdm.auto.tqdm", return_value=mock_progress):
        indexer = EmbeddingIndexer(mock_storage)
        result = indexer.index_missing_embeddings()

    assert result["indexed"] == 2
    assert mock_progress.updated == 2


def test_indexing_with_no_chunks_returns_zero_stats(mock_storage) -> None:
    """Test that indexing an empty storage returns zero indexed/skipped."""
    indexer = EmbeddingIndexer(mock_storage)
    result = indexer.index_missing_embeddings()

    assert result == {"indexed": 0, "skipped": 0}


def test_get_indexing_stats_empty_storage(mock_storage) -> None:
    """Test that get_indexing_stats returns correct stats for empty storage."""
    stats = EmbeddingIndexer(mock_storage).get_indexing_stats()

    assert stats["total_chunks"] == 0
    assert stats["embedded_chunks"] == 0
    assert stats["pending"] == 0


def test_get_indexing_stats_mixed_chunks(mock_storage) -> None:
    """Test that get_indexing_stats correctly counts embedded vs pending chunks."""
    mock_storage.store_chunk(_make_chunk("embedded1", embedding=[0.1] * 384))
    mock_storage.store_chunk(_make_chunk("embedded2", embedding=[0.2] * 384))
    mock_storage.store_chunk(_make_chunk("pending1"))
    mock_storage.store_chunk(_make_chunk("pending2"))
    mock_storage.store_chunk(_make_chunk("pending3"))

    stats = EmbeddingIndexer(mock_storage).get_indexing_stats()

    assert stats["total_chunks"] == 5
    assert stats["embedded_chunks"] == 2
    assert stats["pending"] == 3
