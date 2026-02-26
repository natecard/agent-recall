"""Benchmarks for embedding pipeline performance.

This module measures:
- Embedding indexing speed at different chunk counts (1K, 10K, 100K)
- Query retrieval latency for hybrid search
- Database disk usage per chunk

Run with: pytest tests/benchmarks/benchmark_embeddings.py -v
Or with benchmark CLI: agent-recall benchmark
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


def _create_chunk(content: str, index: int) -> Chunk:
    """Create a test chunk with deterministic content."""
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=SemanticLabel.PATTERN,
        tags=[f"tag{index % 100}"],
    )


def _generate_sample_text(index: int) -> str:
    """Generate sample text for benchmarking."""
    topics = [
        "authentication",
        "database",
        "api",
        "cache",
        "security",
        "performance",
        "testing",
        "deployment",
        "monitoring",
        "logging",
    ]
    topic = topics[index % len(topics)]
    return f"Benchmark chunk {index} about {topic} with sample content."


@pytest.fixture
def benchmark_db() -> Generator[tuple[Path, SQLiteStorage], None, None]:
    """Create a temporary database for benchmarking.

    Yields a tuple of (db_path, storage) and cleans up after.
    """
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "benchmark.db"

    try:
        storage = SQLiteStorage(db_path)
        yield db_path, storage
    finally:
        shutil.rmtree(temp_dir)


def _populate_chunks(storage: SQLiteStorage, n_chunks: int) -> None:
    """Populate storage with n chunks without embeddings."""
    chunks = [_create_chunk(_generate_sample_text(i), i) for i in range(n_chunks)]
    for chunk in chunks:
        storage.store_chunk(chunk)


@pytest.fixture
def small_db(benchmark_db: tuple[Path, SQLiteStorage]) -> tuple[Path, SQLiteStorage]:
    """Create a database with 1K chunks for benchmarking."""
    db_path, storage = benchmark_db
    _populate_chunks(storage, 1000)
    return db_path, storage


@pytest.fixture
def medium_db(benchmark_db: tuple[Path, SQLiteStorage]) -> tuple[Path, SQLiteStorage]:
    """Create a database with 10K chunks for benchmarking."""
    db_path, storage = benchmark_db
    _populate_chunks(storage, 10000)
    return db_path, storage


def test_benchmark_indexing_speed_1k(benchmark, small_db):
    """Benchmark embedding indexing speed for 1K chunks."""
    db_path, storage = small_db
    indexer = EmbeddingIndexer(storage, batch_size=32)

    benchmark(indexer.index_missing_embeddings)


def test_benchmark_indexing_speed_10k(benchmark, medium_db):
    """Benchmark embedding indexing speed for 10K chunks."""
    db_path, storage = medium_db
    indexer = EmbeddingIndexer(storage, batch_size=32)

    benchmark(indexer.index_missing_embeddings)


def test_benchmark_retrieval_latency_1k(benchmark, small_db):
    """Benchmark retrieval latency for 1K chunks with 100 queries."""
    db_path, storage = small_db

    indexer = EmbeddingIndexer(storage, batch_size=32)
    indexer.index_missing_embeddings()

    retriever = Retriever(storage, backend="hybrid")

    queries = [
        "authentication token",
        "database connection",
        "api endpoint",
        "cache invalidation",
        "security vulnerability",
    ]

    def run_queries():
        for _ in range(20):
            for query in queries:
                retriever.search_hybrid(query, top_k=5)

    benchmark(run_queries)


def test_benchmark_retrieval_latency_10k(benchmark, medium_db):
    """Benchmark retrieval latency for 10K chunks with 100 queries."""
    db_path, storage = medium_db

    indexer = EmbeddingIndexer(storage, batch_size=32)
    indexer.index_missing_embeddings()

    retriever = Retriever(storage, backend="hybrid")

    queries = [
        "authentication token",
        "database connection",
        "api endpoint",
        "cache invalidation",
        "security vulnerability",
    ]

    def run_queries():
        for _ in range(20):
            for query in queries:
                retriever.search_hybrid(query, top_k=5)

    benchmark(run_queries)


def test_benchmark_disk_usage():
    """Benchmark disk usage for different chunk counts.

    This test measures .db file size after indexing embeddings.
    """
    results = {}

    for n_chunks in [1000, 10000]:
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / f"benchmark_{n_chunks}.db"

        try:
            storage = SQLiteStorage(db_path)
            _populate_chunks(storage, n_chunks)

            indexer = EmbeddingIndexer(storage, batch_size=32)
            indexer.index_missing_embeddings()

            db_size_bytes = os.path.getsize(db_path)
            db_size_mb = db_size_bytes / (1024 * 1024)
            size_per_chunk_kb = (db_size_bytes / n_chunks) / 1024

            results[n_chunks] = {
                "size_mb": round(db_size_mb, 2),
                "size_per_chunk_kb": round(size_per_chunk_kb, 2),
            }
        finally:
            shutil.rmtree(temp_dir)

    assert results[1000]["size_mb"] > 0
    assert results[10000]["size_mb"] > 0
    assert results[10000]["size_mb"] > results[1000]["size_mb"]

    return results


if __name__ == "__main__":
    print("Running benchmarks...")
    print("\nDisk Usage Benchmark:")
    results = test_benchmark_disk_usage()
    for n_chunks, data in results.items():
        print(f"  {n_chunks} chunks: {data['size_mb']} MB ({data['size_per_chunk_kb']} KB/chunk)")
