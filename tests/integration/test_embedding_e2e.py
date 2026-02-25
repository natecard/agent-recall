"""End-to-end integration tests for the embedding workflow.

These tests verify the complete embedding pipeline from ingestion through retrieval,
ensuring all components integrate properly. Uses real embeddings (not mocked).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.core.retrieve import Retriever
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


def _create_chunk(content: str, label: SemanticLabel = SemanticLabel.PATTERN) -> Chunk:
    """Helper to create a Chunk with default values."""
    return Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content=content,
        label=label,
    )


def _is_relevant_to_query(chunk: Chunk, query: str) -> bool:
    """Simple relevance check: query terms should appear in chunk content or tags."""
    query_lower = query.lower()
    content_lower = chunk.content.lower()
    tags_lower = " ".join(chunk.tags).lower()

    query_terms = set(query_lower.split())
    content_terms = set(content_lower.split())
    tags_terms = set(tags_lower.split())

    return bool(query_terms & (content_terms | tags_terms))


@pytest.fixture
def test_db() -> Generator[SQLiteStorage, None, None]:
    """Create a temporary SQLite database for testing.

    Yields the storage instance and cleans up after the test.
    """
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_embeddings.db"

    try:
        storage = SQLiteStorage(db_path)
        yield storage
    finally:
        shutil.rmtree(temp_dir)


def test_embedding_pipeline_end_to_end(test_db: SQLiteStorage) -> None:
    """Test the complete embedding workflow from ingestion through retrieval.

    This test:
    1. Ingests 20 test chunks covering auth, API, and database topics
    2. Verifies no embeddings exist initially
    3. Runs the indexer to generate embeddings
    4. Verifies all chunks are embedded
    5. Queries via hybrid search
    6. Verifies results are semantically correct
    """
    start_time = time.time()

    chunks = [
        # Auth-related chunks (7 chunks)
        _create_chunk(
            "JWT token validation failed due to expired signature", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "OAuth2 authorization code flow with PKCE for secure authentication",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "Implement refresh token rotation to prevent token reuse attacks", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "API key authentication using Bearer token in Authorization header",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "Multi-factor authentication (MFA) setup with TOTP or SMS codes", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "Role-based access control (RBAC) for user permissions", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "Session management with secure cookie flags (HttpOnly, Secure)", SemanticLabel.PATTERN
        ),
        # API-related chunks (7 chunks)
        _create_chunk(
            "RESTful API design with proper HTTP methods (GET, POST, PUT, DELETE)",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "GraphQL API schema design with resolvers and data loaders", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "Rate limiting middleware to prevent API abuse and throttling", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "API versioning strategies using URL path or header-based approach",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "Webhooks implementation for asynchronous event notifications", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "API request/response logging and monitoring for observability", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "API error handling with standardized error codes and messages", SemanticLabel.PATTERN
        ),
        # Database-related chunks (6 chunks)
        _create_chunk(
            "Database indexing strategies for query performance optimization", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "ACID transaction properties and isolation levels in relational databases",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "Database connection pooling to manage concurrent connections efficiently",
            SemanticLabel.PATTERN,
        ),
        _create_chunk(
            "SQL query optimization using EXPLAIN and query execution plans", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "Database sharding and replication for horizontal scalability", SemanticLabel.PATTERN
        ),
        _create_chunk(
            "ORM query performance: N+1 problem and eager loading solutions", SemanticLabel.PATTERN
        ),
    ]

    for chunk in chunks:
        test_db.store_chunk(chunk)

    pending_chunks = test_db.get_chunks_without_embeddings(limit=100)
    assert len(pending_chunks) == 20, f"Expected 20 pending chunks, got {len(pending_chunks)}"

    embedding_start = time.time()
    indexer = EmbeddingIndexer(test_db, batch_size=32)
    stats = indexer.index_missing_embeddings()
    embedding_time = time.time() - embedding_start

    assert stats == {"indexed": 20, "skipped": 0}, (
        f"Expected stats={{'indexed': 20, 'skipped': 0}}, got {stats}"
    )

    pending_chunks_after = test_db.get_chunks_without_embeddings(limit=100)
    assert pending_chunks_after == [], (
        f"Expected no pending chunks after indexing, got {len(pending_chunks_after)}"
    )

    retriever = Retriever(test_db, backend="hybrid", fusion_k=60)

    jwt_results = retriever.search_hybrid("JWT authentication", top_k=10)
    assert len(jwt_results) > 0, "Expected at least one result for JWT query"

    auth_keywords = {
        "jwt",
        "token",
        "authentication",
        "oauth",
        "authorization",
        "mfa",
        "session",
        "access",
    }
    auth_relevant = sum(
        1 for chunk in jwt_results if auth_keywords & set(chunk.content.lower().split())
    )
    assert auth_relevant >= 3, f"Expected at least 3 auth-related results, got {auth_relevant}"

    db_results = retriever.search_hybrid("database optimization", top_k=10)
    assert len(db_results) > 0, "Expected at least one result for database query"

    db_keywords = {"database", "sql", "query", "index", "transaction", "sharding", "replication"}
    db_relevant = sum(1 for chunk in db_results if db_keywords & set(chunk.content.lower().split()))
    assert db_relevant >= 3, f"Expected at least 3 database-related results, got {db_relevant}"

    jwt_semantic_results = retriever.search_by_vector_similarity("JWT authentication", top_k=5)
    if jwt_semantic_results:
        first_chunk = jwt_semantic_results[0]
        query_terms = set("jwt authentication token".lower().split())
        content_terms = set(first_chunk.content.lower().split())
        assert len(query_terms & content_terms) > 0, "Top result should contain query-related terms"

    db_semantic_results = retriever.search_by_vector_similarity("database optimization", top_k=5)
    if db_semantic_results:
        first_chunk = db_semantic_results[0]
        query_terms = set("database optimization".lower().split())
        content_terms = set(first_chunk.content.lower().split())
        assert len(query_terms & content_terms) > 0, "Top result should contain query-related terms"

    top_3_relevant = sum(
        1 for chunk in jwt_results[:3] if _is_relevant_to_query(chunk, "JWT authentication")
    )
    assert top_3_relevant >= 2, (
        f"Expected at least 2 of top 3 results to be relevant, got {top_3_relevant}"
    )

    total_time = time.time() - start_time
    assert total_time < 60, f"Total execution time {total_time:.2f}s exceeded 60s limit"

    assert embedding_time < 30, f"Embedding phase {embedding_time:.2f}s exceeded 30s limit"
