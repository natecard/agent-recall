from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_recall.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_agent_dir() -> Iterator[Path]:
    """Create a temporary .agent directory for testing."""
    temp_dir = Path(tempfile.mkdtemp()) / ".agent"
    temp_dir.mkdir()
    (temp_dir / "logs").mkdir()
    (temp_dir / "archive").mkdir()
    (temp_dir / "GUARDRAILS.md").write_text("# Guardrails\n")
    (temp_dir / "STYLE.md").write_text("# Style\n")
    (temp_dir / "RECENT.md").write_text("# Recent\n")
    (temp_dir / "config.yaml").write_text("llm:\n  provider: openai\n  model: gpt-4o-mini\n")

    yield temp_dir

    shutil.rmtree(temp_dir.parent)


@pytest.fixture
def mock_storage(temp_agent_dir: Path) -> SQLiteStorage:
    """Create a SQLiteStorage instance for testing."""
    return SQLiteStorage(temp_agent_dir / "state.db")


@pytest.fixture
def sample_chunks() -> list:
    """Create a sample 10-chunk dataset with predetermined embeddings for testing."""
    from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel

    chunks = []
    topics = [
        "JWT authentication and token validation",
        "Database connection pooling",
        "API rate limiting strategies",
        "Cache invalidation patterns",
        "Error handling best practices",
        "Unit testing frameworks",
        "CI/CD pipeline configuration",
        "Microservice communication",
        "Security vulnerability scanning",
        "Performance optimization techniques",
    ]

    for i, topic in enumerate(topics):
        embedding = [0.0] * 384
        embedding[i % 384] = 1.0
        chunks.append(
            Chunk(
                source=ChunkSource.COMPACTION,
                source_ids=[],
                content=topic,
                label=SemanticLabel.PATTERN,
                tags=["test"],
                embedding=embedding,
            )
        )

    return chunks


@pytest.fixture
def mock_embedder():
    """Create a mock embedder for testing without downloading actual models."""
    from unittest.mock import MagicMock

    import numpy as np

    mock = MagicMock()
    mock.encode = lambda texts, convert_to_numpy=True: np.array(
        [[float(i)] * 384 for i, _ in enumerate(texts)],
        dtype=np.float32,
    )
    return mock
