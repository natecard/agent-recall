from __future__ import annotations

from uuid import uuid4

from agent_recall.storage import create_storage_backend
from agent_recall.storage.models import (
    AgentRecallConfig,
    Chunk,
    ChunkSource,
    SemanticLabel,
)
from agent_recall.storage.sqlite import SQLiteStorage
from agent_recall.storage.utils import replicate_chunks_to


def test_replicate_chunks_to_returns_count(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    chunk = Chunk(
        id=uuid4(),
        source=ChunkSource.MANUAL,
        content="Test chunk",
        label=SemanticLabel.PATTERN,
    )
    source.store_chunk(chunk)

    count = replicate_chunks_to(source, target)

    assert count == 1


def test_replicate_chunks_to_preserves_content(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    chunk = Chunk(
        id=uuid4(),
        source=ChunkSource.MANUAL,
        content="Test chunk content",
        label=SemanticLabel.PATTERN,
        tags=["test-tag"],
    )
    source.store_chunk(chunk)

    replicate_chunks_to(source, target)

    target_chunks = target.list_chunks()
    assert len(target_chunks) == 1
    assert target_chunks[0].content == "Test chunk content"
    assert target_chunks[0].label == SemanticLabel.PATTERN
    assert target_chunks[0].tags == ["test-tag"]


def test_replicate_chunks_to_preserves_embeddings(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    chunk = Chunk(
        id=uuid4(),
        source=ChunkSource.MANUAL,
        content="Test chunk with embedding",
        label=SemanticLabel.PATTERN,
        embedding=[0.1, 0.2, 0.3],
    )
    source.store_chunk(chunk)

    replicate_chunks_to(source, target)

    target_chunks = target.list_chunks_with_embeddings()
    assert len(target_chunks) == 1
    assert target_chunks[0].embedding == [0.1, 0.2, 0.3]


def test_replicate_chunks_to_handles_multiple_chunks(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    for i in range(5):
        chunk = Chunk(
            id=uuid4(),
            source=ChunkSource.MANUAL,
            content=f"Test chunk {i}",
            label=SemanticLabel.PATTERN,
        )
        source.store_chunk(chunk)

    count = replicate_chunks_to(source, target)

    assert count == 5
    assert target.count_chunks() == 5


def test_replicate_chunks_to_handles_empty_source(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    count = replicate_chunks_to(source, target)

    assert count == 0
    assert target.count_chunks() == 0


def test_replicate_chunks_to_with_different_labels(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")
    target = SQLiteStorage(tmp_path / "target.db", tenant_id="test", project_id="test")

    labels = [
        SemanticLabel.PATTERN,
        SemanticLabel.GOTCHA,
        SemanticLabel.DECISION_RATIONALE,
        SemanticLabel.EXPLORATION,
    ]
    for i, label in enumerate(labels):
        chunk = Chunk(
            id=uuid4(),
            source=ChunkSource.MANUAL,
            content=f"Chunk {i}",
            label=label,
        )
        source.store_chunk(chunk)

    replicate_chunks_to(source, target)

    target_chunks = target.list_chunks()
    assert len(target_chunks) == 4
    target_labels = {c.label for c in target_chunks}
    assert target_labels == set(labels)


def test_replicate_chunks_to_works_with_shared_storage(tmp_path) -> None:
    source = SQLiteStorage(tmp_path / "source.db", tenant_id="test", project_id="test")

    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {
                    "base_url": f"file://{tmp_path / 'shared-memory'}",
                    "tenant_id": "test-tenant",
                    "project_id": "test-project",
                },
            }
        }
    )
    target = create_storage_backend(config, tmp_path / "target.db")

    chunk = Chunk(
        id=uuid4(),
        source=ChunkSource.MANUAL,
        content="Test chunk",
        label=SemanticLabel.PATTERN,
    )
    source.store_chunk(chunk)

    count = replicate_chunks_to(source, target)

    assert count == 1
    assert target.count_chunks() == 1
