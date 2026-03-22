from __future__ import annotations

import pytest

from agent_recall.memory.migration import VectorMigrationRequest, VectorMigrationService
from agent_recall.memory.policy import MemoryPolicy
from agent_recall.memory.vector_store import LocalVectorStore
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def _memory_cfg() -> dict[str, object]:
    return {
        "embedding_provider": "local",
        "vector_backend": "local",
        "migration_batch_size": 100,
        "cost": {"max_vector_records": 20},
        "privacy": {
            "redaction_patterns": ["token-[0-9]+"],
            "retention_days": 30,
        },
    }


def test_vector_migration_service_dry_run(storage, files) -> None:
    files.write_tier(
        KnowledgeTier.GUARDRAILS,
        "# Guardrails\n\n- Never commit token-123 values\n",
    )
    storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="Use token-999 safely in local tests",
            label=SemanticLabel.PATTERN,
            tags=["auth"],
        )
    )
    memory_cfg = _memory_cfg()
    service = VectorMigrationService(
        storage=storage,
        files=files,
        memory_cfg=memory_cfg,
        policy=MemoryPolicy.from_memory_config(memory_cfg),
        tenant_id="tenant-test",
        project_id="project-test",
        embedding_dimensions=64,
        vector_db_path=files.agent_dir / "vector.db",
    )

    payload = service.migrate(VectorMigrationRequest(dry_run=True))
    assert payload["rows_discovered"] >= 2
    assert payload["rows_written"] == 0
    assert payload["redacted_rows"] >= 1


def test_vector_migration_service_writes_local_vectors(storage, files) -> None:
    storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="Migration vector content",
            label=SemanticLabel.PATTERN,
            tags=["migration"],
        )
    )
    memory_cfg = _memory_cfg()
    service = VectorMigrationService(
        storage=storage,
        files=files,
        memory_cfg=memory_cfg,
        policy=MemoryPolicy.from_memory_config(memory_cfg),
        tenant_id="tenant-test",
        project_id="project-test",
        embedding_dimensions=64,
        vector_db_path=files.agent_dir / "vector.db",
    )

    payload = service.migrate(VectorMigrationRequest(dry_run=False, max_records=1))
    assert payload["rows_migrated"] == 1
    assert payload["rows_written"] == 1

    vector_store = LocalVectorStore(
        files.agent_dir / "vector.db",
        tenant_id="tenant-test",
        project_id="project-test",
    )
    assert len(vector_store.list_records(limit=10)) == 1


def test_vector_migration_service_requires_external_base_url(storage, files) -> None:
    memory_cfg = {
        **_memory_cfg(),
        "embedding_provider": "external",
        "external_embedding_base_url": "",
    }
    service = VectorMigrationService(
        storage=storage,
        files=files,
        memory_cfg=memory_cfg,
        policy=MemoryPolicy.from_memory_config(memory_cfg),
        tenant_id="tenant-test",
        project_id="project-test",
        embedding_dimensions=64,
        vector_db_path=files.agent_dir / "vector.db",
    )

    with pytest.raises(ValueError, match="external_embedding_base_url"):
        service.migrate(VectorMigrationRequest(dry_run=True))
