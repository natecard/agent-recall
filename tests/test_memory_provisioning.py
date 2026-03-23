from __future__ import annotations

import numpy as np

from agent_recall.memory.migration import VectorMigrationRequest
from agent_recall.memory.provisioning import VectorMemoryService, VectorMemorySetupRequest
from agent_recall.storage.models import Chunk, ChunkSource, LogEntry, LogSource, SemanticLabel


def test_vector_memory_service_provision_local(monkeypatch, storage, files) -> None:
    storage.store_chunk(
        Chunk(
            source=ChunkSource.MANUAL,
            source_ids=[],
            content="Provisioned vector memory chunk",
            label=SemanticLabel.PATTERN,
            tags=["memory"],
        )
    )
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id="vector-setup-session",
            content="Provisioned vector memory learning",
            label=SemanticLabel.PATTERN,
            tags=["memory"],
        )
    )

    monkeypatch.setattr(
        "agent_recall.memory.local_model.LocalEmbeddingModelManager.ensure_available",
        lambda self, verify=True: {  # noqa: ARG005
            "state": "ready",
            "model_name": "all-MiniLM-L6-v2",
        },
    )
    monkeypatch.setattr(
        "agent_recall.memory.local_model.LocalEmbeddingModelManager.inspect",
        lambda self: {  # noqa: ARG005
            "state": "ready",
            "model_name": "all-MiniLM-L6-v2",
        },
    )
    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.embed_batch_to_lists",
        lambda texts: [[0.25] * 384 for _ in texts],
    )
    monkeypatch.setattr(
        "agent_recall.memory.embedding_provider.embed_single",
        lambda _text: np.array([0.25] * 384),
    )

    service = VectorMemoryService(storage=storage, files=files)
    payload = service.provision(
        VectorMemorySetupRequest(
            enabled=True,
            backend="local",
            local_model_name="all-MiniLM-L6-v2",
        )
    )

    assert payload["status"] == "ready"
    assert payload["backend"] == "local"
    assert payload["embedding_indexing"]["indexed"] == 1
    status = service.status()
    assert status["enabled"] is True
    assert status["vector_store"]["record_count"] >= 1


def test_vector_memory_service_sync_vectors_skips_when_disabled(storage, files) -> None:
    service = VectorMemoryService(storage=storage, files=files)

    payload = service.sync_vectors(
        request=VectorMigrationRequest(dry_run=True),
        trigger="sync",
    )

    assert payload["status"] == "skipped"
    assert payload["reason"] == "vector_memory_disabled"


def test_vector_memory_service_provision_avoids_second_model_ensure(
    monkeypatch, storage, files
) -> None:
    calls = {"ensure": 0, "sync": 0}

    def fake_ensure(self, verify=True):  # noqa: ANN001, ARG001
        calls["ensure"] += 1
        return {"state": "ready", "model_name": "all-MiniLM-L6-v2"}

    def fake_sync_impl(
        self, request, *, trigger, honor_feature_flag, persist_status, ensure_local_model
    ):  # noqa: ANN001
        calls["sync"] += 1
        assert trigger == "setup"
        assert honor_feature_flag is False
        assert persist_status is False
        assert ensure_local_model is False
        return {"status": "ready", "rows_written": 0}

    monkeypatch.setattr(
        "agent_recall.memory.local_model.LocalEmbeddingModelManager.ensure_available",
        fake_ensure,
    )
    monkeypatch.setattr(
        "agent_recall.core.embedding_indexer.EmbeddingIndexer.index_missing_embeddings",
        lambda self: {"indexed": 0, "skipped": 0},  # noqa: ARG005
    )
    monkeypatch.setattr(
        VectorMemoryService,
        "_sync_vectors_impl",
        fake_sync_impl,
    )

    service = VectorMemoryService(storage=storage, files=files)
    payload = service.provision(
        VectorMemorySetupRequest(
            enabled=True,
            backend="local",
            local_model_name="all-MiniLM-L6-v2",
        )
    )

    assert payload["status"] == "ready"
    assert calls["ensure"] == 1
    assert calls["sync"] == 1
