from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx

from agent_recall.memory.vector_store import (
    LocalVectorStore,
    TurboPufferVectorStore,
    VectorRecord,
    resolve_local_vector_db_path,
)


def test_local_vector_store_upsert_query_and_prune(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "vectors.db", tenant_id="t1", project_id="p1")
    now = datetime.now(UTC).isoformat()
    first = VectorRecord(
        id="v1",
        tenant_id="t1",
        project_id="p1",
        text="alpha vector",
        label="pattern",
        tags=["alpha"],
        embedding=[1.0, 0.0],
        metadata={},
        updated_at=now,
    )
    second = first.model_copy(
        update={
            "id": "v2",
            "text": "beta vector",
            "tags": ["beta"],
            "embedding": [0.0, 1.0],
        }
    )
    assert store.upsert_records([first, second]) == 2
    results = store.query(embedding=[1.0, 0.0], top_k=1)
    assert results[0][0].id == "v1"

    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_vector_records SET updated_at = ? WHERE id = ?",
            (old, "v2"),
        )
    removed = store.prune_older_than(retention_days=90)
    assert removed == 1


def test_turbopuffer_vector_store_retry_and_namespace(monkeypatch) -> None:
    monkeypatch.setenv("TP_KEY", "token")
    calls = {"count": 0}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"matches": [{"id": "v1", "score": 0.9}]}

    class _Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            _ = (exc_type, exc, tb)

        def request(self, method: str, url: str, headers: dict[str, str], json: dict[str, object]):
            _ = (method, url, headers, json)
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.TransportError("retry once")
            return _Response()

    monkeypatch.setattr("agent_recall.memory.vector_store.httpx.Client", _Client)

    store = TurboPufferVectorStore(
        base_url="https://tp.example",
        api_key_env="TP_KEY",
        tenant_id="tenant-a",
        project_id="project-b",
        retry_attempts=2,
    )
    assert store._namespace() == "tenant-a:project-b"
    matches = store.query(embedding=[0.1, 0.2], top_k=3)
    assert matches[0]["id"] == "v1"
    assert calls["count"] == 2


def test_local_vector_store_migrates_legacy_json_embeddings(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    store = LocalVectorStore(db_path, tenant_id="t1", project_id="p1")
    now = datetime.now(UTC).isoformat()
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO memory_vector_records (
                id, tenant_id, project_id, text, label, tags, embedding, metadata, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy",
                "t1",
                "p1",
                "legacy vector",
                "pattern",
                json.dumps(["legacy"]),
                json.dumps([1.0, 0.0]),
                json.dumps({}),
                now,
            ),
        )

    migrated = LocalVectorStore(db_path, tenant_id="t1", project_id="p1")
    results = migrated.query(embedding=[1.0, 0.0], top_k=1)

    assert results[0][0].id == "legacy"
    assert results[0][0].embedding == [1.0, 0.0]


def test_resolve_local_vector_db_path_prefers_semantic_memory_name(tmp_path) -> None:
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    legacy = agent_dir / "vector.db"
    legacy.write_text("placeholder", encoding="utf-8")

    resolved = resolve_local_vector_db_path(agent_dir)

    assert resolved == agent_dir / "semantic-memory.db"
    assert resolved.exists()
