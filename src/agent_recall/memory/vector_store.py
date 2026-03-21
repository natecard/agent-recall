from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    id: str
    tenant_id: str
    project_id: str
    text: str
    label: str
    tags: list[str] = Field(default_factory=list)
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: str


class LocalVectorStore:
    """Deterministic local vector store schema backed by SQLite."""

    _schema = """
    CREATE TABLE IF NOT EXISTS memory_vector_records (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        text TEXT NOT NULL,
        label TEXT NOT NULL,
        tags TEXT NOT NULL,
        embedding TEXT NOT NULL,
        metadata TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_vector_scope
    ON memory_vector_records(tenant_id, project_id, updated_at DESC);
    """

    def __init__(self, db_path: Path, *, tenant_id: str = "default", project_id: str = "default"):
        self.db_path = db_path
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self._schema)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_records(self, records: list[VectorRecord]) -> int:
        written = 0
        with self._connect() as conn:
            for record in records:
                conn.execute(
                    """
                    INSERT INTO memory_vector_records (
                        id, tenant_id, project_id, text, label,
                        tags, embedding, metadata, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        label = excluded.label,
                        tags = excluded.tags,
                        embedding = excluded.embedding,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at
                    """,
                    (
                        record.id,
                        record.tenant_id,
                        record.project_id,
                        record.text,
                        record.label,
                        json.dumps(record.tags, separators=(",", ":")),
                        json.dumps(record.embedding, separators=(",", ":")),
                        json.dumps(record.metadata, separators=(",", ":")),
                        record.updated_at,
                    ),
                )
                written += 1
        return written

    def list_records(self, *, limit: int = 500) -> list[VectorRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memory_vector_records
                WHERE tenant_id = ? AND project_id = ?
                ORDER BY updated_at DESC, id ASC
                LIMIT ?
                """,
                (self.tenant_id, self.project_id, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def query(self, *, embedding: list[float], top_k: int = 10) -> list[tuple[VectorRecord, float]]:
        candidates = self.list_records(limit=max(top_k * 10, 200))
        scored: list[tuple[VectorRecord, float]] = []
        for record in candidates:
            score = _cosine_similarity(embedding, record.embedding)
            scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[: max(1, int(top_k))]

    def prune_older_than(self, *, retention_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=max(0, int(retention_days)))
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memory_vector_records
                WHERE tenant_id = ? AND project_id = ? AND updated_at < ?
                """,
                (self.tenant_id, self.project_id, cutoff.isoformat()),
            )
            return int(cursor.rowcount or 0)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> VectorRecord:
        tags = json.loads(str(row["tags"])) if row["tags"] else []
        embedding = json.loads(str(row["embedding"])) if row["embedding"] else []
        metadata = json.loads(str(row["metadata"])) if row["metadata"] else {}
        return VectorRecord(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            project_id=str(row["project_id"]),
            text=str(row["text"]),
            label=str(row["label"]),
            tags=tags if isinstance(tags, list) else [],
            embedding=embedding if isinstance(embedding, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
            updated_at=str(row["updated_at"]),
        )


class TurboPufferVectorStore:
    """TurboPuffer adapter with auth, tenancy mapping, and retry policy."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        tenant_id: str,
        project_id: str,
        timeout_seconds: float = 10.0,
        retry_attempts: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.retry_attempts = max(1, int(retry_attempts))

    def _headers(self) -> dict[str, str]:
        token = os.getenv(self.api_key_env, "").strip()
        if not token:
            raise RuntimeError(f"Missing TurboPuffer API key in env var {self.api_key_env}")
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _namespace(self) -> str:
        return f"{self.tenant_id}:{self.project_id}"

    def upsert_records(self, records: list[VectorRecord]) -> int:
        payload = {
            "namespace": self._namespace(),
            "records": [record.model_dump() for record in records],
        }
        self._request("POST", "/vectors/upsert", payload)
        return len(records)

    def query(self, *, embedding: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        payload = {
            "namespace": self._namespace(),
            "embedding": embedding,
            "top_k": max(1, int(top_k)),
        }
        response = self._request("POST", "/vectors/query", payload)
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("matches"), list):
            return [item for item in data["matches"] if isinstance(item, dict)]
        return []

    def _request(self, method: str, path: str, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.request(
                        method,
                        f"{self.base_url}{path}",
                        headers=self._headers(),
                        json=payload,
                    )
                response.raise_for_status()
                return response
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retry_attempts:
                    time.sleep(0.2 * attempt)
                    continue
                break
        raise RuntimeError(f"TurboPuffer request failed: {last_error}") from last_error


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
