from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID, uuid4

from agent_recall.storage.metadata import FeedbackMetadata
from agent_recall.storage.normalize import (
    dump_json_compact,
    normalize_limit,
    normalize_non_empty_text,
)


def retrieval_query_hash(query: str) -> str:
    normalized = " ".join(query.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def record_retrieval_feedback(
    storage: Any,
    *,
    query: str,
    chunk_id: UUID,
    score: int,
    actor: str = "user",
    source: str = "cli",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_query = normalize_non_empty_text(query)
    if not normalized_query:
        raise ValueError("query is required")
    normalized_score = int(score)
    if normalized_score not in {-1, 1}:
        raise ValueError("score must be -1 (downvote) or 1 (upvote)")
    chunk_id_text = str(chunk_id)
    with storage._connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM chunks WHERE id = ? AND tenant_id = ? AND project_id = ? LIMIT 1",
            (chunk_id_text, storage.tenant_id, storage.project_id),
        ).fetchone()
        if not exists:
            raise ValueError(f"chunk not found for feedback: {chunk_id_text}")

        now = storage._now_iso()
        metadata_payload = FeedbackMetadata.from_value(metadata).to_dict()
        payload = {
            "id": str(uuid4()),
            "tenant_id": storage.tenant_id,
            "project_id": storage.project_id,
            "query_text": normalized_query,
            "query_hash": retrieval_query_hash(normalized_query),
            "chunk_id": chunk_id_text,
            "score": normalized_score,
            "actor": normalize_non_empty_text(actor) or "user",
            "source": normalize_non_empty_text(source) or "cli",
            "metadata": metadata_payload,
            "created_at": now,
            "updated_at": now,
        }
        conn.execute(
            """
            INSERT INTO retrieval_feedback (
                id, tenant_id, project_id, query_text, query_hash, chunk_id, score,
                actor, source, metadata, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["tenant_id"],
                payload["project_id"],
                payload["query_text"],
                payload["query_hash"],
                payload["chunk_id"],
                payload["score"],
                payload["actor"],
                payload["source"],
                dump_json_compact(payload["metadata"]),
                payload["created_at"],
                payload["updated_at"],
            ),
        )
    return payload


def list_retrieval_feedback(
    storage: Any,
    *,
    limit: int = 100,
    query: str | None = None,
    chunk_id: UUID | None = None,
) -> list[dict[str, Any]]:
    filters: list[Any] = [storage.tenant_id, storage.project_id]
    sql = (
        "SELECT id, query_text, query_hash, chunk_id, score, actor, source, metadata, "
        "created_at, updated_at "
        "FROM retrieval_feedback "
        "WHERE tenant_id = ? AND project_id = ?"
    )
    if query is not None and normalize_non_empty_text(query):
        query_hash = retrieval_query_hash(str(query))
        sql += " AND query_hash = ?"
        filters.append(query_hash)
    if chunk_id is not None:
        sql += " AND chunk_id = ?"
        filters.append(str(chunk_id))
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    filters.append(normalize_limit(limit))

    with storage._connect() as conn:
        rows = conn.execute(sql, tuple(filters)).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        metadata = FeedbackMetadata.from_value(row["metadata"]).to_dict()
        result.append(
            {
                "id": str(row["id"]),
                "query_text": str(row["query_text"]),
                "query_hash": str(row["query_hash"]),
                "chunk_id": str(row["chunk_id"]),
                "score": int(row["score"]),
                "actor": str(row["actor"]),
                "source": str(row["source"]),
                "metadata": metadata,
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
        )
    return result


def get_retrieval_feedback_scores(
    storage: Any,
    *,
    query: str,
    chunk_ids: list[UUID],
) -> dict[UUID, float]:
    unique_chunk_ids = sorted(
        {str(chunk_id) for chunk_id in chunk_ids if normalize_non_empty_text(chunk_id)}
    )
    if not unique_chunk_ids:
        return {}
    query_hash = retrieval_query_hash(query)
    placeholders = ",".join("?" for _ in unique_chunk_ids)

    with storage._connect() as conn:
        specific_rows = conn.execute(
            f"""
            SELECT chunk_id, AVG(score) AS avg_score
            FROM retrieval_feedback
            WHERE tenant_id = ? AND project_id = ?
            AND query_hash = ?
            AND chunk_id IN ({placeholders})
            GROUP BY chunk_id
            """,
            (storage.tenant_id, storage.project_id, query_hash, *unique_chunk_ids),
        ).fetchall()
        global_rows = conn.execute(
            f"""
            SELECT chunk_id, AVG(score) AS avg_score
            FROM retrieval_feedback
            WHERE tenant_id = ? AND project_id = ?
            AND chunk_id IN ({placeholders})
            GROUP BY chunk_id
            """,
            (storage.tenant_id, storage.project_id, *unique_chunk_ids),
        ).fetchall()

    specific_scores = {str(row["chunk_id"]): float(row["avg_score"]) for row in specific_rows}
    global_scores = {str(row["chunk_id"]): float(row["avg_score"]) for row in global_rows}

    merged_scores: dict[UUID, float] = {}
    for chunk_id_text in unique_chunk_ids:
        score = specific_scores.get(chunk_id_text)
        if score is None:
            score = global_scores.get(chunk_id_text, 0.0) * 0.5
        clamped = max(-1.0, min(1.0, float(score)))
        try:
            merged_scores[UUID(chunk_id_text)] = clamped
        except ValueError:
            continue
    return merged_scores
