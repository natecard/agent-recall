from __future__ import annotations

from typing import Any

from agent_recall.storage.normalize import normalize_limit, normalize_non_empty_text


def replace_topic_threads(storage: Any, threads: list[dict[str, Any]]) -> int:
    now = storage._now_iso()
    inserted = 0
    with storage._connect() as conn:
        conn.execute(
            "DELETE FROM topic_thread_links WHERE tenant_id = ? AND project_id = ?",
            (storage.tenant_id, storage.project_id),
        )
        conn.execute(
            "DELETE FROM topic_threads WHERE tenant_id = ? AND project_id = ?",
            (storage.tenant_id, storage.project_id),
        )
        for thread in threads:
            thread_id = str(thread.get("thread_id", "")).strip()
            title = str(thread.get("title", "")).strip()
            summary = str(thread.get("summary", "")).strip()
            if not thread_id or not title:
                continue
            try:
                score = float(thread.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            try:
                entry_count = int(thread.get("entry_count", 0))
            except (TypeError, ValueError):
                entry_count = 0
            try:
                source_session_count = int(thread.get("source_session_count", 0))
            except (TypeError, ValueError):
                source_session_count = 0
            last_seen_at = str(thread.get("last_seen_at", "")).strip() or now
            conn.execute(
                """
                INSERT INTO topic_threads (
                    thread_id, tenant_id, project_id, title, summary, score, entry_count,
                    source_session_count, last_seen_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    storage.tenant_id,
                    storage.project_id,
                    title,
                    summary or title,
                    score,
                    max(0, entry_count),
                    max(0, source_session_count),
                    last_seen_at,
                    now,
                    now,
                ),
            )
            inserted += 1
            links_raw = thread.get("links")
            links = links_raw if isinstance(links_raw, list) else []
            for link in links:
                if not isinstance(link, dict):
                    continue
                entry_id = normalize_non_empty_text(link.get("entry_id"))
                chunk_id = normalize_non_empty_text(link.get("chunk_id"))
                source_session_id = normalize_non_empty_text(link.get("source_session_id"))
                if not entry_id and not chunk_id and not source_session_id:
                    continue
                created_at = normalize_non_empty_text(link.get("created_at")) or now
                conn.execute(
                    """
                    INSERT OR IGNORE INTO topic_thread_links (
                        thread_id, tenant_id, project_id, entry_id, chunk_id,
                        source_session_id, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        storage.tenant_id,
                        storage.project_id,
                        entry_id,
                        chunk_id,
                        source_session_id,
                        created_at,
                    ),
                )
    return inserted


def list_topic_threads(storage: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, summary, score, entry_count, source_session_count,
                   last_seen_at, created_at, updated_at
            FROM topic_threads
            WHERE tenant_id = ? AND project_id = ?
            ORDER BY score DESC, last_seen_at DESC, thread_id ASC
            LIMIT ?
            """,
            (storage.tenant_id, storage.project_id, normalize_limit(limit)),
        ).fetchall()
    return [
        {
            "thread_id": str(row["thread_id"]),
            "title": str(row["title"]),
            "summary": str(row["summary"]),
            "score": float(row["score"]),
            "entry_count": int(row["entry_count"]),
            "source_session_count": int(row["source_session_count"]),
            "last_seen_at": str(row["last_seen_at"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def get_topic_thread(
    storage: Any,
    thread_id: str,
    *,
    limit_links: int = 50,
) -> dict[str, Any] | None:
    normalized = str(thread_id).strip()
    if not normalized:
        return None
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT thread_id, title, summary, score, entry_count, source_session_count,
                   last_seen_at, created_at, updated_at
            FROM topic_threads
            WHERE thread_id = ? AND tenant_id = ? AND project_id = ?
            LIMIT 1
            """,
            (normalized, storage.tenant_id, storage.project_id),
        ).fetchone()
        if not row:
            return None
        link_rows = conn.execute(
            """
            SELECT l.entry_id, l.chunk_id, l.source_session_id, l.created_at,
                   e.content AS entry_content, c.content AS chunk_content
            FROM topic_thread_links l
            LEFT JOIN log_entries e
              ON e.id = l.entry_id AND e.tenant_id = l.tenant_id AND e.project_id = l.project_id
            LEFT JOIN chunks c
              ON c.id = l.chunk_id AND c.tenant_id = l.tenant_id AND c.project_id = l.project_id
            WHERE l.thread_id = ? AND l.tenant_id = ? AND l.project_id = ?
            ORDER BY l.created_at DESC
            LIMIT ?
            """,
            (normalized, storage.tenant_id, storage.project_id, normalize_limit(limit_links)),
        ).fetchall()

    return {
        "thread_id": str(row["thread_id"]),
        "title": str(row["title"]),
        "summary": str(row["summary"]),
        "score": float(row["score"]),
        "entry_count": int(row["entry_count"]),
        "source_session_count": int(row["source_session_count"]),
        "last_seen_at": str(row["last_seen_at"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "links": [
            {
                "entry_id": str(link["entry_id"]) if link["entry_id"] else None,
                "chunk_id": str(link["chunk_id"]) if link["chunk_id"] else None,
                "source_session_id": str(link["source_session_id"])
                if link["source_session_id"]
                else None,
                "created_at": str(link["created_at"]),
                "entry_content": str(link["entry_content"]) if link["entry_content"] else None,
                "chunk_content": str(link["chunk_content"]) if link["chunk_content"] else None,
            }
            for link in link_rows
        ],
    }
