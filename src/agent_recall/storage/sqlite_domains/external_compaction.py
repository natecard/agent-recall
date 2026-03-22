from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from agent_recall.storage.normalize import (
    dump_json_compact,
    normalize_limit,
    normalize_non_empty_text,
    parse_json_string_list,
    utc_now_iso,
)


def list_external_compaction_states(storage: Any, limit: int | None = None) -> list[dict[str, str]]:
    query = (
        "SELECT source_session_id, source_hash, processed_at, updated_at "
        "FROM external_compaction_state "
        "WHERE tenant_id = ? AND project_id = ? "
        "ORDER BY processed_at DESC"
    )
    params: tuple[Any, ...]
    if isinstance(limit, int) and limit > 0:
        query += " LIMIT ?"
        params = (storage.tenant_id, storage.project_id, int(limit))
    else:
        params = (storage.tenant_id, storage.project_id)
    with storage._connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "source_session_id": str(row["source_session_id"]),
            "source_hash": str(row["source_hash"]),
            "processed_at": str(row["processed_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def upsert_external_compaction_state(
    storage: Any,
    source_session_id: str,
    *,
    source_hash: str,
    processed_at: str,
) -> None:
    storage._validate_namespace()
    source_id = normalize_non_empty_text(source_session_id)
    source_hash_value = normalize_non_empty_text(source_hash)
    processed_at_value = normalize_non_empty_text(processed_at)
    if not source_id:
        raise ValueError("source_session_id is required")
    if not source_hash_value:
        raise ValueError("source_hash is required")
    if not processed_at_value:
        raise ValueError("processed_at is required")

    now = utc_now_iso()
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO external_compaction_state
                (
                    source_session_id,
                    tenant_id,
                    project_id,
                    source_hash,
                    processed_at,
                    updated_at
                )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_session_id) DO UPDATE SET
                source_hash=excluded.source_hash,
                processed_at=excluded.processed_at,
                updated_at=excluded.updated_at
            """,
            (
                source_id,
                storage.tenant_id,
                storage.project_id,
                source_hash_value,
                processed_at_value,
                now,
            ),
        )


def delete_external_compaction_state(storage: Any, source_session_id: str) -> int:
    source_id = normalize_non_empty_text(source_session_id)
    if not source_id:
        return 0
    with storage._connect() as conn:
        cursor = conn.execute(
            "DELETE FROM external_compaction_state "
            "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?",
            (source_id, storage.tenant_id, storage.project_id),
        )
        return int(cursor.rowcount or 0)


def queue_note_key(
    *,
    tier: str,
    line: str,
    source_session_ids: list[str],
) -> str:
    normalized_line = " ".join(line.strip().lower().split())
    normalized_sources = sorted({str(item).strip() for item in source_session_ids if item})
    raw = f"{tier.strip().upper()}|{normalized_line}|{json.dumps(normalized_sources)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def enqueue_external_compaction_queue(
    storage: Any,
    notes: list[dict[str, Any]],
    *,
    actor: str,
) -> list[dict[str, Any]]:
    storage._validate_namespace()
    now = utc_now_iso()
    created_ids: list[int] = []
    with storage._connect() as conn:
        for note in notes:
            tier = str(note.get("tier", "")).strip().upper()
            line = str(note.get("line", "")).strip()
            source_session_ids_raw = note.get("source_session_ids", [])
            source_session_ids = (
                sorted({str(item).strip() for item in source_session_ids_raw if str(item).strip()})
                if isinstance(source_session_ids_raw, list)
                else []
            )
            if not tier or not line:
                continue
            note_key = queue_note_key(
                tier=tier,
                line=line,
                source_session_ids=source_session_ids,
            )
            conn.execute(
                """
                INSERT INTO external_compaction_queue (
                    tenant_id,
                    project_id,
                    note_key,
                    state,
                    tier,
                    line,
                    source_session_ids,
                    actor,
                    action_timestamp,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, project_id, note_key) DO NOTHING
                """,
                (
                    storage.tenant_id,
                    storage.project_id,
                    note_key,
                    tier,
                    line,
                    dump_json_compact(source_session_ids),
                    (normalize_non_empty_text(actor) or "system"),
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM external_compaction_queue
                WHERE tenant_id = ? AND project_id = ? AND note_key = ?
                LIMIT 1
                """,
                (storage.tenant_id, storage.project_id, note_key),
            ).fetchone()
            if row:
                created_ids.append(int(row["id"]))

    if not created_ids:
        return []
    ids = sorted(set(created_ids))
    placeholders = ",".join("?" for _ in ids)
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, state, tier, line, source_session_ids, actor, action_timestamp,
                   created_at, updated_at, approved_at, rejected_at, applied_at
            FROM external_compaction_queue
            WHERE tenant_id = ? AND project_id = ? AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            (storage.tenant_id, storage.project_id, *ids),
        ).fetchall()
    return [serialize_external_compaction_queue_row(row) for row in rows]


def list_external_compaction_queue(
    storage: Any,
    *,
    states: list[str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    filters = [storage.tenant_id, storage.project_id]
    query = (
        "SELECT id, state, tier, line, source_session_ids, actor, action_timestamp, "
        "created_at, updated_at, approved_at, rejected_at, applied_at "
        "FROM external_compaction_queue "
        "WHERE tenant_id = ? AND project_id = ?"
    )
    normalized_states = [str(item).strip().lower() for item in (states or []) if str(item).strip()]
    if normalized_states:
        placeholders = ",".join("?" for _ in normalized_states)
        query += f" AND state IN ({placeholders})"
        filters.extend(normalized_states)
    query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    filters.append(normalize_limit(limit))
    with storage._connect() as conn:
        rows = conn.execute(query, tuple(filters)).fetchall()
    return [serialize_external_compaction_queue_row(row) for row in rows]


def update_external_compaction_queue_state(
    storage: Any,
    *,
    ids: list[int],
    target_state: str,
    actor: str,
) -> dict[str, int]:
    normalized_ids = sorted({int(item) for item in ids if int(item) > 0})
    if not normalized_ids:
        return {"updated": 0, "skipped": 0}

    state = str(target_state).strip().lower()
    if state not in {"approved", "rejected", "applied"}:
        raise ValueError("target_state must be approved, rejected, or applied")

    allowed_from: dict[str, set[str]] = {
        "approved": {"pending"},
        "rejected": {"pending", "approved"},
        "applied": {"approved"},
    }
    from_states = allowed_from[state]
    from_placeholders = ",".join("?" for _ in from_states)
    id_placeholders = ",".join("?" for _ in normalized_ids)
    now = utc_now_iso()
    actor_value = normalize_non_empty_text(actor) or "system"
    updates: dict[str, str | None] = {
        "approved_at": now if state == "approved" else None,
        "rejected_at": now if state == "rejected" else None,
        "applied_at": now if state == "applied" else None,
    }

    set_parts = [
        "state = ?",
        "actor = ?",
        "action_timestamp = ?",
        "updated_at = ?",
        "approved_at = ?",
        "rejected_at = ?",
        "applied_at = ?",
    ]
    params: list[Any] = [
        state,
        actor_value,
        now,
        now,
        updates["approved_at"],
        updates["rejected_at"],
        updates["applied_at"],
        storage.tenant_id,
        storage.project_id,
        *from_states,
        *normalized_ids,
    ]
    with storage._connect() as conn:
        cursor = conn.execute(
            f"""
            UPDATE external_compaction_queue
            SET {", ".join(set_parts)}
            WHERE tenant_id = ? AND project_id = ?
            AND state IN ({from_placeholders})
            AND id IN ({id_placeholders})
            """,
            tuple(params),
        )
        updated = int(cursor.rowcount or 0)

    return {"updated": updated, "skipped": max(0, len(normalized_ids) - updated)}


def record_external_compaction_evidence(storage: Any, notes: list[dict[str, Any]]) -> int:
    storage._validate_namespace()
    now = utc_now_iso()
    written = 0
    with storage._connect() as conn:
        for note in notes:
            tier = str(note.get("tier", "")).strip().upper()
            line = str(note.get("line", "")).strip()
            source_ids_raw = note.get("source_session_ids", [])
            if not isinstance(source_ids_raw, list):
                source_ids_raw = []
            source_ids = sorted({str(item).strip() for item in source_ids_raw if str(item).strip()})
            if not tier or not line or not source_ids:
                continue
            for source_id in source_ids:
                cursor = conn.execute(
                    """
                    INSERT INTO external_compaction_evidence (
                        tenant_id,
                        project_id,
                        tier,
                        line,
                        source_session_id,
                        merged_at,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, project_id, tier, line, source_session_id)
                    DO UPDATE SET
                        merged_at=excluded.merged_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        storage.tenant_id,
                        storage.project_id,
                        tier,
                        line,
                        source_id,
                        now,
                        now,
                        now,
                    ),
                )
                written += int(cursor.rowcount or 0)
    return written


def list_external_compaction_evidence(storage: Any, limit: int = 200) -> list[dict[str, Any]]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, tier, line, source_session_id, merged_at, created_at, updated_at
            FROM external_compaction_evidence
            WHERE tenant_id = ? AND project_id = ?
            ORDER BY merged_at DESC, id DESC
            LIMIT ?
            """,
            (storage.tenant_id, storage.project_id, normalize_limit(limit)),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "tier": str(row["tier"]),
            "line": str(row["line"]),
            "source_session_id": str(row["source_session_id"]),
            "merged_at": str(row["merged_at"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def serialize_external_compaction_queue_row(row: sqlite3.Row) -> dict[str, Any]:
    source_ids = parse_json_string_list(row["source_session_ids"])
    return {
        "id": int(row["id"]),
        "state": str(row["state"]),
        "tier": str(row["tier"]),
        "line": str(row["line"]),
        "source_session_ids": source_ids,
        "actor": str(row["actor"]),
        "timestamp": str(row["action_timestamp"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "approved_at": str(row["approved_at"]) if row["approved_at"] else None,
        "rejected_at": str(row["rejected_at"]) if row["rejected_at"] else None,
        "applied_at": str(row["applied_at"]) if row["applied_at"] else None,
    }
