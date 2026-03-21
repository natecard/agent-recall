from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from agent_recall.storage.base import Storage, validate_shared_namespace
from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    ChunkSource,
    CurationStatus,
    LogEntry,
    LogSource,
    ScoredChunk,
    SemanticLabel,
    Session,
    SessionCheckpoint,
    SessionStatus,
)

CHUNKS_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    tags,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, tags)
    VALUES (NEW.rowid, NEW.content, NEW.tags);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, tags)
    VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, tags)
    VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
    INSERT INTO chunks_fts(rowid, content, tags)
    VALUES (NEW.rowid, NEW.content, NEW.tags);
END;
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    task TEXT NOT NULL,
    summary TEXT,
    entry_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS log_entries (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT REFERENCES sessions(id),
    source TEXT NOT NULL,
    source_session_id TEXT,
    timestamp TEXT NOT NULL,
    content TEXT NOT NULL,
    label TEXT NOT NULL,
    tags TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    curation_status TEXT NOT NULL DEFAULT 'approved',
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    content TEXT NOT NULL,
    label TEXT NOT NULL,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL,
    token_count INTEGER,
    embedding BLOB,
    embedding_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embedding_indices (
    chunk_id TEXT NOT NULL,
    session_id TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, session_id, tenant_id, project_id)
);

{CHUNKS_FTS_SCHEMA}

CREATE TABLE IF NOT EXISTS processed_sessions (
    source_session_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_session ON log_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_entries_label ON log_entries(label);
CREATE INDEX IF NOT EXISTS idx_chunks_label ON chunks(label);

CREATE TABLE IF NOT EXISTS session_checkpoints (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    source_session_id TEXT NOT NULL UNIQUE,
    last_message_timestamp TEXT,
    last_message_index INTEGER,
    content_hash TEXT,
    checkpoint_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON session_checkpoints(source_session_id);

CREATE TABLE IF NOT EXISTS background_sync_status (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    is_running INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    sessions_processed INTEGER NOT NULL DEFAULT 0,
    learnings_extracted INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    pid INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_compaction_state (
    source_session_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    source_hash TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_external_compaction_state_processed_at
ON external_compaction_state(tenant_id, project_id, processed_at DESC);

CREATE TABLE IF NOT EXISTS external_compaction_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    note_key TEXT NOT NULL,
    state TEXT NOT NULL,
    tier TEXT NOT NULL,
    line TEXT NOT NULL,
    source_session_ids TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'system',
    action_timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_at TEXT,
    rejected_at TEXT,
    applied_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_external_compaction_queue_note_key
ON external_compaction_queue(tenant_id, project_id, note_key);
CREATE INDEX IF NOT EXISTS idx_external_compaction_queue_state
ON external_compaction_queue(tenant_id, project_id, state, updated_at DESC);

CREATE TABLE IF NOT EXISTS external_compaction_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    tier TEXT NOT NULL,
    line TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    merged_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(tenant_id, project_id, tier, line, source_session_id)
);

CREATE INDEX IF NOT EXISTS idx_external_compaction_evidence_source
ON external_compaction_evidence(tenant_id, project_id, source_session_id, merged_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_feedback (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    score INTEGER NOT NULL,
    actor TEXT NOT NULL DEFAULT 'user',
    source TEXT NOT NULL DEFAULT 'cli',
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_scope_query
ON retrieval_feedback(tenant_id, project_id, query_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_feedback_scope_chunk
ON retrieval_feedback(tenant_id, project_id, chunk_id, created_at DESC);

CREATE TABLE IF NOT EXISTS topic_threads (
    thread_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    entry_count INTEGER NOT NULL DEFAULT 0,
    source_session_count INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topic_threads_scope_last_seen
ON topic_threads(tenant_id, project_id, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS topic_thread_links (
    thread_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    entry_id TEXT,
    chunk_id TEXT,
    source_session_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(thread_id, tenant_id, project_id, entry_id, chunk_id, source_session_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_thread_links_scope
ON topic_thread_links(tenant_id, project_id, thread_id, created_at DESC);

CREATE TABLE IF NOT EXISTS rule_confidence (
    rule_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    tier TEXT NOT NULL,
    line TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.6,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced_at TEXT,
    last_decayed_at TEXT,
    is_stale INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rule_confidence_scope
ON rule_confidence(tenant_id, project_id, confidence ASC, is_stale DESC);

CREATE TABLE IF NOT EXISTS rule_confidence_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    project_id TEXT NOT NULL DEFAULT 'default',
    tier TEXT NOT NULL,
    line TEXT NOT NULL,
    confidence REAL NOT NULL,
    reinforcement_count INTEGER NOT NULL,
    last_reinforced_at TEXT,
    last_decayed_at TEXT,
    is_stale INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rule_confidence_archive_scope
ON rule_confidence_archive(tenant_id, project_id, archived_at DESC);
"""

SCOPE_INDEXES_BY_TABLE = {
    "sessions": "idx_sessions_scope",
    "log_entries": "idx_entries_scope",
    "chunks": "idx_chunks_scope",
    "processed_sessions": "idx_processed_sessions_scope",
    "session_checkpoints": "idx_session_checkpoints_scope",
    "background_sync_status": "idx_background_sync_status_scope",
    "embedding_indices": "idx_embedding_indices_scope",
    "external_compaction_state": "idx_external_compaction_state_scope",
    "external_compaction_queue": "idx_external_compaction_queue_scope",
    "external_compaction_evidence": "idx_external_compaction_evidence_scope",
    "retrieval_feedback": "idx_retrieval_feedback_scope",
    "topic_threads": "idx_topic_threads_scope",
    "topic_thread_links": "idx_topic_thread_links_scope_compact",
    "rule_confidence": "idx_rule_confidence_scope_compact",
    "rule_confidence_archive": "idx_rule_confidence_archive_scope_compact",
}


class SQLiteStorage(Storage):
    def __init__(
        self,
        db_path: Path,
        tenant_id: str = "default",
        project_id: str = "default",
        strict_namespace_validation: bool = False,
    ) -> None:
        self.db_path = db_path
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.strict_namespace_validation = strict_namespace_validation
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        self._migrate_db()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
        return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]

    def _ensure_scope_indexes(self, conn: sqlite3.Connection) -> None:
        for table, index_name in SCOPE_INDEXES_BY_TABLE.items():
            columns = self._table_columns(conn, table)
            if "tenant_id" in columns and "project_id" in columns:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}(tenant_id, project_id)"
                )

    def _migrate_db(self) -> None:
        """Ensure schema has required columns for tenant/project isolation."""
        tables = [
            "sessions",
            "log_entries",
            "chunks",
            "processed_sessions",
            "session_checkpoints",
            "background_sync_status",
            "embedding_indices",
            "external_compaction_state",
            "external_compaction_queue",
            "external_compaction_evidence",
            "retrieval_feedback",
            "topic_threads",
            "topic_thread_links",
            "rule_confidence",
            "rule_confidence_archive",
        ]
        with self._connect() as conn:
            for table in tables:
                # Check if table exists first (it should from SCHEMA, but safety check)
                exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not exists:
                    continue

                columns = self._table_columns(conn, table)
                if "tenant_id" not in columns:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                    )
                    columns.append("tenant_id")
                if "project_id" not in columns:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN project_id TEXT NOT NULL DEFAULT 'default'"
                    )
                    columns.append("project_id")
                if table == "chunks" and "embedding" not in columns:
                    conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
                    columns.append("embedding")
                if table == "chunks" and "embedding_version" not in columns:
                    conn.execute(
                        "ALTER TABLE chunks ADD COLUMN embedding_version INTEGER NOT NULL DEFAULT 0"
                    )
                    columns.append("embedding_version")
                if table == "log_entries" and "curation_status" not in columns:
                    conn.execute(
                        "ALTER TABLE log_entries "
                        "ADD COLUMN curation_status TEXT NOT NULL DEFAULT 'approved'"
                    )
                    columns.append("curation_status")
            self._ensure_scope_indexes(conn)

    def _validate_namespace(self) -> None:
        """Validate namespace if strict mode is enabled."""
        if self.strict_namespace_validation:
            validate_shared_namespace(self.tenant_id, self.project_id)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_session(self, session: Session) -> None:
        self._validate_namespace()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions (
                       id, tenant_id, project_id, status, started_at, ended_at,
                       task, summary, entry_count
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(session.id),
                    self.tenant_id,
                    self.project_id,
                    session.status.value,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.task,
                    session.summary,
                    session.entry_count,
                ),
            )

    def get_session(self, session_id: UUID) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND tenant_id = ? AND project_id = ?",
                (str(session_id), self.tenant_id, self.project_id),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def get_active_session(self) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT * FROM sessions "
                    "WHERE status = ? AND tenant_id = ? AND project_id = ? "
                    "ORDER BY started_at DESC LIMIT 1"
                ),
                (SessionStatus.ACTIVE.value, self.tenant_id, self.project_id),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    (
                        "SELECT * FROM sessions "
                        "WHERE tenant_id = ? AND project_id = ? "
                        "ORDER BY started_at DESC LIMIT ?"
                    ),
                    (self.tenant_id, self.project_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    (
                        "SELECT * FROM sessions "
                        "WHERE status = ? AND tenant_id = ? AND project_id = ? "
                        "ORDER BY started_at DESC LIMIT ?"
                    ),
                    (status.value, self.tenant_id, self.project_id, limit),
                ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def update_session(self, session: Session) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions SET status=?, ended_at=?, summary=?, entry_count=?
                   WHERE id=? AND tenant_id=? AND project_id=?""",
                (
                    session.status.value,
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.summary,
                    session.entry_count,
                    str(session.id),
                    self.tenant_id,
                    self.project_id,
                ),
            )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            status=SessionStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            task=row["task"],
            summary=row["summary"],
            entry_count=row["entry_count"],
        )

    def append_entry(self, entry: LogEntry) -> None:
        self._validate_namespace()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO log_entries
                   (
                        id, tenant_id, project_id, session_id, source, source_session_id, timestamp,
                        content, label, tags, confidence, curation_status, metadata
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(entry.id),
                    self.tenant_id,
                    self.project_id,
                    str(entry.session_id) if entry.session_id else None,
                    entry.source.value,
                    entry.source_session_id,
                    entry.timestamp.isoformat(),
                    entry.content,
                    entry.label.value,
                    json.dumps(entry.tags),
                    entry.confidence,
                    entry.curation_status.value,
                    json.dumps(entry.metadata),
                ),
            )
            if entry.session_id:
                conn.execute(
                    (
                        "UPDATE sessions SET entry_count = entry_count + 1 "
                        "WHERE id = ? AND tenant_id = ? AND project_id = ?"
                    ),
                    (str(entry.session_id), self.tenant_id, self.project_id),
                )

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT * FROM log_entries "
                    "WHERE session_id = ? AND tenant_id = ? AND project_id = ? "
                    "ORDER BY timestamp"
                ),
                (str(session_id), self.tenant_id, self.project_id),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entries_by_source_session(
        self,
        source_session_id: str,
        limit: int = 200,
    ) -> list[LogEntry]:
        normalized = source_session_id.strip()
        if not normalized:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT * FROM log_entries "
                    "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ? "
                    "ORDER BY timestamp ASC LIMIT ?"
                ),
                (normalized, self.tenant_id, self.project_id, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entries_by_label(
        self,
        labels: list[SemanticLabel],
        limit: int = 100,
        curation_status: CurationStatus = CurationStatus.APPROVED,
    ) -> list[LogEntry]:
        if not labels:
            return []

        placeholders = ",".join("?" * len(labels))
        with self._connect() as conn:
            rows = conn.execute(
                (
                    f"SELECT * FROM log_entries "
                    f"WHERE label IN ({placeholders}) "
                    "AND curation_status = ? "
                    "AND tenant_id = ? AND project_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?"
                ),
                [label.value for label in labels]
                + [curation_status.value, self.tenant_id, self.project_id, limit],
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def list_entries_by_curation_status(
        self,
        status: CurationStatus | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        query = "SELECT * FROM log_entries WHERE tenant_id = ? AND project_id = ?"
        params: list[Any] = [self.tenant_id, self.project_id]
        if status is None:
            status = CurationStatus.APPROVED
        query += " AND curation_status = ?"
        params.append(status.value)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def update_entry_curation_status(
        self,
        entry_id: UUID,
        status: CurationStatus,
    ) -> LogEntry | None:
        self._validate_namespace()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM log_entries WHERE id = ? AND tenant_id = ? AND project_id = ?",
                (str(entry_id), self.tenant_id, self.project_id),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                (
                    "UPDATE log_entries SET curation_status = ? "
                    "WHERE id = ? AND tenant_id = ? AND project_id = ?"
                ),
                (status.value, str(entry_id), self.tenant_id, self.project_id),
            )
            refreshed = conn.execute(
                "SELECT * FROM log_entries WHERE id = ? AND tenant_id = ? AND project_id = ?",
                (str(entry_id), self.tenant_id, self.project_id),
            ).fetchone()
        return self._row_to_entry(refreshed) if refreshed else None

    def count_log_entries(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM log_entries WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
        return int(row["n"]) if row else 0

    def _row_to_entry(self, row: sqlite3.Row) -> LogEntry:
        raw_status = row["curation_status"] if "curation_status" in row.keys() else None
        status_value = str(raw_status) if raw_status else CurationStatus.APPROVED.value
        try:
            curation_status = CurationStatus(status_value)
        except ValueError:
            curation_status = CurationStatus.APPROVED
        return LogEntry(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            session_id=UUID(row["session_id"]) if row["session_id"] else None,
            source=LogSource(row["source"]),
            source_session_id=row["source_session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            content=row["content"],
            label=SemanticLabel(row["label"]),
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            curation_status=curation_status,
            metadata=json.loads(row["metadata"]),
        )

    def store_chunk(self, chunk: Chunk) -> None:
        self._validate_namespace()
        for attempt in range(2):
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO chunks
                           (
                               id, tenant_id, project_id, source, source_ids, content, label,
                               tags, created_at, token_count, embedding, embedding_version
                           )
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(chunk.id),
                            self.tenant_id,
                            self.project_id,
                            chunk.source.value,
                            json.dumps([str(item) for item in chunk.source_ids]),
                            chunk.content,
                            chunk.label.value,
                            json.dumps(chunk.tags),
                            chunk.created_at.isoformat(),
                            chunk.token_count,
                            self._serialize_embedding(chunk.embedding),
                            chunk.embedding_version,
                        ),
                    )
                return
            except sqlite3.DatabaseError as exc:
                if attempt == 0 and self._is_chunks_fts_corruption(exc):
                    self.rebuild_chunks_fts()
                    continue
                raise

    def index_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        self.save_embedding(chunk_id=chunk_id, embedding=embedding, version=1)

    def save_embedding(self, chunk_id: UUID, embedding: list[float], version: int = 1) -> None:
        self._validate_namespace()
        with self._connect() as conn:
            conn.execute(
                (
                    "UPDATE chunks "
                    "SET embedding = ?, embedding_version = ? "
                    "WHERE id = ? AND tenant_id = ? AND project_id = ?"
                ),
                (
                    self._serialize_embedding(embedding),
                    int(version),
                    str(chunk_id),
                    self.tenant_id,
                    self.project_id,
                ),
            )

    def load_embedding(self, chunk_id: UUID) -> tuple[list[float], int] | None:
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT embedding, embedding_version FROM chunks "
                    "WHERE id = ? AND tenant_id = ? AND project_id = ?"
                ),
                (str(chunk_id), self.tenant_id, self.project_id),
            ).fetchone()
        if not row:
            return None

        embedding = self._deserialize_embedding(row["embedding"])
        if embedding is None:
            return None

        version = int(row["embedding_version"]) if row["embedding_version"] is not None else 0
        return embedding, version

    def get_chunks_without_embeddings(self, limit: int = 100) -> list[Chunk]:
        max_limit = max(1, int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT * FROM chunks "
                    "WHERE embedding IS NULL "
                    "AND tenant_id = ? AND project_id = ? "
                    "ORDER BY created_at DESC, id ASC LIMIT ?"
                ),
                (self.tenant_id, self.project_id, max_limit),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def mark_embedding_indexed(self, chunk_id: UUID, session_id: UUID | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        session_value = str(session_id) if session_id is not None else None
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO embedding_indices "
                    "(chunk_id, session_id, tenant_id, project_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, COALESCE(("
                    "SELECT created_at FROM embedding_indices "
                    "WHERE chunk_id = ? AND session_id IS ? "
                    "AND tenant_id = ? AND project_id = ?"
                    "), ?), ?)"
                ),
                (
                    str(chunk_id),
                    session_value,
                    self.tenant_id,
                    self.project_id,
                    str(chunk_id),
                    session_value,
                    self.tenant_id,
                    self.project_id,
                    now,
                    now,
                ),
            )

    def get_embedding_index_status(self) -> dict[str, int]:
        with self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
            embedded_row = conn.execute(
                (
                    "SELECT COUNT(*) AS n FROM chunks "
                    "WHERE embedding IS NOT NULL AND tenant_id = ? AND project_id = ?"
                ),
                (self.tenant_id, self.project_id),
            ).fetchone()

        total_chunks = int(total_row["n"]) if total_row else 0
        embedded_chunks = int(embedded_row["n"]) if embedded_row else 0
        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "pending": max(0, total_chunks - embedded_chunks),
        }

    @staticmethod
    def _is_chunks_fts_corruption(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return any(
            token in lowered
            for token in (
                "vtable constructor failed: chunks_fts",
                "invalid fts5 file format",
                "malformed",
            )
        )

    def rebuild_chunks_fts(self) -> None:
        """Rebuild chunks FTS table and triggers from canonical chunk rows."""
        rebuild_script = (
            """
        DROP TRIGGER IF EXISTS chunks_ai;
        DROP TRIGGER IF EXISTS chunks_ad;
        DROP TRIGGER IF EXISTS chunks_au;
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS chunks_fts_data;
        DROP TABLE IF EXISTS chunks_fts_idx;
        DROP TABLE IF EXISTS chunks_fts_docsize;
        DROP TABLE IF EXISTS chunks_fts_config;
        """
            + CHUNKS_FTS_SCHEMA
            + """
        INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');
        """
        )
        with self._connect() as conn:
            conn.executescript(rebuild_script)

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT 1 FROM chunks "
                    "WHERE content = ? AND label = ? "
                    "AND tenant_id = ? AND project_id = ? "
                    "LIMIT 1"
                ),
                (content, label.value, self.tenant_id, self.project_id),
            ).fetchone()
        return row is not None

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
        return int(row["n"]) if row else 0

    @staticmethod
    def _serialize_embedding(embedding: list[float] | None) -> bytes | None:
        if embedding is None:
            return None
        return json.dumps(embedding).encode("utf-8")

    @staticmethod
    def _deserialize_embedding(raw: Any) -> list[float] | None:
        if raw is None:
            return None

        if isinstance(raw, bytes):
            serialized = raw.decode("utf-8", errors="ignore")
        else:
            serialized = str(raw)

        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None

        embedding: list[float] = []
        for value in payload:
            if isinstance(value, (int, float)):
                embedding.append(float(value))
            else:
                return None
        return embedding

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """SELECT c.* FROM chunks c
                       JOIN chunks_fts fts ON c.rowid = fts.rowid
                       WHERE chunks_fts MATCH ?
                       AND c.tenant_id = ? AND c.project_id = ?
                       ORDER BY bm25(chunks_fts) LIMIT ?""",
                    (query, self.tenant_id, self.project_id, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM chunks
                   WHERE embedding IS NOT NULL
                   AND tenant_id = ? AND project_id = ?
                   ORDER BY created_at DESC, id ASC""",
                (self.tenant_id, self.project_id),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM chunks
                   WHERE tenant_id = ? AND project_id = ?
                   ORDER BY created_at DESC, id ASC""",
                (self.tenant_id, self.project_id),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def _has_vec_extension(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT vec_version()").fetchone()
            return True
        except Exception:
            return False

    def _calculate_cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def search_chunks_by_embedding(
        self, embedding: list[float], limit: int = 10
    ) -> list[ScoredChunk]:
        with self._connect() as conn:
            has_vec = self._has_vec_extension(conn)

            if has_vec:
                return self._search_with_vec_extension(conn, embedding, limit)

            return self._search_with_fallback(conn, embedding, limit)

    def _search_with_vec_extension(
        self, conn: sqlite3.Connection, embedding: list[float], limit: int
    ) -> list[ScoredChunk]:
        embedding_json = json.dumps(embedding)
        try:
            rows = conn.execute(
                """SELECT c.*, vec_distance_cosine(c.embedding, ?) as score
                   FROM chunks c
                   WHERE c.embedding IS NOT NULL
                   AND c.tenant_id = ? AND c.project_id = ?
                   ORDER BY score DESC
                   LIMIT ?""",
                (embedding_json, self.tenant_id, self.project_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return self._search_with_fallback(conn, embedding, limit)

        return [self._row_to_scored_chunk(row) for row in rows]

    def _search_with_fallback(
        self, conn: sqlite3.Connection, embedding: list[float], limit: int
    ) -> list[ScoredChunk]:
        rows = conn.execute(
            """SELECT * FROM chunks
               WHERE embedding IS NOT NULL
               AND tenant_id = ? AND project_id = ?
               ORDER BY created_at DESC, id ASC""",
            (self.tenant_id, self.project_id),
        ).fetchall()

        scored_chunks: list[tuple[ScoredChunk, float]] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            if chunk.embedding is not None:
                score = self._calculate_cosine_similarity(embedding, chunk.embedding)
                scored_chunk = ScoredChunk(**chunk.model_dump(), score=score)
                scored_chunks.append((scored_chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:limit]]

    def _row_to_scored_chunk(self, row: sqlite3.Row) -> ScoredChunk:
        chunk = self._row_to_chunk(row)
        score = row["score"] if "score" in row.keys() else 0.0
        return ScoredChunk(**chunk.model_dump(), score=score)

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            source=ChunkSource(row["source"]),
            source_ids=[UUID(item) for item in json.loads(row["source_ids"])],
            content=row["content"],
            label=SemanticLabel(row["label"]),
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            token_count=row["token_count"],
            embedding=self._deserialize_embedding(row["embedding"]),
            embedding_version=(
                int(row["embedding_version"]) if "embedding_version" in row.keys() else 0
            ),
        )

    def is_session_processed(self, source_session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT 1 FROM processed_sessions "
                    "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?"
                ),
                (source_session_id, self.tenant_id, self.project_id),
            ).fetchone()
        return row is not None

    def mark_session_processed(self, source_session_id: str) -> None:
        self._validate_namespace()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO processed_sessions "
                    "(source_session_id, tenant_id, project_id, processed_at) VALUES (?, ?, ?, ?)"
                ),
                (source_session_id, self.tenant_id, self.project_id, datetime.now(UTC).isoformat()),
            )

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Clear processed-session markers and return number removed."""
        with self._connect() as conn:
            if source_session_id:
                cursor = conn.execute(
                    (
                        "DELETE FROM processed_sessions "
                        "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?"
                    ),
                    (source_session_id, self.tenant_id, self.project_id),
                )
                return int(cursor.rowcount or 0)

            if source:
                normalized = source.strip().lower().replace("_", "-")
                pattern = f"{normalized}-%"
                cursor = conn.execute(
                    (
                        "DELETE FROM processed_sessions "
                        "WHERE source_session_id LIKE ? AND tenant_id = ? AND project_id = ?"
                    ),
                    (pattern, self.tenant_id, self.project_id),
                )
                return int(cursor.rowcount or 0)

            cursor = conn.execute(
                "DELETE FROM processed_sessions WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            )
            return int(cursor.rowcount or 0)

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        """Retrieve checkpoint for a source session, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, tenant_id, project_id, source_session_id, last_message_timestamp,
                          last_message_index, content_hash, checkpoint_at, updated_at
                   FROM session_checkpoints
                   WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?""",
                (source_session_id, self.tenant_id, self.project_id),
            ).fetchone()

        if not row:
            return None

        return SessionCheckpoint(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            source_session_id=row["source_session_id"],
            last_message_timestamp=(
                datetime.fromisoformat(row["last_message_timestamp"])
                if row["last_message_timestamp"]
                else None
            ),
            last_message_index=row["last_message_index"],
            content_hash=row["content_hash"],
            checkpoint_at=datetime.fromisoformat(row["checkpoint_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """Persist or update a session checkpoint."""
        self._validate_namespace()
        now = datetime.now(UTC)
        checkpoint.updated_at = now

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO session_checkpoints
                    (id, tenant_id, project_id, source_session_id,
                     last_message_timestamp, last_message_index,
                     content_hash, checkpoint_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_session_id) DO UPDATE SET
                    last_message_timestamp=excluded.last_message_timestamp,
                    last_message_index=excluded.last_message_index,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at""",
                (
                    str(checkpoint.id),
                    self.tenant_id,
                    self.project_id,
                    checkpoint.source_session_id,
                    checkpoint.last_message_timestamp.isoformat()
                    if checkpoint.last_message_timestamp
                    else None,
                    checkpoint.last_message_index,
                    checkpoint.content_hash,
                    checkpoint.checkpoint_at.isoformat(),
                    checkpoint.updated_at.isoformat(),
                ),
            )

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Clear session checkpoints and return number removed.

        Args:
            source: Optional source name to clear all checkpoints for that source
            source_session_id: Optional specific session ID to clear
        """
        with self._connect() as conn:
            if source_session_id:
                cursor = conn.execute(
                    (
                        "DELETE FROM session_checkpoints "
                        "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?"
                    ),
                    (source_session_id, self.tenant_id, self.project_id),
                )
                return int(cursor.rowcount or 0)

            if source:
                normalized = source.strip().lower().replace("_", "-")
                pattern = f"{normalized}-%"
                cursor = conn.execute(
                    (
                        "DELETE FROM session_checkpoints "
                        "WHERE source_session_id LIKE ? AND tenant_id = ? AND project_id = ?"
                    ),
                    (pattern, self.tenant_id, self.project_id),
                )
                return int(cursor.rowcount or 0)

            cursor = conn.execute(
                "DELETE FROM session_checkpoints WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            )
            return int(cursor.rowcount or 0)

    def get_stats(self) -> dict[str, int]:
        """Get knowledge base statistics."""
        with self._connect() as conn:
            stats: dict[str, int] = {}

            row = conn.execute(
                "SELECT COUNT(*) AS count FROM processed_sessions "
                "WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
            stats["processed_sessions"] = int(row["count"]) if row else 0

            row = conn.execute(
                "SELECT COUNT(*) AS count FROM log_entries WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
            stats["log_entries"] = int(row["count"]) if row else 0

            row = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
            stats["chunks"] = int(row["count"]) if row else 0

            row = conn.execute(
                "SELECT COUNT(*) AS count FROM session_checkpoints "
                "WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            ).fetchone()
            stats["checkpoints"] = int(row["count"]) if row else 0

        return stats

    def get_last_processed_at(self) -> datetime | None:
        """Return the most recent processed-session timestamp."""
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT MAX(processed_at) AS last_processed FROM processed_sessions "
                    "WHERE tenant_id = ? AND project_id = ?"
                ),
                (self.tenant_id, self.project_id),
            ).fetchone()
        if not row:
            return None
        raw_value = row["last_processed"]
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(str(raw_value))
        except ValueError:
            return None

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Summarize recent source sessions inferred from log entries."""
        with self._connect() as conn:
            sessions = conn.execute(
                """
                SELECT source_session_id, MAX(timestamp) AS last_timestamp, COUNT(*) AS entry_count
                FROM log_entries
                WHERE source_session_id IS NOT NULL AND TRIM(source_session_id) != ''
                AND tenant_id = ? AND project_id = ?
                GROUP BY source_session_id
                ORDER BY last_timestamp DESC
                LIMIT ?
                """,
                (self.tenant_id, self.project_id, limit),
            ).fetchall()

            results: list[dict[str, Any]] = []
            for session_row in sessions:
                source_session_id = str(session_row["source_session_id"])
                highlights = conn.execute(
                    """
                    SELECT label, content
                    FROM log_entries
                    WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 5
                    """,
                    (source_session_id, self.tenant_id, self.project_id),
                ).fetchall()
                results.append(
                    {
                        "source_session_id": source_session_id,
                        "last_timestamp": datetime.fromisoformat(
                            str(session_row["last_timestamp"])
                        ),
                        "entry_count": int(session_row["entry_count"]),
                        "highlights": [
                            {
                                "label": str(item["label"]),
                                "content": str(item["content"]),
                            }
                            for item in highlights
                        ],
                    }
                )

        return results

    def list_external_compaction_states(self, limit: int | None = None) -> list[dict[str, str]]:
        """List external compaction state rows for this namespace."""
        query = (
            "SELECT source_session_id, source_hash, processed_at, updated_at "
            "FROM external_compaction_state "
            "WHERE tenant_id = ? AND project_id = ? "
            "ORDER BY processed_at DESC"
        )
        params: tuple[Any, ...]
        if isinstance(limit, int) and limit > 0:
            query += " LIMIT ?"
            params = (self.tenant_id, self.project_id, int(limit))
        else:
            params = (self.tenant_id, self.project_id)
        with self._connect() as conn:
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
        self,
        source_session_id: str,
        *,
        source_hash: str,
        processed_at: str,
    ) -> None:
        """Insert/update external compaction state for a source session."""
        self._validate_namespace()
        source_id = source_session_id.strip()
        source_hash_value = source_hash.strip()
        processed_at_value = processed_at.strip()
        if not source_id:
            raise ValueError("source_session_id is required")
        if not source_hash_value:
            raise ValueError("source_hash is required")
        if not processed_at_value:
            raise ValueError("processed_at is required")

        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
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
                    self.tenant_id,
                    self.project_id,
                    source_hash_value,
                    processed_at_value,
                    now,
                ),
            )

    def delete_external_compaction_state(self, source_session_id: str) -> int:
        """Delete one external compaction state row."""
        source_id = source_session_id.strip()
        if not source_id:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM external_compaction_state "
                "WHERE source_session_id = ? AND tenant_id = ? AND project_id = ?",
                (source_id, self.tenant_id, self.project_id),
            )
            return int(cursor.rowcount or 0)

    @staticmethod
    def _queue_note_key(
        *,
        tier: str,
        line: str,
        source_session_ids: list[str],
    ) -> str:
        normalized_line = " ".join(line.strip().lower().split())
        normalized_sources = sorted({str(item).strip() for item in source_session_ids if item})
        raw = f"{tier.strip().upper()}|{normalized_line}|{json.dumps(normalized_sources)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _retrieval_query_hash(query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def enqueue_external_compaction_queue(
        self,
        notes: list[dict[str, Any]],
        *,
        actor: str,
    ) -> list[dict[str, Any]]:
        self._validate_namespace()
        now = datetime.now(UTC).isoformat()
        created_ids: list[int] = []
        with self._connect() as conn:
            for note in notes:
                tier = str(note.get("tier", "")).strip().upper()
                line = str(note.get("line", "")).strip()
                source_session_ids_raw = note.get("source_session_ids", [])
                source_session_ids = (
                    sorted(
                        {str(item).strip() for item in source_session_ids_raw if str(item).strip()}
                    )
                    if isinstance(source_session_ids_raw, list)
                    else []
                )
                if not tier or not line:
                    continue
                note_key = self._queue_note_key(
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
                        self.tenant_id,
                        self.project_id,
                        note_key,
                        tier,
                        line,
                        json.dumps(source_session_ids, separators=(",", ":")),
                        actor.strip() or "system",
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
                    (self.tenant_id, self.project_id, note_key),
                ).fetchone()
                if row:
                    created_ids.append(int(row["id"]))

        if not created_ids:
            return []
        ids = sorted(set(created_ids))
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, state, tier, line, source_session_ids, actor, action_timestamp,
                       created_at, updated_at, approved_at, rejected_at, applied_at
                FROM external_compaction_queue
                WHERE tenant_id = ? AND project_id = ? AND id IN ({placeholders})
                ORDER BY id ASC
                """,
                (self.tenant_id, self.project_id, *ids),
            ).fetchall()
        return [self._serialize_external_compaction_queue_row(row) for row in rows]

    def list_external_compaction_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        filters = [self.tenant_id, self.project_id]
        query = (
            "SELECT id, state, tier, line, source_session_ids, actor, action_timestamp, "
            "created_at, updated_at, approved_at, rejected_at, applied_at "
            "FROM external_compaction_queue "
            "WHERE tenant_id = ? AND project_id = ?"
        )
        normalized_states = [
            str(item).strip().lower() for item in (states or []) if str(item).strip()
        ]
        if normalized_states:
            placeholders = ",".join("?" for _ in normalized_states)
            query += f" AND state IN ({placeholders})"
            filters.extend(normalized_states)
        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        filters.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(filters)).fetchall()
        return [self._serialize_external_compaction_queue_row(row) for row in rows]

    def update_external_compaction_queue_state(
        self,
        *,
        ids: list[int],
        target_state: str,
        actor: str,
    ) -> dict[str, int]:
        normalized_ids = sorted({int(item) for item in ids if int(item) > 0})
        if not normalized_ids:
            return {"updated": 0, "skipped": 0}

        state = target_state.strip().lower()
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
        now = datetime.now(UTC).isoformat()
        actor_value = actor.strip() or "system"
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
            self.tenant_id,
            self.project_id,
            *from_states,
            *normalized_ids,
        ]
        with self._connect() as conn:
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

    def record_external_compaction_evidence(self, notes: list[dict[str, Any]]) -> int:
        self._validate_namespace()
        now = datetime.now(UTC).isoformat()
        written = 0
        with self._connect() as conn:
            for note in notes:
                tier = str(note.get("tier", "")).strip().upper()
                line = str(note.get("line", "")).strip()
                source_ids_raw = note.get("source_session_ids", [])
                if not isinstance(source_ids_raw, list):
                    source_ids_raw = []
                source_ids = sorted(
                    {str(item).strip() for item in source_ids_raw if str(item).strip()}
                )
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
                            self.tenant_id,
                            self.project_id,
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

    def list_external_compaction_evidence(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, tier, line, source_session_id, merged_at, created_at, updated_at
                FROM external_compaction_evidence
                WHERE tenant_id = ? AND project_id = ?
                ORDER BY merged_at DESC, id DESC
                LIMIT ?
                """,
                (self.tenant_id, self.project_id, max(1, int(limit))),
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

    def record_retrieval_feedback(
        self,
        *,
        query: str,
        chunk_id: UUID,
        score: int,
        actor: str = "user",
        source: str = "cli",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("query is required")
        normalized_score = int(score)
        if normalized_score not in {-1, 1}:
            raise ValueError("score must be -1 (downvote) or 1 (upvote)")
        chunk_id_text = str(chunk_id)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM chunks WHERE id = ? AND tenant_id = ? AND project_id = ? LIMIT 1",
                (chunk_id_text, self.tenant_id, self.project_id),
            ).fetchone()
            if not exists:
                raise ValueError(f"chunk not found for feedback: {chunk_id_text}")

            now = datetime.now(UTC).isoformat()
            row_id = str(uuid4())
            payload = {
                "id": row_id,
                "tenant_id": self.tenant_id,
                "project_id": self.project_id,
                "query_text": normalized_query,
                "query_hash": self._retrieval_query_hash(normalized_query),
                "chunk_id": chunk_id_text,
                "score": normalized_score,
                "actor": str(actor).strip() or "user",
                "source": str(source).strip() or "cli",
                "metadata": metadata or {},
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
                    json.dumps(payload["metadata"], separators=(",", ":")),
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
        return payload

    def list_retrieval_feedback(
        self,
        *,
        limit: int = 100,
        query: str | None = None,
        chunk_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[Any] = [self.tenant_id, self.project_id]
        sql = (
            "SELECT id, query_text, query_hash, chunk_id, score, actor, source, metadata, "
            "created_at, updated_at "
            "FROM retrieval_feedback "
            "WHERE tenant_id = ? AND project_id = ?"
        )
        if query is not None and str(query).strip():
            query_hash = self._retrieval_query_hash(str(query))
            sql += " AND query_hash = ?"
            filters.append(query_hash)
        if chunk_id is not None:
            sql += " AND chunk_id = ?"
            filters.append(str(chunk_id))
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        filters.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(filters)).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            metadata = {}
            try:
                parsed = json.loads(str(row["metadata"]))
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}
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
        self,
        *,
        query: str,
        chunk_ids: list[UUID],
    ) -> dict[UUID, float]:
        unique_chunk_ids = sorted(
            {str(chunk_id) for chunk_id in chunk_ids if str(chunk_id).strip()}
        )
        if not unique_chunk_ids:
            return {}
        query_hash = self._retrieval_query_hash(query)
        placeholders = ",".join("?" for _ in unique_chunk_ids)

        with self._connect() as conn:
            specific_rows = conn.execute(
                f"""
                SELECT chunk_id, AVG(score) AS avg_score
                FROM retrieval_feedback
                WHERE tenant_id = ? AND project_id = ?
                AND query_hash = ?
                AND chunk_id IN ({placeholders})
                GROUP BY chunk_id
                """,
                (self.tenant_id, self.project_id, query_hash, *unique_chunk_ids),
            ).fetchall()
            global_rows = conn.execute(
                f"""
                SELECT chunk_id, AVG(score) AS avg_score
                FROM retrieval_feedback
                WHERE tenant_id = ? AND project_id = ?
                AND chunk_id IN ({placeholders})
                GROUP BY chunk_id
                """,
                (self.tenant_id, self.project_id, *unique_chunk_ids),
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

    def replace_topic_threads(self, threads: list[dict[str, Any]]) -> int:
        now = datetime.now(UTC).isoformat()
        inserted = 0
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM topic_thread_links WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
            )
            conn.execute(
                "DELETE FROM topic_threads WHERE tenant_id = ? AND project_id = ?",
                (self.tenant_id, self.project_id),
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
                        self.tenant_id,
                        self.project_id,
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
                    entry_id = str(link.get("entry_id", "")).strip() or None
                    chunk_id = str(link.get("chunk_id", "")).strip() or None
                    source_session_id = str(link.get("source_session_id", "")).strip() or None
                    if not entry_id and not chunk_id and not source_session_id:
                        continue
                    created_at = str(link.get("created_at", "")).strip() or now
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
                            self.tenant_id,
                            self.project_id,
                            entry_id,
                            chunk_id,
                            source_session_id,
                            created_at,
                        ),
                    )
        return inserted

    def list_topic_threads(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id, title, summary, score, entry_count, source_session_count,
                       last_seen_at, created_at, updated_at
                FROM topic_threads
                WHERE tenant_id = ? AND project_id = ?
                ORDER BY score DESC, last_seen_at DESC, thread_id ASC
                LIMIT ?
                """,
                (self.tenant_id, self.project_id, max(1, int(limit))),
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
        self,
        thread_id: str,
        *,
        limit_links: int = 50,
    ) -> dict[str, Any] | None:
        normalized = thread_id.strip()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT thread_id, title, summary, score, entry_count, source_session_count,
                       last_seen_at, created_at, updated_at
                FROM topic_threads
                WHERE thread_id = ? AND tenant_id = ? AND project_id = ?
                LIMIT 1
                """,
                (normalized, self.tenant_id, self.project_id),
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
                (normalized, self.tenant_id, self.project_id, max(1, int(limit_links))),
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
                    "source_session_id": (
                        str(link["source_session_id"]) if link["source_session_id"] else None
                    ),
                    "created_at": str(link["created_at"]),
                    "entry_content": str(link["entry_content"]) if link["entry_content"] else None,
                    "chunk_content": str(link["chunk_content"]) if link["chunk_content"] else None,
                }
                for link in link_rows
            ],
        }

    def sync_rule_confidence(
        self,
        rules: list[dict[str, Any]],
        *,
        default_confidence: float = 0.6,
        reinforcement_factor: float = 0.15,
    ) -> dict[str, int]:
        now = datetime.now(UTC).isoformat()
        default_value = max(0.0, min(1.0, float(default_confidence)))
        factor = max(0.0, min(1.0, float(reinforcement_factor)))
        inserted = 0
        updated = 0
        seen_rule_ids: set[str] = set()
        with self._connect() as conn:
            for rule in rules:
                rule_id = str(rule.get("rule_id", "")).strip()
                tier = str(rule.get("tier", "")).strip().upper()
                line = str(rule.get("line", "")).strip()
                if not rule_id or not tier or not line:
                    continue
                if rule_id in seen_rule_ids:
                    continue
                seen_rule_ids.add(rule_id)
                row = conn.execute(
                    """
                    SELECT confidence, reinforcement_count
                    FROM rule_confidence
                    WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                    LIMIT 1
                    """,
                    (rule_id, self.tenant_id, self.project_id),
                ).fetchone()
                if row:
                    current = max(0.0, min(1.0, float(row["confidence"])))
                    reinforced = current + ((1.0 - current) * factor)
                    reinforcement_count = int(row["reinforcement_count"]) + 1
                    conn.execute(
                        """
                        UPDATE rule_confidence
                        SET tier = ?, line = ?, confidence = ?, reinforcement_count = ?,
                            last_reinforced_at = ?, is_stale = 0, updated_at = ?
                        WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                        """,
                        (
                            tier,
                            line,
                            max(0.0, min(1.0, reinforced)),
                            reinforcement_count,
                            now,
                            now,
                            rule_id,
                            self.tenant_id,
                            self.project_id,
                        ),
                    )
                    updated += 1
                else:
                    conn.execute(
                        """
                        INSERT INTO rule_confidence (
                            rule_id, tenant_id, project_id, tier, line, confidence,
                            reinforcement_count, last_reinforced_at, last_decayed_at, is_stale,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rule_id,
                            self.tenant_id,
                            self.project_id,
                            tier,
                            line,
                            default_value,
                            1,
                            now,
                            None,
                            0,
                            now,
                            now,
                        ),
                    )
                    inserted += 1
        return {"inserted": inserted, "updated": updated, "total": inserted + updated}

    def list_rule_confidence(
        self,
        *,
        limit: int = 200,
        stale_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT rule_id, tier, line, confidence, reinforcement_count, "
            "last_reinforced_at, last_decayed_at, is_stale, created_at, updated_at "
            "FROM rule_confidence WHERE tenant_id = ? AND project_id = ?"
        )
        params: list[Any] = [self.tenant_id, self.project_id]
        if stale_only:
            query += " AND is_stale = 1"
        query += " ORDER BY confidence ASC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "rule_id": str(row["rule_id"]),
                "tier": str(row["tier"]),
                "line": str(row["line"]),
                "confidence": float(row["confidence"]),
                "reinforcement_count": int(row["reinforcement_count"]),
                "last_reinforced_at": (
                    str(row["last_reinforced_at"]) if row["last_reinforced_at"] else None
                ),
                "last_decayed_at": str(row["last_decayed_at"]) if row["last_decayed_at"] else None,
                "is_stale": bool(int(row["is_stale"])),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def decay_rule_confidence(
        self,
        *,
        half_life_days: float = 45.0,
        stale_after_days: float = 60.0,
    ) -> dict[str, int]:
        now_dt = datetime.now(UTC)
        half_life = max(1.0, float(half_life_days))
        stale_after = max(1.0, float(stale_after_days))
        decayed = 0
        stale_marked = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rule_id, confidence, is_stale, last_reinforced_at, created_at
                FROM rule_confidence
                WHERE tenant_id = ? AND project_id = ?
                """,
                (self.tenant_id, self.project_id),
            ).fetchall()
            for row in rows:
                anchor_raw = row["last_reinforced_at"] or row["created_at"]
                try:
                    anchor_dt = datetime.fromisoformat(str(anchor_raw))
                except ValueError:
                    anchor_dt = now_dt
                elapsed_days = max(0.0, (now_dt - anchor_dt).total_seconds() / 86_400.0)
                current_confidence = max(0.0, min(1.0, float(row["confidence"])))
                decayed_confidence = current_confidence * (0.5 ** (elapsed_days / half_life))
                stale = elapsed_days >= stale_after
                previous_stale = bool(int(row["is_stale"]))
                if stale and not previous_stale:
                    stale_marked += 1
                if abs(decayed_confidence - current_confidence) > 0.0001 or stale != previous_stale:
                    conn.execute(
                        """
                        UPDATE rule_confidence
                        SET confidence = ?, is_stale = ?, last_decayed_at = ?, updated_at = ?
                        WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                        """,
                        (
                            max(0.0, min(1.0, decayed_confidence)),
                            1 if stale else 0,
                            now_dt.isoformat(),
                            now_dt.isoformat(),
                            str(row["rule_id"]),
                            self.tenant_id,
                            self.project_id,
                        ),
                    )
                    decayed += 1
        return {"decayed": decayed, "stale_marked": stale_marked}

    def archive_and_prune_rule_confidence(
        self,
        *,
        max_confidence: float = 0.35,
        stale_only: bool = True,
        dry_run: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        threshold = max(0.0, min(1.0, float(max_confidence)))
        query = (
            "SELECT rule_id, tier, line, confidence, reinforcement_count, "
            "last_reinforced_at, last_decayed_at, is_stale "
            "FROM rule_confidence "
            "WHERE tenant_id = ? AND project_id = ? AND confidence <= ?"
        )
        params: list[Any] = [self.tenant_id, self.project_id, threshold]
        if stale_only:
            query += " AND is_stale = 1"
        query += " ORDER BY confidence ASC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            candidates = [
                {
                    "rule_id": str(row["rule_id"]),
                    "tier": str(row["tier"]),
                    "line": str(row["line"]),
                    "confidence": float(row["confidence"]),
                    "reinforcement_count": int(row["reinforcement_count"]),
                    "last_reinforced_at": (
                        str(row["last_reinforced_at"]) if row["last_reinforced_at"] else None
                    ),
                    "last_decayed_at": (
                        str(row["last_decayed_at"]) if row["last_decayed_at"] else None
                    ),
                    "is_stale": bool(int(row["is_stale"])),
                }
                for row in rows
            ]
            if dry_run or not candidates:
                return candidates

            archived_at = datetime.now(UTC).isoformat()
            for candidate in candidates:
                conn.execute(
                    """
                    INSERT INTO rule_confidence_archive (
                        rule_id, tenant_id, project_id, tier, line, confidence,
                        reinforcement_count, last_reinforced_at, last_decayed_at,
                        is_stale, archived_at, reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate["rule_id"],
                        self.tenant_id,
                        self.project_id,
                        candidate["tier"],
                        candidate["line"],
                        candidate["confidence"],
                        candidate["reinforcement_count"],
                        candidate["last_reinforced_at"],
                        candidate["last_decayed_at"],
                        1 if candidate["is_stale"] else 0,
                        archived_at,
                        "low-confidence prune",
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM rule_confidence
                    WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                    """,
                    (candidate["rule_id"], self.tenant_id, self.project_id),
                )
        return candidates

    def get_rule_confidence_summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_rules,
                    SUM(CASE WHEN is_stale = 1 THEN 1 ELSE 0 END) AS stale_rules,
                    SUM(CASE WHEN confidence <= 0.35 THEN 1 ELSE 0 END) AS low_confidence_rules,
                    AVG(confidence) AS average_confidence,
                    MIN(COALESCE(last_reinforced_at, created_at)) AS oldest_signal_at
                FROM rule_confidence
                WHERE tenant_id = ? AND project_id = ?
                """,
                (self.tenant_id, self.project_id),
            ).fetchone()
        if row is None:
            return {
                "total_rules": 0,
                "stale_rules": 0,
                "low_confidence_rules": 0,
                "average_confidence": 0.0,
                "oldest_signal_at": None,
            }
        return {
            "total_rules": int(row["total_rules"] or 0),
            "stale_rules": int(row["stale_rules"] or 0),
            "low_confidence_rules": int(row["low_confidence_rules"] or 0),
            "average_confidence": float(row["average_confidence"] or 0.0),
            "oldest_signal_at": str(row["oldest_signal_at"]) if row["oldest_signal_at"] else None,
        }

    @staticmethod
    def _serialize_external_compaction_queue_row(row: sqlite3.Row) -> dict[str, Any]:
        source_ids_raw = row["source_session_ids"]
        source_ids: list[str] = []
        if isinstance(source_ids_raw, str):
            try:
                parsed = json.loads(source_ids_raw)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                source_ids = [str(item) for item in parsed if str(item).strip()]
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

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        """Retrieve current background sync status, creating default if not exists."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, tenant_id, project_id, is_running, started_at,
                          completed_at, sessions_processed, learnings_extracted,
                          error_message, pid, updated_at
                   FROM background_sync_status
                   WHERE tenant_id = ? AND project_id = ?
                   ORDER BY updated_at DESC LIMIT 1""",
                (self.tenant_id, self.project_id),
            ).fetchone()

        if not row:
            return BackgroundSyncStatus(tenant_id=self.tenant_id, project_id=self.project_id)

        return BackgroundSyncStatus(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            is_running=bool(row["is_running"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            sessions_processed=row["sessions_processed"],
            learnings_extracted=row["learnings_extracted"],
            error_message=row["error_message"],
            pid=row["pid"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        """Persist background sync status."""
        self._validate_namespace()
        from datetime import UTC

        status.updated_at = datetime.now(UTC)
        # Ensure status object matches our scope
        status.tenant_id = self.tenant_id
        status.project_id = self.project_id

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO background_sync_status
                    (id, tenant_id, project_id, is_running, started_at, completed_at,
                     sessions_processed, learnings_extracted, error_message, pid, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                    is_running=excluded.is_running,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    sessions_processed=excluded.sessions_processed,
                    learnings_extracted=excluded.learnings_extracted,
                    error_message=excluded.error_message,
                    pid=excluded.pid,
                    updated_at=excluded.updated_at""",
                (
                    str(status.id),
                    self.tenant_id,
                    self.project_id,
                    1 if status.is_running else 0,
                    status.started_at.isoformat() if status.started_at else None,
                    status.completed_at.isoformat() if status.completed_at else None,
                    status.sessions_processed,
                    status.learnings_extracted,
                    status.error_message,
                    status.pid,
                    status.updated_at.isoformat(),
                ),
            )

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        """Mark background sync as started and return updated status."""
        status = self.get_background_sync_status()
        status.is_running = True
        status.pid = pid
        status.started_at = datetime.now(UTC)
        status.error_message = None
        self.save_background_sync_status(status)
        return status

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        """Mark background sync as completed and return updated status."""
        status = self.get_background_sync_status()
        status.is_running = False
        status.completed_at = datetime.now(UTC)
        status.sessions_processed = sessions_processed
        status.learnings_extracted = learnings_extracted
        status.error_message = error_message
        status.pid = None
        self.save_background_sync_status(status)
        return status
