from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

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
