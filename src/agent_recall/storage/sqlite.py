from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from agent_recall.storage.base import Storage
from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    ChunkSource,
    LogEntry,
    LogSource,
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
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    task TEXT NOT NULL,
    summary TEXT,
    entry_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS log_entries (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    source TEXT NOT NULL,
    source_session_id TEXT,
    timestamp TEXT NOT NULL,
    content TEXT NOT NULL,
    label TEXT NOT NULL,
    tags TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    content TEXT NOT NULL,
    label TEXT NOT NULL,
    tags TEXT NOT NULL,
    created_at TEXT NOT NULL,
    token_count INTEGER,
    embedding BLOB
);

{CHUNKS_FTS_SCHEMA}

CREATE TABLE IF NOT EXISTS processed_sessions (
    source_session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_session ON log_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_entries_label ON log_entries(label);
CREATE INDEX IF NOT EXISTS idx_chunks_label ON chunks(label);

CREATE TABLE IF NOT EXISTS session_checkpoints (
    id TEXT PRIMARY KEY,
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
    is_running INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    sessions_processed INTEGER NOT NULL DEFAULT 0,
    learnings_extracted INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    pid INTEGER,
    updated_at TEXT NOT NULL
);
"""


class SQLiteStorage(Storage):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

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
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions (
                       id, status, started_at, ended_at, task, summary, entry_count
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(session.id),
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
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (str(session_id),)).fetchone()
        return self._row_to_session(row) if row else None

    def get_active_session(self) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY started_at DESC LIMIT 1",
                (SessionStatus.ACTIVE.value,),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE status = ? ORDER BY started_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def update_session(self, session: Session) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions SET status=?, ended_at=?, summary=?, entry_count=?
                   WHERE id=?""",
                (
                    session.status.value,
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.summary,
                    session.entry_count,
                    str(session.id),
                ),
            )

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=UUID(row["id"]),
            status=SessionStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            task=row["task"],
            summary=row["summary"],
            entry_count=row["entry_count"],
        )

    def append_entry(self, entry: LogEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO log_entries
                   (
                       id, session_id, source, source_session_id, timestamp,
                       content, label, tags, confidence, metadata
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(entry.id),
                    str(entry.session_id) if entry.session_id else None,
                    entry.source.value,
                    entry.source_session_id,
                    entry.timestamp.isoformat(),
                    entry.content,
                    entry.label.value,
                    json.dumps(entry.tags),
                    entry.confidence,
                    json.dumps(entry.metadata),
                ),
            )
            if entry.session_id:
                conn.execute(
                    "UPDATE sessions SET entry_count = entry_count + 1 WHERE id = ?",
                    (str(entry.session_id),),
                )

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM log_entries WHERE session_id = ? ORDER BY timestamp",
                (str(session_id),),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        if not labels:
            return []

        placeholders = ",".join("?" * len(labels))
        with self._connect() as conn:
            rows = conn.execute(
                (
                    f"SELECT * FROM log_entries WHERE label IN ({placeholders}) "
                    "ORDER BY timestamp DESC LIMIT ?"
                ),
                [label.value for label in labels] + [limit],
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count_log_entries(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM log_entries").fetchone()
        return int(row["n"]) if row else 0

    def _row_to_entry(self, row: sqlite3.Row) -> LogEntry:
        return LogEntry(
            id=UUID(row["id"]),
            session_id=UUID(row["session_id"]) if row["session_id"] else None,
            source=LogSource(row["source"]),
            source_session_id=row["source_session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            content=row["content"],
            label=SemanticLabel(row["label"]),
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]),
        )

    def store_chunk(self, chunk: Chunk) -> None:
        for attempt in range(2):
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO chunks
                           (
                               id, source, source_ids, content, label,
                               tags, created_at, token_count, embedding
                           )
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(chunk.id),
                            chunk.source.value,
                            json.dumps([str(item) for item in chunk.source_ids]),
                            chunk.content,
                            chunk.label.value,
                            json.dumps(chunk.tags),
                            chunk.created_at.isoformat(),
                            chunk.token_count,
                            self._serialize_embedding(chunk.embedding),
                        ),
                    )
                return
            except sqlite3.DatabaseError as exc:
                if attempt == 0 and self._is_chunks_fts_corruption(exc):
                    self.rebuild_chunks_fts()
                    continue
                raise

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
                "SELECT 1 FROM chunks WHERE content = ? AND label = ? LIMIT 1",
                (content, label.value),
            ).fetchone()
        return row is not None

    def count_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
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
                       ORDER BY bm25(chunks_fts) LIMIT ?""",
                    (query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM chunks
                   WHERE embedding IS NOT NULL
                   ORDER BY created_at DESC, id ASC"""
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=UUID(row["id"]),
            source=ChunkSource(row["source"]),
            source_ids=[UUID(item) for item in json.loads(row["source_ids"])],
            content=row["content"],
            label=SemanticLabel(row["label"]),
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            token_count=row["token_count"],
            embedding=self._deserialize_embedding(row["embedding"]),
        )

    def is_session_processed(self, source_session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM processed_sessions WHERE source_session_id = ?",
                (source_session_id,),
            ).fetchone()
        return row is not None

    def mark_session_processed(self, source_session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO processed_sessions "
                    "(source_session_id, processed_at) VALUES (?, ?)"
                ),
                (source_session_id, datetime.now(UTC).isoformat()),
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
                    "DELETE FROM processed_sessions WHERE source_session_id = ?",
                    (source_session_id,),
                )
                return int(cursor.rowcount or 0)

            if source:
                normalized = source.strip().lower().replace("_", "-")
                pattern = f"{normalized}-%"
                cursor = conn.execute(
                    "DELETE FROM processed_sessions WHERE source_session_id LIKE ?",
                    (pattern,),
                )
                return int(cursor.rowcount or 0)

            cursor = conn.execute("DELETE FROM processed_sessions")
            return int(cursor.rowcount or 0)

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        """Retrieve checkpoint for a source session, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, source_session_id, last_message_timestamp,
                          last_message_index, content_hash, checkpoint_at, updated_at
                   FROM session_checkpoints
                   WHERE source_session_id = ?""",
                (source_session_id,),
            ).fetchone()

        if not row:
            return None

        return SessionCheckpoint(
            id=UUID(row["id"]),
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
        now = datetime.now(UTC)
        checkpoint.updated_at = now

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO session_checkpoints
                    (id, source_session_id, last_message_timestamp, last_message_index,
                     content_hash, checkpoint_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_session_id) DO UPDATE SET
                    last_message_timestamp=excluded.last_message_timestamp,
                    last_message_index=excluded.last_message_index,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at""",
                (
                    str(checkpoint.id),
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
                    "DELETE FROM session_checkpoints WHERE source_session_id = ?",
                    (source_session_id,),
                )
                return int(cursor.rowcount or 0)

            if source:
                normalized = source.strip().lower().replace("_", "-")
                pattern = f"{normalized}-%"
                cursor = conn.execute(
                    "DELETE FROM session_checkpoints WHERE source_session_id LIKE ?",
                    (pattern,),
                )
                return int(cursor.rowcount or 0)

            cursor = conn.execute("DELETE FROM session_checkpoints")
            return int(cursor.rowcount or 0)

    def get_stats(self) -> dict[str, int]:
        """Get knowledge base statistics."""
        with self._connect() as conn:
            stats: dict[str, int] = {}

            row = conn.execute("SELECT COUNT(*) AS count FROM processed_sessions").fetchone()
            stats["processed_sessions"] = int(row["count"]) if row else 0

            row = conn.execute("SELECT COUNT(*) AS count FROM log_entries").fetchone()
            stats["log_entries"] = int(row["count"]) if row else 0

            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            stats["chunks"] = int(row["count"]) if row else 0

            row = conn.execute("SELECT COUNT(*) AS count FROM session_checkpoints").fetchone()
            stats["checkpoints"] = int(row["count"]) if row else 0

        return stats

    def get_last_processed_at(self) -> datetime | None:
        """Return the most recent processed-session timestamp."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(processed_at) AS last_processed FROM processed_sessions"
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
                GROUP BY source_session_id
                ORDER BY last_timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            results: list[dict[str, Any]] = []
            for session_row in sessions:
                source_session_id = str(session_row["source_session_id"])
                highlights = conn.execute(
                    """
                    SELECT label, content
                    FROM log_entries
                    WHERE source_session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 5
                    """,
                    (source_session_id,),
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

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        """Retrieve current background sync status, creating default if not exists."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, is_running, started_at, completed_at, sessions_processed,
                          learnings_extracted, error_message, pid, updated_at
                   FROM background_sync_status
                   ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()

        if not row:
            return BackgroundSyncStatus()

        return BackgroundSyncStatus(
            id=UUID(row["id"]),
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
        from datetime import UTC

        status.updated_at = datetime.now(UTC)

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO background_sync_status
                    (id, is_running, started_at, completed_at, sessions_processed,
                     learnings_extracted, error_message, pid, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
