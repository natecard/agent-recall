from __future__ import annotations

import sqlite3
from pathlib import Path

from agent_recall.storage.sqlite import SQLiteStorage

LEGACY_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    task TEXT NOT NULL,
    summary TEXT,
    entry_count INTEGER DEFAULT 0
);

CREATE TABLE log_entries (
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

CREATE TABLE chunks (
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

CREATE TABLE processed_sessions (
    source_session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
);

CREATE TABLE session_checkpoints (
    id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL UNIQUE,
    last_message_timestamp TEXT,
    last_message_index INTEGER,
    content_hash TEXT,
    checkpoint_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE background_sync_status (
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

PARTIAL_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    task TEXT NOT NULL,
    summary TEXT,
    entry_count INTEGER DEFAULT 0
);

INSERT INTO sessions (id, tenant_id, status, started_at, task)
VALUES ('session-1', 'tenant-a', 'active', '2026-02-01T00:00:00+00:00', 'task');
"""


def _column_names(db_path: Path, table: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row[1]) for row in rows]


def _index_exists(db_path: Path, index_name: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        ).fetchone()
    return row is not None


def test_sqlite_storage_migrates_legacy_schema_before_scope_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(LEGACY_SCHEMA)

    SQLiteStorage(db_path)

    for table in (
        "sessions",
        "log_entries",
        "chunks",
        "processed_sessions",
        "session_checkpoints",
        "background_sync_status",
    ):
        columns = _column_names(db_path, table)
        assert "tenant_id" in columns
        assert "project_id" in columns

    assert _index_exists(db_path, "idx_sessions_scope")
    assert _index_exists(db_path, "idx_entries_scope")
    assert _index_exists(db_path, "idx_chunks_scope")


def test_sqlite_storage_migrates_when_only_project_column_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(PARTIAL_SCHEMA)

    SQLiteStorage(db_path)

    columns = _column_names(db_path, "sessions")
    assert "tenant_id" in columns
    assert "project_id" in columns
    assert _index_exists(db_path, "idx_sessions_scope")

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT tenant_id, project_id FROM sessions WHERE id='session-1'"
        ).fetchone()
    assert row == ("tenant-a", "default")
