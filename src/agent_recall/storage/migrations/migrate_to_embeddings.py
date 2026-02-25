from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


def detect_old_schema(storage: SQLiteStorage) -> bool:
    """Detect if the database has old schema (pre-embeddings).

    Returns True if the database is missing embedding columns.
    """
    db_path = storage.db_path
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        except sqlite3.OperationalError:
            return False

    has_embedding = "embedding" in columns
    has_version = "embedding_version" in columns

    return not (has_embedding and has_version)


def create_backup(db_path: Path, verify: bool = True) -> Path | None:
    """Create a timestamped backup of the database.

    Args:
        db_path: Path to the database file
        verify: Whether to verify backup is readable

    Returns:
        Path to the backup file, or None if backup failed
    """
    if not db_path.exists():
        logger.warning(f"Database file not found: {db_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}.db.backup-{timestamp}{db_path.suffix}"

    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

        if verify:
            with sqlite3.connect(backup_path) as conn:
                conn.execute("SELECT COUNT(*) FROM chunks").fetchone()

        return backup_path
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return None


def migrate_database(storage: SQLiteStorage, backup: bool = True) -> dict:
    """Migrate database to support embeddings.

    Args:
        storage: SQLiteStorage instance
        backup: Whether to create a backup before migration

    Returns:
        Dictionary with migration results:
        - backup_path: Path to backup file (or None if no backup created)
        - chunks_before: Number of chunks before migration
        - chunks_after: Number of chunks after migration
        - embedded_before: Number of embedded chunks before
        - embedded_after: Number of embedded chunks after
        - indexed: Number of chunks indexed during migration
        - already_migrated: Whether the database was already migrated
    """
    db_path = storage.db_path

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]

    has_embedding = "embedding" in columns
    has_version = "embedding_version" in columns
    already_migrated = has_embedding and has_version

    indexer = EmbeddingIndexer(storage)
    stats_before = indexer.get_indexing_stats()
    chunks_before = stats_before["total_chunks"]
    embedded_before = stats_before["embedded_chunks"]

    backup_path = None
    if backup and not already_migrated:
        backup_path = create_backup(db_path)
        if backup_path is None:
            return {
                "backup_path": None,
                "chunks_before": chunks_before,
                "chunks_after": chunks_before,
                "embedded_before": embedded_before,
                "embedded_after": embedded_before,
                "indexed": 0,
                "already_migrated": already_migrated,
                "error": "Backup failed",
            }

    if already_migrated:
        logger.info("Database already has embedding columns. Running indexing only.")
        result = indexer.index_missing_embeddings()
        stats_after = indexer.get_indexing_stats()
        return {
            "backup_path": backup_path,
            "chunks_before": chunks_before,
            "chunks_after": stats_after["total_chunks"],
            "embedded_before": embedded_before,
            "embedded_after": stats_after["embedded_chunks"],
            "indexed": result["indexed"],
            "already_migrated": True,
            "error": None,
        }

    with sqlite3.connect(db_path) as conn:
        if "embedding" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            logger.info("Added 'embedding' column to chunks table")

        if "embedding_version" not in columns:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN embedding_version INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("Added 'embedding_version' column to chunks table")

        conn.commit()

    logger.info("Schema migration complete. Starting embedding indexing...")

    result = indexer.index_missing_embeddings()

    stats_after = indexer.get_indexing_stats()

    return {
        "backup_path": backup_path,
        "chunks_before": chunks_before,
        "chunks_after": stats_after["total_chunks"],
        "embedded_before": embedded_before,
        "embedded_after": stats_after["embedded_chunks"],
        "indexed": result["indexed"],
        "already_migrated": False,
        "error": None,
    }


def get_migration_preview(storage: SQLiteStorage) -> dict:
    """Get a preview of what the migration will do without making changes.

    Args:
        storage: SQLiteStorage instance

    Returns:
        Dictionary with preview information:
        - needs_migration: Whether schema migration is needed
        - total_chunks: Total number of chunks
        - embedded_chunks: Number of already embedded chunks
        - pending_chunks: Number of chunks pending embedding
    """
    db_path = storage.db_path

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()]

    has_embedding = "embedding" in columns
    has_version = "embedding_version" in columns
    needs_migration = not (has_embedding and has_version)

    indexer = EmbeddingIndexer(storage)
    stats = indexer.get_indexing_stats()

    return {
        "needs_migration": needs_migration,
        "total_chunks": stats["total_chunks"],
        "embedded_chunks": stats["embedded_chunks"],
        "pending_chunks": stats["pending"],
    }
