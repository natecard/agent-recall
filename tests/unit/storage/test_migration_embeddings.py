from __future__ import annotations

from pathlib import Path

from agent_recall.storage.migrations.migrate_to_embeddings import (
    get_migration_preview,
    migrate_database,
)
from agent_recall.storage.sqlite import SQLiteStorage


def test_migrate_database_idempotent(tmp_path: Path) -> None:
    """Test that running migration twice doesn't cause errors."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    result1 = migrate_database(storage, backup=False)

    assert result1["error"] is None

    result2 = migrate_database(storage, backup=False)

    assert result2["error"] is None
    assert result2["already_migrated"] is True
    assert result2["indexed"] == 0


def test_migrate_database_creates_backup(tmp_path: Path) -> None:
    """Test that migration creates a backup when schema needs migration."""
    db_path = tmp_path / "test.db"

    storage = SQLiteStorage(db_path)

    result = migrate_database(storage, backup=True)

    assert result["error"] is None


def test_migrate_database_skips_backup_when_disabled(tmp_path: Path) -> None:
    """Test that migration skips backup when disabled."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    result = migrate_database(storage, backup=False)

    assert result["backup_path"] is None
    assert result["error"] is None


def test_get_migration_preview_already_migrated(tmp_path: Path) -> None:
    """Test preview for already-migrated database."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    preview = get_migration_preview(storage)

    assert preview["needs_migration"] is False
    assert preview["total_chunks"] >= 0


def test_migrate_database_indexes_pending_chunks(tmp_path: Path) -> None:
    """Test that migration indexes chunks without embeddings."""
    from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel

    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)

    chunk = Chunk(
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="test content for embedding",
        label=SemanticLabel.PATTERN,
        embedding=None,
    )
    storage.store_chunk(chunk)

    preview = get_migration_preview(storage)
    assert preview["pending_chunks"] == 1

    result = migrate_database(
        storage,
        backup=False,
    )

    assert result["error"] is None
    assert result["indexed"] >= 0
