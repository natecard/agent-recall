from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from agent_recall.storage.models import SessionCheckpoint


def test_save_and_get_checkpoint(storage) -> None:
    """Test saving and retrieving a checkpoint."""
    checkpoint = SessionCheckpoint(
        source_session_id="cursor-abc123",
        last_message_timestamp=datetime(2026, 2, 12, 10, 0, 0, tzinfo=UTC),
        last_message_index=42,
        content_hash="abc123hash",
    )

    storage.save_session_checkpoint(checkpoint)

    retrieved = storage.get_session_checkpoint("cursor-abc123")
    assert retrieved is not None
    assert retrieved.source_session_id == "cursor-abc123"
    assert retrieved.last_message_timestamp == datetime(2026, 2, 12, 10, 0, 0, tzinfo=UTC)
    assert retrieved.last_message_index == 42
    assert retrieved.content_hash == "abc123hash"
    assert isinstance(retrieved.id, UUID)
    assert retrieved.checkpoint_at is not None
    assert retrieved.updated_at is not None


def test_get_checkpoint_not_found(storage) -> None:
    """Test retrieving a non-existent checkpoint returns None."""
    result = storage.get_session_checkpoint("nonexistent-session")
    assert result is None


def test_update_existing_checkpoint(storage) -> None:
    """Test that saving a checkpoint with same session_id updates it."""
    checkpoint1 = SessionCheckpoint(
        source_session_id="cursor-abc123",
        last_message_index=10,
    )
    storage.save_session_checkpoint(checkpoint1)

    checkpoint2 = SessionCheckpoint(
        source_session_id="cursor-abc123",
        last_message_index=20,
        last_message_timestamp=datetime(2026, 2, 12, 11, 0, 0, tzinfo=UTC),
    )
    storage.save_session_checkpoint(checkpoint2)

    retrieved = storage.get_session_checkpoint("cursor-abc123")
    assert retrieved is not None
    assert retrieved.last_message_index == 20
    assert retrieved.last_message_timestamp == datetime(2026, 2, 12, 11, 0, 0, tzinfo=UTC)


def test_checkpoint_optional_fields(storage) -> None:
    """Test that checkpoint fields are optional."""
    checkpoint = SessionCheckpoint(
        source_session_id="cursor-minimal",
    )

    storage.save_session_checkpoint(checkpoint)

    retrieved = storage.get_session_checkpoint("cursor-minimal")
    assert retrieved is not None
    assert retrieved.source_session_id == "cursor-minimal"
    assert retrieved.last_message_timestamp is None
    assert retrieved.last_message_index is None
    assert retrieved.content_hash is None


def test_clear_session_checkpoint_by_id(storage) -> None:
    """Test clearing a specific checkpoint by session ID."""
    checkpoint = SessionCheckpoint(source_session_id="cursor-to-clear")
    storage.save_session_checkpoint(checkpoint)

    removed = storage.clear_session_checkpoints(source_session_id="cursor-to-clear")
    assert removed == 1

    retrieved = storage.get_session_checkpoint("cursor-to-clear")
    assert retrieved is None


def test_clear_session_checkpoints_by_source(storage) -> None:
    """Test clearing all checkpoints for a source."""
    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="cursor-1"))
    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="cursor-2"))
    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="claude-code-1"))

    removed = storage.clear_session_checkpoints(source="cursor")
    assert removed == 2

    assert storage.get_session_checkpoint("cursor-1") is None
    assert storage.get_session_checkpoint("cursor-2") is None
    assert storage.get_session_checkpoint("claude-code-1") is not None


def test_clear_all_session_checkpoints(storage) -> None:
    """Test clearing all checkpoints."""
    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="cursor-1"))
    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="claude-code-1"))

    removed = storage.clear_session_checkpoints()
    assert removed == 2

    assert storage.get_session_checkpoint("cursor-1") is None
    assert storage.get_session_checkpoint("claude-code-1") is None


def test_checkpoint_in_stats(storage) -> None:
    """Test that checkpoint count appears in stats."""
    initial_stats = storage.get_stats()
    initial_count = initial_stats.get("checkpoints", 0)

    storage.save_session_checkpoint(SessionCheckpoint(source_session_id="cursor-stats"))

    stats = storage.get_stats()
    assert stats["checkpoints"] == initial_count + 1


def test_checkpoint_updated_at_changes(storage) -> None:
    """Test that updated_at timestamp changes on update."""
    checkpoint = SessionCheckpoint(source_session_id="cursor-update-test")
    storage.save_session_checkpoint(checkpoint)

    first_retrieved = storage.get_session_checkpoint("cursor-update-test")
    first_updated = first_retrieved.updated_at

    import time

    time.sleep(0.01)  # Ensure time difference

    checkpoint.last_message_index = 100
    storage.save_session_checkpoint(checkpoint)

    second_retrieved = storage.get_session_checkpoint("cursor-update-test")
    second_updated = second_retrieved.updated_at

    assert second_updated > first_updated
