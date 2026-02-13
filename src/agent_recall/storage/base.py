from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    LogEntry,
    SemanticLabel,
    Session,
    SessionCheckpoint,
    SessionStatus,
)


class Storage(Protocol):
    """
    Protocol defining the interface for storage backends.

    This class provides a contract for different storage implementations (e.g.,
    local SQLite, shared remote database) to ensure they are interchangeable
    throughout the application.
    """

    def create_session(self, session: Session) -> None:
        """Create a new session."""
        ...

    def get_session(self, session_id: UUID) -> Session | None:
        """Retrieve a session by its ID."""
        ...

    def get_active_session(self) -> Session | None:
        """Retrieve the currently active session, if any."""
        ...

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        """List recent sessions, optionally filtering by status."""
        ...

    def update_session(self, session: Session) -> None:
        """Update an existing session's mutable fields."""
        ...

    def append_entry(self, entry: LogEntry) -> None:
        """Append a new log entry to a session."""
        ...

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        """Retrieve all log entries for a given session."""
        ...

    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        """Retrieve recent log entries matching a set of semantic labels."""
        ...

    def count_log_entries(self) -> int:
        """Return the total number of log entries in the database."""
        ...

    def store_chunk(self, chunk: Chunk) -> None:
        """Store a new chunk of recalled knowledge."""
        ...

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        """Check if a chunk with the same content and label already exists."""
        ...

    def count_chunks(self) -> int:
        """Return the total number of chunks in the database."""
        ...

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Perform a full-text search over chunks."""
        ...

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        """List all chunks that have associated embeddings."""
        ...

    def is_session_processed(self, source_session_id: str) -> bool:
        """Check if a source session (from an external agent) has already been processed."""
        ...

    def mark_session_processed(self, source_session_id: str) -> None:
        """Mark a source session as processed to prevent duplicate ingestion."""
        ...

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove processed-session markers, returning the count of removed items."""
        ...

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        """Retrieve the incremental sync checkpoint for a source session."""
        ...

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """Create or update an incremental sync checkpoint."""
        ...

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove sync checkpoints, returning the count of removed items."""
        ...

    def get_stats(self) -> dict[str, int]:
        """Get high-level statistics about the knowledge base."""
        ...

    def get_last_processed_at(self) -> datetime | None:
        """Get the timestamp of the most recently processed external session."""
        ...

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Summarize recent source sessions inferred from log entries."""
        ...

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        """Retrieve the current status of the background sync process."""
        ...

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        """Persist the status of the background sync process."""
        ...

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        """Mark background sync as started and return the updated status."""
        ...

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        """Mark background sync as completed and return the updated status."""
        ...
