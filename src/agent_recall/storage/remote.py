from __future__ import annotations

from datetime import datetime
from typing import Any, NoReturn
from uuid import UUID

from agent_recall.storage.base import Storage
from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    LogEntry,
    SemanticLabel,
    Session,
    SessionCheckpoint,
    SessionStatus,
    SharedStorageConfig,
)


class RemoteStorage(Storage):
    """
    Storage implementation for a shared remote backend.

    This class provides a contract for different storage implementations (e.g.,
    local SQLite, shared remote database) to ensure they are interchangeable
    throughout the application.
    """

    def __init__(self, config: SharedStorageConfig) -> None:
        self.config = config
        self._raise_not_implemented()

    @staticmethod
    def _raise_not_implemented() -> NoReturn:
        raise NotImplementedError(
            "Shared storage backend is not implemented yet. Keep "
            "`storage.backend: local` until the shared backend client is shipped."
        )

    def create_session(self, session: Session) -> None:
        """Create a new session."""
        self._raise_not_implemented()

    def get_session(self, session_id: UUID) -> Session | None:
        """Retrieve a session by its ID."""
        self._raise_not_implemented()

    def get_active_session(self) -> Session | None:
        """Retrieve the currently active session, if any."""
        self._raise_not_implemented()

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        """List recent sessions, optionally filtering by status."""
        self._raise_not_implemented()

    def update_session(self, session: Session) -> None:
        """Update an existing session's mutable fields."""
        self._raise_not_implemented()

    def append_entry(self, entry: LogEntry) -> None:
        """Append a new log entry to a session."""
        self._raise_not_implemented()

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        """Retrieve all log entries for a given session."""
        self._raise_not_implemented()

    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        """Retrieve recent log entries matching a set of semantic labels."""
        self._raise_not_implemented()

    def count_log_entries(self) -> int:
        """Return the total number of log entries in the database."""
        self._raise_not_implemented()

    def store_chunk(self, chunk: Chunk) -> None:
        """Store a new chunk of recalled knowledge."""
        self._raise_not_implemented()

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        """Check if a chunk with the same content and label already exists."""
        self._raise_not_implemented()

    def count_chunks(self) -> int:
        """Return the total number of chunks in the database."""
        self._raise_not_implemented()

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Perform a full-text search over chunks."""
        self._raise_not_implemented()

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        """List all chunks that have associated embeddings."""
        self._raise_not_implemented()

    def is_session_processed(self, source_session_id: str) -> bool:
        """Check if a source session (from an external agent) has already been processed."""
        self._raise_not_implemented()

    def mark_session_processed(self, source_session_id: str) -> None:
        """Mark a source session as processed to prevent duplicate ingestion."""
        self._raise_not_implemented()

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove processed-session markers, returning the count of removed items."""
        self._raise_not_implemented()

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        """Retrieve the incremental sync checkpoint for a source session."""
        self._raise_not_implemented()

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """Create or update an incremental sync checkpoint."""
        self._raise_not_implemented()

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove sync checkpoints, returning the count of removed items."""
        self._raise_not_implemented()

    def get_stats(self) -> dict[str, int]:
        """Get high-level statistics about the knowledge base."""
        self._raise_not_implemented()

    def get_last_processed_at(self) -> datetime | None:
        """Get the timestamp of the most recently processed external session."""
        self._raise_not_implemented()

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Summarize recent source sessions inferred from log entries."""
        self._raise_not_implemented()

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        """Retrieve the current status of the background sync process."""
        self._raise_not_implemented()

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        """Persist the status of the background sync process."""
        self._raise_not_implemented()

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        """Mark background sync as started and return the updated status."""
        self._raise_not_implemented()

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        """Mark background sync as completed and return the updated status."""
        self._raise_not_implemented()
