from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
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


class SharedBackendUnavailableError(Exception):
    """Raised when the shared storage backend is unreachable."""

    pass


class NamespaceValidationError(Exception):
    """Raised when namespace metadata is missing or invalid for shared storage."""

    pass


def validate_shared_namespace(tenant_id: str, project_id: str) -> None:
    """Validate that tenant and project IDs are properly set for shared storage.

    Raises:
        NamespaceValidationError: If tenant_id or project_id is missing or set to default values.
    """
    if not tenant_id or tenant_id.strip() == "" or tenant_id == "default":
        raise NamespaceValidationError(
            f"Shared storage requires explicit tenant_id. Got: {tenant_id!r}"
        )
    if not project_id or project_id.strip() == "" or project_id == "default":
        raise NamespaceValidationError(
            f"Shared storage requires explicit project_id. Got: {project_id!r}"
        )


class Storage(ABC):
    """
    Abstract Base Class defining the interface for storage backends.

    This class provides a contract for different storage implementations (e.g.,
    local SQLite, shared remote database) to ensure they are interchangeable
    throughout the application.
    """

    @abstractmethod
    def create_session(self, session: Session) -> None:
        """Create a new session."""
        ...

    @abstractmethod
    def get_session(self, session_id: UUID) -> Session | None:
        """Retrieve a session by its ID."""
        ...

    @abstractmethod
    def get_active_session(self) -> Session | None:
        """Retrieve the currently active session, if any."""
        ...

    @abstractmethod
    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        """List recent sessions, optionally filtering by status."""
        ...

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update an existing session's mutable fields."""
        ...

    @abstractmethod
    def append_entry(self, entry: LogEntry) -> None:
        """Append a new log entry to a session."""
        ...

    @abstractmethod
    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        """Retrieve all log entries for a given session."""
        ...

    @abstractmethod
    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        """Retrieve recent log entries matching a set of semantic labels."""
        ...

    @abstractmethod
    def count_log_entries(self) -> int:
        """Return the total number of log entries in the database."""
        ...

    @abstractmethod
    def store_chunk(self, chunk: Chunk) -> None:
        """Store a new chunk of recalled knowledge."""
        ...

    @abstractmethod
    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        """Check if a chunk with the same content and label already exists."""
        ...

    @abstractmethod
    def count_chunks(self) -> int:
        """Return the total number of chunks in the database."""
        ...

    @abstractmethod
    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Perform a full-text search over chunks."""
        ...

    @abstractmethod
    def list_chunks_with_embeddings(self) -> list[Chunk]:
        """List all chunks that have associated embeddings."""
        ...

    @abstractmethod
    def is_session_processed(self, source_session_id: str) -> bool:
        """Check if a source session (from an external agent) has already been processed."""
        ...

    @abstractmethod
    def mark_session_processed(self, source_session_id: str) -> None:
        """Mark a source session as processed to prevent duplicate ingestion."""
        ...

    @abstractmethod
    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove processed-session markers, returning the count of removed items."""
        ...

    @abstractmethod
    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        """Retrieve the incremental sync checkpoint for a source session."""
        ...

    @abstractmethod
    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        """Create or update an incremental sync checkpoint."""
        ...

    @abstractmethod
    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        """Remove sync checkpoints, returning the count of removed items."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """Get high-level statistics about the knowledge base."""
        ...

    @abstractmethod
    def get_last_processed_at(self) -> datetime | None:
        """Get the timestamp of the most recently processed external session."""
        ...

    @abstractmethod
    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Summarize recent source sessions inferred from log entries."""
        ...

    @abstractmethod
    def get_background_sync_status(self) -> BackgroundSyncStatus:
        """Retrieve the current status of the background sync process."""
        ...

    @abstractmethod
    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        """Persist the status of the background sync process."""
        ...

    @abstractmethod
    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        """Mark background sync as started and return the updated status."""
        ...

    @abstractmethod
    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        """Mark background sync as completed and return the updated status."""
        ...
