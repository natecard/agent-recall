from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    CurationStatus,
    LogEntry,
    ScoredChunk,
    SemanticLabel,
    Session,
    SessionCheckpoint,
    SessionStatus,
)


class SharedBackendUnavailableError(Exception):
    """Raised when the shared storage backend is unreachable."""

    pass


class PermissionDeniedError(Exception):
    """Raised when the shared storage role is not permitted to perform an action."""

    pass


class NamespaceValidationError(Exception):
    """Raised when namespace metadata is missing or invalid for shared storage."""

    pass


class UnsupportedStorageCapabilityError(RuntimeError):
    """Raised when an optional storage capability is not supported by a backend."""

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(f"Storage backend does not support capability: {capability}")


@dataclass(frozen=True)
class StorageCapabilities:
    external_compaction_state: bool = False
    external_compaction_queue: bool = False
    external_compaction_evidence: bool = False
    retrieval_feedback: bool = False
    topic_threads: bool = False
    rule_confidence: bool = False

    def merge(self, other: StorageCapabilities | None) -> StorageCapabilities:
        if other is None:
            return self
        return StorageCapabilities(
            external_compaction_state=(
                self.external_compaction_state or other.external_compaction_state
            ),
            external_compaction_queue=(
                self.external_compaction_queue or other.external_compaction_queue
            ),
            external_compaction_evidence=(
                self.external_compaction_evidence or other.external_compaction_evidence
            ),
            retrieval_feedback=(self.retrieval_feedback or other.retrieval_feedback),
            topic_threads=(self.topic_threads or other.topic_threads),
            rule_confidence=(self.rule_confidence or other.rule_confidence),
        )


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

    @property
    def capabilities(self) -> StorageCapabilities:
        """Declared optional capabilities supported by this storage backend."""
        return StorageCapabilities()

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
    def get_entries_by_source_session(
        self,
        source_session_id: str,
        limit: int = 200,
    ) -> list[LogEntry]:
        """Retrieve log entries for a specific imported source session."""
        ...

    @abstractmethod
    def get_entries_by_label(
        self,
        labels: list[SemanticLabel],
        limit: int = 100,
        curation_status: CurationStatus = CurationStatus.APPROVED,
    ) -> list[LogEntry]:
        """Retrieve recent log entries matching a set of semantic labels."""
        ...

    @abstractmethod
    def list_entries_by_curation_status(
        self, status: CurationStatus | None = None, limit: int = 100
    ) -> list[LogEntry]:
        """Retrieve log entries matching a curation status (defaults to approved)."""
        ...

    @abstractmethod
    def update_entry_curation_status(
        self, entry_id: UUID, status: CurationStatus
    ) -> LogEntry | None:
        """Update a log entry's curation status, returning the entry if found."""
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
    def index_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        """Index the vector embedding for an existing chunk.

        This separates vector indexing from content storage, enabling split-backend
        architectures where vector storage can be handled by a specialized service.

        Args:
            chunk_id: The UUID of the chunk to index.
            embedding: The vector embedding to associate with the chunk.
        """
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
    def list_chunks(self) -> list[Chunk]:
        """List all chunks in storage.

        Returns:
            List of all chunks, regardless of whether they have embeddings.
        """
        ...

    @abstractmethod
    def search_chunks_by_embedding(
        self, embedding: list[float], limit: int = 10
    ) -> list[ScoredChunk]:
        """Search chunks by vector embedding similarity.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results to return.

        Returns:
            List of ScoredChunk objects sorted by similarity (highest score first).
        """
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

    # External compaction review queue APIs (optional by backend).
    def list_external_compaction_states(self, limit: int | None = None) -> list[dict[str, str]]:
        raise UnsupportedStorageCapabilityError("external_compaction_state")

    def upsert_external_compaction_state(
        self,
        source_session_id: str,
        *,
        source_hash: str,
        processed_at: str,
    ) -> None:
        raise UnsupportedStorageCapabilityError("external_compaction_state")

    def delete_external_compaction_state(self, source_session_id: str) -> int:
        raise UnsupportedStorageCapabilityError("external_compaction_state")

    def enqueue_external_compaction_queue(
        self,
        notes: list[dict[str, Any]],
        *,
        actor: str,
    ) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("external_compaction_queue")

    def list_external_compaction_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("external_compaction_queue")

    def update_external_compaction_queue_state(
        self,
        *,
        ids: list[int],
        target_state: str,
        actor: str,
    ) -> dict[str, int]:
        raise UnsupportedStorageCapabilityError("external_compaction_queue")

    def record_external_compaction_evidence(self, notes: list[dict[str, Any]]) -> int:
        raise UnsupportedStorageCapabilityError("external_compaction_evidence")

    def list_external_compaction_evidence(self, limit: int = 200) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("external_compaction_evidence")

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
        raise UnsupportedStorageCapabilityError("retrieval_feedback")

    def list_retrieval_feedback(
        self,
        *,
        limit: int = 100,
        query: str | None = None,
        chunk_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("retrieval_feedback")

    def get_retrieval_feedback_scores(
        self,
        *,
        query: str,
        chunk_ids: list[UUID],
    ) -> dict[UUID, float]:
        raise UnsupportedStorageCapabilityError("retrieval_feedback")

    def replace_topic_threads(self, threads: list[dict[str, Any]]) -> int:
        raise UnsupportedStorageCapabilityError("topic_threads")

    def list_topic_threads(self, *, limit: int = 20) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("topic_threads")

    def get_topic_thread(
        self,
        thread_id: str,
        *,
        limit_links: int = 50,
    ) -> dict[str, Any] | None:
        raise UnsupportedStorageCapabilityError("topic_threads")

    def sync_rule_confidence(
        self,
        rules: list[dict[str, Any]],
        *,
        default_confidence: float = 0.6,
        reinforcement_factor: float = 0.15,
    ) -> dict[str, int]:
        raise UnsupportedStorageCapabilityError("rule_confidence")

    def list_rule_confidence(
        self,
        *,
        limit: int = 200,
        stale_only: bool = False,
    ) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("rule_confidence")

    def decay_rule_confidence(
        self,
        *,
        half_life_days: float = 45.0,
        stale_after_days: float = 60.0,
    ) -> dict[str, int]:
        raise UnsupportedStorageCapabilityError("rule_confidence")

    def archive_and_prune_rule_confidence(
        self,
        *,
        max_confidence: float = 0.35,
        stale_only: bool = True,
        dry_run: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        raise UnsupportedStorageCapabilityError("rule_confidence")

    def get_rule_confidence_summary(self) -> dict[str, Any]:
        raise UnsupportedStorageCapabilityError("rule_confidence")
