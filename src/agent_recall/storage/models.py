from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """UTC now with timezone info for stable serialization."""
    return datetime.now(UTC)


class SemanticLabel(StrEnum):
    """Determines compaction behavior and tier promotion."""

    HARD_FAILURE = "hard_failure"
    GOTCHA = "gotcha"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    DECISION_RATIONALE = "decision"
    EXPLORATION = "exploration"
    NARRATIVE = "narrative"


class LogSource(StrEnum):
    """Provenance tracking for log entries."""

    EXPLICIT = "explicit"
    INGESTED = "ingested"
    EXTRACTED = "extracted"
    MANUAL = "manual"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class LogEntry(BaseModel):
    """Atomic unit of captured knowledge. Immutable after creation."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID | None = None
    source: LogSource
    source_session_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    content: str = Field(..., min_length=1, max_length=10_000)
    label: SemanticLabel
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class Session(BaseModel):
    """Container for a work session."""

    id: UUID = Field(default_factory=uuid4)
    status: SessionStatus = SessionStatus.ACTIVE
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    task: str = Field(..., min_length=1)
    summary: str | None = None
    entry_count: int = 0


class ChunkSource(StrEnum):
    LOG_ENTRY = "log_entry"
    COMPACTION = "compaction"
    IMPORT = "import"
    MANUAL = "manual"


class Chunk(BaseModel):
    """Indexed unit for retrieval."""

    id: UUID = Field(default_factory=uuid4)
    source: ChunkSource
    source_ids: list[UUID] = Field(default_factory=list)
    content: str
    label: SemanticLabel
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    token_count: int | None = None
    embedding: list[float] | None = None


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(
        default="anthropic",
        description=(
            "LLM provider: anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible"
        ),
    )
    model: str = Field(default="claude-sonnet-4-20250514", description="Model name/identifier")
    base_url: str | None = Field(
        default=None,
        description="API base URL (required for openai-compatible; optional for local providers)",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable for API key (optional override)",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")
    timeout: float = Field(default=120.0, gt=0, description="Request timeout in seconds")


class CompactionConfig(BaseModel):
    """Compaction trigger thresholds."""

    max_recent_tokens: int = 1500
    max_sessions_before_compact: int = 5
    promote_pattern_after_occurrences: int = 3
    archive_sessions_older_than_days: int = 30


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    backend: str = "fts5"
    top_k: int = 5
    embedding_enabled: bool = False
    embedding_dimensions: int = Field(default=64, ge=8, le=4096)


class ThemeConfig(BaseModel):
    """CLI theme configuration."""

    name: str = "dark+"


class SessionCheckpoint(BaseModel):
    """Checkpoint for incremental session sync.

    Tracks progress within a session to enable delta-based ingestion.
    Only new messages beyond the checkpoint will be processed.
    """

    id: UUID = Field(default_factory=uuid4)
    source_session_id: str = Field(
        ..., description="Session identifier from source (e.g., cursor-abc123)"
    )
    last_message_timestamp: datetime | None = Field(
        default=None,
        description="Timestamp of last processed message (for timestamp-based sources)",
    )
    last_message_index: int | None = Field(
        default=None,
        description="Index of last processed message (for index-based sources)",
    )
    content_hash: str | None = Field(
        default=None,
        description="Hash of processed content for detecting changes",
    )
    checkpoint_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    model_config = ConfigDict(frozen=False)


class BackgroundSyncStatus(BaseModel):
    """Status tracking for background sync operations.

    Persisted to enable status queries and prevent duplicate sync races.
    """

    id: UUID = Field(default_factory=uuid4)
    is_running: bool = Field(default=False, description="Whether sync is currently active")
    started_at: datetime | None = Field(default=None, description="When current/last sync started")
    completed_at: datetime | None = Field(default=None, description="When last sync completed")
    sessions_processed: int = Field(default=0, description="Number of sessions processed")
    learnings_extracted: int = Field(default=0, description="Number of learnings extracted")
    error_message: str | None = Field(default=None, description="Error message if sync failed")
    pid: int | None = Field(default=None, description="Process ID of running sync")
    updated_at: datetime = Field(default_factory=utcnow)

    model_config = ConfigDict(frozen=False)


class AgentRecallConfig(BaseModel):
    """Root configuration for .agent/config.yaml."""

    extends: list[str] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
