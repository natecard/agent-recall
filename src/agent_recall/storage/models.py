from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
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


class CurationStatus(StrEnum):
    """Curation workflow status for extracted learnings."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class LogEntry(BaseModel):
    """Atomic unit of captured knowledge. Immutable after creation."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "default"
    project_id: str = "default"
    session_id: UUID | None = None
    source: LogSource
    source_session_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    content: str = Field(..., min_length=1, max_length=10_000)
    label: SemanticLabel
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    curation_status: CurationStatus = CurationStatus.APPROVED
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class Session(BaseModel):
    """Container for a work session."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "default"
    project_id: str = "default"
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
    tenant_id: str = "default"
    project_id: str = "default"
    source: ChunkSource
    source_ids: list[UUID] = Field(default_factory=list)
    content: str
    label: SemanticLabel
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    token_count: int | None = None
    embedding: list[float] | None = None


class AuditAction(StrEnum):
    """Audit actions for shared storage mutations."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    CLEAR = "clear"
    START = "start"
    COMPLETE = "complete"


class AuditEvent(BaseModel):
    """Immutable audit event for shared storage mutations."""

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "default"
    project_id: str = "default"
    actor: str = "system"
    action: AuditAction
    resource_type: str
    resource_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


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
    index_decision_entries: bool = True
    index_decision_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    index_exploration_entries: bool = True
    index_exploration_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    index_narrative_entries: bool = False
    index_narrative_min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    archive_sessions_older_than_days: int = 30


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    backend: Literal["fts5", "hybrid"] = "fts5"
    top_k: int = Field(default=5, ge=1)
    fusion_k: int = Field(default=60, ge=1)
    rerank_enabled: bool = False
    rerank_candidate_k: int = Field(default=20, ge=1)
    embedding_enabled: bool = False
    embedding_dimensions: int = Field(default=64, ge=8, le=4096)


class SharedStorageConfig(BaseModel):
    """Connection settings for the shared storage backend."""

    base_url: str | None = Field(
        default=None,
        description="Shared backend base URL (e.g., https://memory.example.com)",
    )
    api_key_env: str = Field(
        default="AGENT_RECALL_SHARED_API_KEY",
        min_length=1,
        description="Environment variable containing shared backend API token",
    )
    require_api_key: bool = Field(
        default=False,
        description="Fail if api_key_env is unset/empty when using HTTP shared backend",
    )
    role: Literal["admin", "writer", "reader"] = Field(
        default="writer",
        description="Shared backend role for RBAC enforcement",
    )
    allow_promote: bool = Field(
        default=True,
        description="Allow promote/dedup actions when role permits",
    )
    audit_enabled: bool = Field(
        default=True,
        description="Emit audit events for shared backend mutations",
    )
    audit_actor: str = Field(
        default="system",
        min_length=1,
        description="Actor name recorded in shared backend audit events",
    )
    timeout_seconds: float = Field(default=10.0, gt=0.0, description="HTTP timeout in seconds")
    retry_attempts: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Retry attempts for transient shared backend failures",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for isolation (e.g., org-123)",
    )
    project_id: str = Field(
        default="default",
        description="Project identifier for isolation (e.g., repo-abc)",
    )


class StorageConfig(BaseModel):
    """Storage backend selection and connection settings."""

    backend: Literal["local", "shared"] = "local"
    shared: SharedStorageConfig = Field(default_factory=SharedStorageConfig)


class ThemeConfig(BaseModel):
    """CLI theme configuration."""

    name: str = "dark+"


class RalphNotificationEvent(StrEnum):
    """Events that can trigger Ralph notifications."""

    ITERATION_COMPLETE = "iteration_complete"
    VALIDATION_FAILED = "validation_failed"
    LOOP_FINISHED = "loop_finished"
    BUDGET_EXCEEDED = "budget_exceeded"


class RalphLoopConfig(BaseModel):
    """Configuration for Ralph loop control and defaults."""

    enabled: bool = False
    max_iterations: int = Field(default=10, ge=1)
    sleep_seconds: int = Field(default=2, ge=0)
    compact_mode: Literal["always", "on-failure", "off"] = "always"
    selected_prd_ids: list[str] | None = Field(
        default=None,
        description="Optional. PRD item IDs to include; None means all items (model decides)",
    )

    class NotificationConfig(BaseModel):
        """Notification preferences for Ralph loop events."""

        enabled: bool = False
        events: list[RalphNotificationEvent] = Field(
            default_factory=lambda: [
                RalphNotificationEvent.ITERATION_COMPLETE,
                RalphNotificationEvent.VALIDATION_FAILED,
                RalphNotificationEvent.LOOP_FINISHED,
            ]
        )

    class ForecastConfig(BaseModel):
        """Configuration for Ralph forecast generation."""

        window: int = Field(default=5, ge=0)
        use_llm: bool = False
        llm_on_consecutive_failures: int = Field(default=2, ge=1)
        llm_model: str | None = None

    class SynthesisConfig(BaseModel):
        """Configuration for Ralph climate synthesis."""

        auto_after_loop: bool = True
        max_guardrails: int = Field(default=30, ge=1)
        max_style: int = Field(default=30, ge=1)

    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)


class AdapterConfig(BaseModel):
    """Context adapter configuration."""

    enabled: bool = False
    output_dir: str = ".agent/context"
    token_budget: int | None = Field(
        default=None,
        description="Optional token budget for adapter payload context",
    )
    per_adapter_token_budget: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-adapter token budgets (tokens)",
    )
    per_provider_token_budget: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-provider token budgets (tokens)",
    )
    per_model_token_budget: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-model token budgets (tokens)",
    )


class SessionCheckpoint(BaseModel):
    """Checkpoint for incremental session sync.

    Tracks progress within a session to enable delta-based ingestion.
    Only new messages beyond the checkpoint will be processed.
    """

    id: UUID = Field(default_factory=uuid4)
    tenant_id: str = "default"
    project_id: str = "default"
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
    tenant_id: str = "default"
    project_id: str = "default"
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
    storage: StorageConfig = Field(default_factory=StorageConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    ralph: RalphLoopConfig = Field(default_factory=RalphLoopConfig)
    adapters: AdapterConfig = Field(default_factory=AdapterConfig)
