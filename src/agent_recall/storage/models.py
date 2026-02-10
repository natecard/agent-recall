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


class ThemeConfig(BaseModel):
    """CLI theme configuration."""

    name: str = "dark+"


class AgentRecallConfig(BaseModel):
    """Root configuration for .agent/config.yaml."""

    extends: list[str] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
