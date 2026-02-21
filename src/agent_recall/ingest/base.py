from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_recall.ingest.health import HealthStatus, SourceHealthResult


class RawToolCall(BaseModel):
    """Normalized tool invocation from any agent transcript."""

    tool: str
    args: dict = Field(default_factory=dict)
    result: str | None = None
    success: bool = True
    duration_ms: int | None = None


class RawMessage(BaseModel):
    """Normalized message from any agent transcript."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime | None = None
    tool_calls: list[RawToolCall] = Field(default_factory=list)


class RawSession(BaseModel):
    """Normalized session from any agent tool."""

    source: str  # "cursor" | "claude-code" | "opencode" | "codex"
    session_id: str
    title: str | None = None
    project_path: Path | None = None
    started_at: datetime
    ended_at: datetime | None = None
    messages: list[RawMessage]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionIngester(ABC):
    """Abstract base for discovering and parsing native agent sessions."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this ingester (e.g. "cursor", "claude-code")."""

    @abstractmethod
    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        """Find session files/databases that may have new content."""

    @abstractmethod
    def parse_session(self, path: Path) -> RawSession:
        """Parse a native session file/database into a normalized RawSession."""

    @abstractmethod
    def get_session_id(self, path: Path) -> str:
        """Extract unique session identifier used for deduplication tracking."""

    def check_health(self) -> SourceHealthResult:
        """Check if the source is available and return health status.

        Default implementation checks if any sessions can be discovered.
        Subclasses can override for more specific health checks.
        """
        start_time = time.time()
        try:
            sessions = self.discover_sessions()
            latency_ms = int((time.time() - start_time) * 1000)
            if sessions:
                last_seen_path = str(sessions[-1]) if sessions else None
                return SourceHealthResult(
                    status=HealthStatus.OK,
                    latency_ms=latency_ms,
                    last_seen_path=last_seen_path,
                )
            return SourceHealthResult(
                status=HealthStatus.DEGRADED,
                latency_ms=latency_ms,
                error_message="No sessions found",
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return SourceHealthResult(
                status=HealthStatus.UNAVAILABLE,
                latency_ms=latency_ms,
                error_message=str(e),
            )
