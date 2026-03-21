from __future__ import annotations

from typing import Any
from uuid import UUID

from agent_recall.storage.base import Storage
from agent_recall.storage.models import LogEntry, LogSource, SemanticLabel


class LogWriter:
    def __init__(self, storage: Storage):
        self.storage = storage

    def log(
        self,
        content: str,
        label: SemanticLabel,
        session_id: UUID | None = None,
        tags: list[str] | None = None,
        source: LogSource = LogSource.EXPLICIT,
        confidence: float = 1.0,
        source_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        entry = LogEntry(
            session_id=session_id,
            source=source,
            source_session_id=source_session_id,
            content=content,
            label=label,
            tags=tags or [],
            confidence=confidence,
            metadata=metadata or {},
        )
        self.storage.append_entry(entry)
        return entry
