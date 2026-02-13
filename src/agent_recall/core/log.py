from __future__ import annotations

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
    ) -> LogEntry:
        entry = LogEntry(
            session_id=session_id,
            source=source,
            content=content,
            label=label,
            tags=tags or [],
            confidence=confidence,
        )
        self.storage.append_entry(entry)
        return entry
