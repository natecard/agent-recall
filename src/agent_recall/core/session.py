from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from agent_recall.storage.base import Storage
from agent_recall.storage.models import Session, SessionStatus


def utcnow() -> datetime:
    return datetime.now(UTC)


class SessionManager:
    def __init__(self, storage: Storage):
        self.storage = storage

    def start(self, task: str) -> Session:
        active = self.storage.get_active_session()
        if active:
            active.status = SessionStatus.ABANDONED
            active.ended_at = utcnow()
            self.storage.update_session(active)

        session = Session(task=task)
        self.storage.create_session(session)
        return session

    def end(self, session_id: UUID, summary: str) -> Session:
        session = self.storage.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.status = SessionStatus.COMPLETED
        session.ended_at = utcnow()
        session.summary = summary
        self.storage.update_session(session)
        return session

    def get_active(self) -> Session | None:
        return self.storage.get_active_session()
