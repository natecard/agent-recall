from __future__ import annotations

from agent_recall.core.session import SessionManager
from agent_recall.storage.models import SessionStatus


def test_start_session(storage) -> None:
    mgr = SessionManager(storage)
    session = mgr.start("Test task")

    assert session.task == "Test task"
    assert session.status == SessionStatus.ACTIVE


def test_end_session(storage) -> None:
    mgr = SessionManager(storage)
    session = mgr.start("Test task")
    ended = mgr.end(session.id, "Completed successfully")

    assert ended.status == SessionStatus.COMPLETED
    assert ended.summary == "Completed successfully"
    assert ended.ended_at is not None


def test_abandon_previous_session(storage) -> None:
    mgr = SessionManager(storage)
    session1 = mgr.start("First task")
    _session2 = mgr.start("Second task")

    retrieved = storage.get_session(session1.id)
    assert retrieved is not None
    assert retrieved.status == SessionStatus.ABANDONED
