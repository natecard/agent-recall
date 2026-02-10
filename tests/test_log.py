from __future__ import annotations

from agent_recall.core.log import LogWriter
from agent_recall.core.session import SessionManager
from agent_recall.storage.models import LogSource, SemanticLabel


def test_log_with_session(storage) -> None:
    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)

    session = session_mgr.start("Test task")
    entry = log_writer.log(
        content="Redis needs REDIS_MAX_CONNECTIONS",
        label=SemanticLabel.GOTCHA,
        session_id=session.id,
        tags=["redis", "config"],
    )

    assert entry.content == "Redis needs REDIS_MAX_CONNECTIONS"
    assert entry.label == SemanticLabel.GOTCHA
    assert entry.session_id == session.id
    assert entry.source == LogSource.EXPLICIT

    refreshed = storage.get_session(session.id)
    assert refreshed is not None
    assert refreshed.entry_count == 1


def test_log_without_session(storage) -> None:
    log_writer = LogWriter(storage)

    entry = log_writer.log(
        content="General observation",
        label=SemanticLabel.PATTERN,
    )

    assert entry.session_id is None
