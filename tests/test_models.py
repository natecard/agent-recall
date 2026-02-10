from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_recall.storage.models import (
    LogEntry,
    LogSource,
    SemanticLabel,
    Session,
    SessionStatus,
)


def test_log_entry_defaults() -> None:
    entry = LogEntry(
        source=LogSource.EXPLICIT,
        content="Observed failure in migration",
        label=SemanticLabel.GOTCHA,
    )

    assert entry.confidence == 1.0
    assert entry.tags == []
    assert entry.metadata == {}


def test_log_entry_is_frozen() -> None:
    entry = LogEntry(
        source=LogSource.EXPLICIT,
        content="Immutable check",
        label=SemanticLabel.NARRATIVE,
    )

    with pytest.raises(ValidationError):
        entry.content = "changed"


def test_session_defaults() -> None:
    session = Session(task="Implement CLI")

    assert session.status == SessionStatus.ACTIVE
    assert session.entry_count == 0
    assert session.ended_at is None


def test_log_entry_validation_errors() -> None:
    with pytest.raises(ValidationError):
        LogEntry(
            source=LogSource.EXPLICIT,
            content="",
            label=SemanticLabel.NARRATIVE,
        )
