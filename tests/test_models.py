from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_recall.storage.models import (
    AgentRecallConfig,
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


def test_storage_config_supports_shared_backend() -> None:
    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {
                    "base_url": "https://memory.example.com",
                    "api_key_env": "TEAM_MEMORY_API_KEY",
                    "timeout_seconds": 6.5,
                    "retry_attempts": 4,
                },
            }
        }
    )

    assert config.storage.backend == "shared"
    assert config.storage.shared.base_url == "https://memory.example.com"
    assert config.storage.shared.api_key_env == "TEAM_MEMORY_API_KEY"
    assert config.storage.shared.timeout_seconds == 6.5
    assert config.storage.shared.retry_attempts == 4
