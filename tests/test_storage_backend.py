from __future__ import annotations

import pytest

from agent_recall.storage import create_storage_backend
from agent_recall.storage.models import AgentRecallConfig
from agent_recall.storage.sqlite import SQLiteStorage


def test_create_storage_backend_uses_sqlite_for_local(tmp_path) -> None:
    config = AgentRecallConfig.model_validate({"storage": {"backend": "local"}})

    storage = create_storage_backend(config, tmp_path / "state.db")

    assert isinstance(storage, SQLiteStorage)


def test_create_storage_backend_shared_raises_not_implemented(tmp_path) -> None:
    config = AgentRecallConfig.model_validate(
        {
            "storage": {
                "backend": "shared",
                "shared": {"base_url": "https://memory.example.com"},
            }
        }
    )

    with pytest.raises(NotImplementedError, match="Shared storage backend is not implemented yet"):
        create_storage_backend(config, tmp_path / "state.db")
