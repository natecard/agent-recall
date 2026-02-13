from __future__ import annotations

from pathlib import Path

from agent_recall.storage.base import Storage
from agent_recall.storage.models import AgentRecallConfig


def create_storage_backend(config: AgentRecallConfig, db_path: Path) -> Storage:
    """
    Factory function to get the configured storage backend.
    """
    if config.storage.backend == "local":
        from agent_recall.storage.sqlite import SQLiteStorage

        return SQLiteStorage(db_path)
    # This is where a remote storage implementation would go
    # elif config.storage.backend == "remote":
    #     from agent_recall.storage.remote import RemoteStorage
    #     return RemoteStorage()
    else:
        raise ValueError(f"Unsupported storage backend: {config.storage.backend}")
