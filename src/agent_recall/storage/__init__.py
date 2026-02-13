from __future__ import annotations

from pathlib import Path

from agent_recall.storage.base import Storage
from agent_recall.storage.models import AgentRecallConfig


def create_storage_backend(config: AgentRecallConfig, db_path: Path) -> Storage:
    """
    Factory function to get the configured storage backend.
    """
    tenant_id = config.storage.shared.tenant_id
    project_id = config.storage.shared.project_id

    if config.storage.backend == "local":
        from agent_recall.storage.sqlite import SQLiteStorage

        return SQLiteStorage(db_path, tenant_id=tenant_id, project_id=project_id)

    if config.storage.backend == "shared":
        from agent_recall.storage.remote import RemoteStorage

        return RemoteStorage(config.storage.shared, local_db_path=db_path)

    else:
        raise ValueError(f"Unsupported storage backend: {config.storage.backend}")
