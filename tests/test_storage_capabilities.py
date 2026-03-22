from __future__ import annotations

from pathlib import Path

import pytest

from agent_recall.storage.base import UnsupportedStorageCapabilityError
from agent_recall.storage.models import SharedStorageConfig
from agent_recall.storage.remote import RemoteStorage, _HTTPClient
from agent_recall.storage.sqlite import SQLiteStorage


def _shared_config(base_url: str) -> SharedStorageConfig:
    return SharedStorageConfig(
        base_url=base_url,
        tenant_id="team-tenant",
        project_id="repo-project",
        retry_attempts=1,
        audit_enabled=False,
    )


def _close_http_delegate(storage: RemoteStorage) -> None:
    delegate = storage._delegate
    if isinstance(delegate, _HTTPClient):
        delegate._client.close()


def test_sqlite_storage_declares_optional_capabilities(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.db")
    caps = storage.capabilities
    assert caps.external_compaction_state is True
    assert caps.external_compaction_queue is True
    assert caps.external_compaction_evidence is True
    assert caps.retrieval_feedback is True
    assert caps.topic_threads is True
    assert caps.rule_confidence is True


def test_http_client_declares_optional_domains_unsupported() -> None:
    client = _HTTPClient(_shared_config("https://memory.example.com"))
    caps = client.capabilities
    assert caps.external_compaction_state is False
    assert caps.external_compaction_queue is False
    assert caps.external_compaction_evidence is False
    assert caps.retrieval_feedback is False
    assert caps.topic_threads is False
    assert caps.rule_confidence is False
    client._client.close()


def test_remote_storage_merges_delegate_and_local_capabilities(tmp_path: Path) -> None:
    storage = RemoteStorage(
        _shared_config("https://memory.example.com"),
        local_db_path=tmp_path / "local.db",
    )
    caps = storage.capabilities
    assert caps.external_compaction_state is True
    assert caps.external_compaction_queue is True
    assert caps.external_compaction_evidence is True
    assert caps.retrieval_feedback is True
    assert caps.topic_threads is True
    assert caps.rule_confidence is True
    _close_http_delegate(storage)


def test_remote_storage_raises_for_unsupported_capability_without_local(tmp_path: Path) -> None:
    storage = RemoteStorage(_shared_config("https://memory.example.com"), local_db_path=None)
    with pytest.raises(UnsupportedStorageCapabilityError, match="retrieval_feedback"):
        storage.list_retrieval_feedback(limit=10)
    _close_http_delegate(storage)
