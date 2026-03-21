from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from agent_recall.storage.http_server import create_shared_backend_wsgi_app
from agent_recall.storage.models import LogEntry, LogSource, SemanticLabel, SharedStorageConfig
from agent_recall.storage.remote import RemoteStorage, _HTTPClient
from agent_recall.storage.sqlite import SQLiteStorage


def _seed_source_entries(storage, source_session_id: str) -> None:
    base = datetime(2026, 3, 21, 12, 0, tzinfo=UTC)
    for index, content in enumerate(
        [
            "first extracted learning",
            "second extracted learning",
            "third extracted learning",
        ]
    ):
        storage.append_entry(
            LogEntry(
                source=LogSource.EXTRACTED,
                source_session_id=source_session_id,
                timestamp=base + timedelta(minutes=index),
                content=content,
                label=SemanticLabel.PATTERN,
            )
        )
    storage.append_entry(
        LogEntry(
            source=LogSource.EXTRACTED,
            source_session_id="other-source-session",
            timestamp=base + timedelta(minutes=5),
            content="other source content",
            label=SemanticLabel.GOTCHA,
        )
    )


def _mount_wsgi_transport(client: _HTTPClient, app) -> None:
    original_client = client._client
    headers = dict(original_client.headers)
    original_client.close()
    client._client = httpx.Client(
        transport=httpx.WSGITransport(app=app),
        base_url="http://test-server",
        headers=headers,
        timeout=5.0,
    )


def _serialize_entries(entries: list[LogEntry]) -> list[tuple[str | None, str, str, str]]:
    return [
        (
            entry.source_session_id,
            entry.timestamp.isoformat(),
            entry.content,
            entry.label.value,
        )
        for entry in entries
    ]


def test_source_session_lookup_parity_local_shared_file_http(tmp_path: Path) -> None:
    source_session_id = "source-session-parity-1"
    tenant_id = "parity-tenant"
    project_id = "parity-project"

    local_storage = SQLiteStorage(tmp_path / "local.db", tenant_id=tenant_id, project_id=project_id)
    _seed_source_entries(local_storage, source_session_id)

    shared_storage = RemoteStorage(
        SharedStorageConfig(
            base_url=f"sqlite://{tmp_path / 'shared'}",
            tenant_id=tenant_id,
            project_id=project_id,
        )
    )
    _seed_source_entries(shared_storage, source_session_id)

    http_db_path = tmp_path / "http" / "state.db"
    http_seed_storage = SQLiteStorage(http_db_path, tenant_id=tenant_id, project_id=project_id)
    _seed_source_entries(http_seed_storage, source_session_id)

    app = create_shared_backend_wsgi_app(
        http_db_path,
        default_tenant_id=tenant_id,
        default_project_id=project_id,
    )
    http_client = _HTTPClient(
        SharedStorageConfig(
            base_url="http://test-server",
            tenant_id=tenant_id,
            project_id=project_id,
        )
    )
    _mount_wsgi_transport(http_client, app)

    local_entries = local_storage.get_entries_by_source_session(source_session_id, limit=2)
    shared_entries = shared_storage.get_entries_by_source_session(source_session_id, limit=2)
    http_entries = http_client.get_entries_by_source_session(source_session_id, limit=2)

    assert _serialize_entries(local_entries) == _serialize_entries(shared_entries)
    assert _serialize_entries(local_entries) == _serialize_entries(http_entries)
    assert [entry.content for entry in local_entries] == [
        "first extracted learning",
        "second extracted learning",
    ]


def test_http_source_session_lookup_respects_scope_and_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_session_id = "scope-session-1"
    db_path = tmp_path / "scoped-state.db"
    tenant_a = SQLiteStorage(db_path, tenant_id="tenant-a", project_id="project-a")
    tenant_b = SQLiteStorage(db_path, tenant_id="tenant-b", project_id="project-b")

    _seed_source_entries(tenant_a, source_session_id)
    _seed_source_entries(tenant_b, source_session_id)

    token = "test-shared-token"
    app = create_shared_backend_wsgi_app(
        db_path,
        default_tenant_id="tenant-a",
        default_project_id="project-a",
        bearer_token=token,
    )

    monkeypatch.setenv("AGENT_RECALL_SHARED_API_KEY", token)
    authorized_client = _HTTPClient(
        SharedStorageConfig(
            base_url="http://test-server",
            tenant_id="tenant-a",
            project_id="project-a",
            require_api_key=True,
        )
    )
    _mount_wsgi_transport(authorized_client, app)
    scoped_entries = authorized_client.get_entries_by_source_session(source_session_id, limit=10)
    assert all(entry.content != "other source content" for entry in scoped_entries)
    assert len(scoped_entries) == 3

    monkeypatch.delenv("AGENT_RECALL_SHARED_API_KEY", raising=False)
    unauthorized_client = _HTTPClient(
        SharedStorageConfig(
            base_url="http://test-server",
            tenant_id="tenant-a",
            project_id="project-a",
            require_api_key=False,
        )
    )
    _mount_wsgi_transport(unauthorized_client, app)
    with pytest.raises(httpx.HTTPStatusError):
        unauthorized_client.get_entries_by_source_session(source_session_id, limit=10)


def test_http_source_session_lookup_normalizes_not_found_to_empty_list(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    app = create_shared_backend_wsgi_app(db_path)
    client = _HTTPClient(
        SharedStorageConfig(
            base_url="http://test-server",
            tenant_id="default",
            project_id="default",
        )
    )
    _mount_wsgi_transport(client, app)

    entries = client.get_entries_by_source_session("missing-source-session", limit=5)
    assert entries == []
