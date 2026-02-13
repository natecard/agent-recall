import uuid
from datetime import UTC, datetime

import httpx
import pytest
import respx

from agent_recall.storage.models import (
    SemanticLabel,
    Session,
    SessionStatus,
    SharedStorageConfig,
)
from agent_recall.storage.remote import RemoteStorage, SharedBackendUnavailableError


@pytest.fixture
def http_config():
    return SharedStorageConfig(
        base_url="http://test-server",
        retry_attempts=1,
        timeout_seconds=1.0,
    )


@pytest.fixture
def storage(http_config):
    return RemoteStorage(http_config)


@respx.mock
def test_create_session(storage):
    session = Session(
        id=uuid.uuid4(),
        task="test task",
        status=SessionStatus.ACTIVE,
        started_at=datetime.now(UTC),
    )

    route = respx.post("http://test-server/sessions").mock(return_value=httpx.Response(201))

    storage.create_session(session)

    assert route.called
    assert route.calls.last.request.content == session.model_dump_json().encode()


@respx.mock
def test_get_session(storage):
    session_id = uuid.uuid4()
    session_data = {
        "id": str(session_id),
        "task": "test task",
        "status": "active",
        "started_at": datetime.now(UTC).isoformat(),
        "entry_count": 0,
    }

    respx.get(f"http://test-server/sessions/{session_id}").mock(
        return_value=httpx.Response(200, json=session_data)
    )

    session = storage.get_session(session_id)
    assert session is not None
    assert session.id == session_id
    assert session.task == "test task"


@respx.mock
def test_get_session_not_found(storage):
    session_id = uuid.uuid4()
    respx.get(f"http://test-server/sessions/{session_id}").mock(
        return_value=httpx.Response(404)
    )

    session = storage.get_session(session_id)
    assert session is None


@respx.mock
def test_list_sessions(storage):
    respx.get("http://test-server/sessions").mock(
        return_value=httpx.Response(200, json=[])
    )

    sessions = storage.list_sessions(limit=10, status=SessionStatus.COMPLETED)
    assert sessions == []

    last_request = respx.calls.last.request
    assert last_request.url.params["limit"] == "10"
    assert last_request.url.params["status"] == "completed"


@respx.mock
def test_has_chunk(storage):
    respx.post("http://test-server/chunks/exists").mock(
        return_value=httpx.Response(200, json={"exists": True})
    )

    exists = storage.has_chunk("some content", SemanticLabel.GOTCHA)
    assert exists is True

    last_request = respx.calls.last.request
    import json
    body = json.loads(last_request.content)
    assert body["content"] == "some content"
    assert body["label"] == "gotcha"


@respx.mock
def test_get_background_sync_status(storage):
    now = datetime.now(UTC)
    status_data = {
        "is_running": True,
        "last_run_at": now.isoformat(),
        "last_status": "success",
        "pid": 12345,
    }

    respx.get("http://test-server/background-sync/status").mock(
        return_value=httpx.Response(200, json=status_data)
    )

    status = storage.get_background_sync_status()
    assert status.is_running is True
    assert status.pid == 12345


@respx.mock
def test_http_error_raises_exception(storage):
    respx.get("http://test-server/stats").mock(
        return_value=httpx.Response(500)
    )

    # RemoteStorage wraps unexpected errors in SharedBackendUnavailableError
    # But since 500 is an HTTPStatusError (not TransportError), it might bubble up?
    # Let's check RemoteStorage._execute.
    # It catches (httpx.TransportError, httpx.TimeoutException, ...).
    # httpx.HTTPStatusError is NOT in that list.
    # So it should raise httpx.HTTPStatusError.

    with pytest.raises(httpx.HTTPStatusError):
        storage.get_stats()


@respx.mock
def test_transport_error_retries_and_raises_unavailable(http_config):
    # Configure retry attempts
    http_config.retry_attempts = 2
    storage = RemoteStorage(http_config)

    route = respx.get("http://test-server/stats").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(SharedBackendUnavailableError) as exc:
        storage.get_stats()

    assert "failed after 2 attempts" in str(exc.value)
    assert route.call_count == 2
