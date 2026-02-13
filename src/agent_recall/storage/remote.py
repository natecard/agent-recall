from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID

import httpx

from agent_recall.storage.base import (
    SharedBackendUnavailableError,
    Storage,
    validate_shared_namespace,
)
from agent_recall.storage.models import (
    BackgroundSyncStatus,
    Chunk,
    LogEntry,
    SemanticLabel,
    Session,
    SessionCheckpoint,
    SessionStatus,
    SharedStorageConfig,
)
from agent_recall.storage.sqlite import SQLiteStorage


def resolve_shared_db_path(base_url: str | None) -> Path | str:
    if not base_url:
        raise ValueError("Shared storage backend requires `storage.shared.base_url` to be set.")
    raw = base_url.strip()
    if not raw:
        raise ValueError("Shared storage backend requires a non-empty `storage.shared.base_url`.")

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return raw

    path_value: str
    if parsed.scheme in {"file", "sqlite"}:
        path_value = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path_value = f"//{parsed.netloc}{path_value}"
    elif parsed.scheme == "" or (len(parsed.scheme) == 1 and raw[1:2] == ":"):
        path_value = raw
    else:
        raise ValueError(
            "Unsupported shared storage URL scheme. "
            "Use `http(s)://`, `file://`, `sqlite://`, or a filesystem path."
        )

    db_path = Path(path_value).expanduser()
    if not db_path.is_absolute():
        db_path = db_path.resolve()
    if raw.endswith(("/", "\\")) or not db_path.suffix:
        db_path = db_path / "state.db"
    return db_path


class _HTTPClient(Storage):
    """Internal client for the remote HTTP storage backend."""

    def __init__(self, config: SharedStorageConfig) -> None:
        if not config.base_url:
            raise ValueError("Shared storage backend requires `storage.shared.base_url`.")

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(config.api_key_env)
        if config.require_api_key and not api_key:
            raise ValueError(
                f"Shared storage backend requires {config.api_key_env} to be set when "
                "require_api_key is enabled."
            )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Add isolation headers
        headers["X-Tenant-ID"] = config.tenant_id
        headers["X-Project-ID"] = config.project_id

        self._client = httpx.Client(
            base_url=config.base_url, headers=headers, timeout=config.timeout_seconds
        )

    def create_session(self, session: Session) -> None:
        response = self._client.post("/sessions", content=session.model_dump_json())
        response.raise_for_status()

    def get_session(self, session_id: UUID) -> Session | None:
        response = self._client.get(f"/sessions/{session_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Session.model_validate(response.json())

    def get_active_session(self) -> Session | None:
        response = self._client.get("/sessions/active")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Session.model_validate(response.json())

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        params = {"limit": limit}
        if status:
            params["status"] = status.value
        response = self._client.get("/sessions", params=params)
        response.raise_for_status()
        return [Session.model_validate(s) for s in response.json()]

    def update_session(self, session: Session) -> None:
        response = self._client.put(f"/sessions/{session.id}", content=session.model_dump_json())
        response.raise_for_status()

    def append_entry(self, entry: LogEntry) -> None:
        response = self._client.post("/entries", content=entry.model_dump_json())
        response.raise_for_status()

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        response = self._client.get(f"/sessions/{session_id}/entries")
        response.raise_for_status()
        return [LogEntry.model_validate(e) for e in response.json()]

    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        params = {
            "labels": [label.value for label in labels],
            "limit": limit,
        }
        # httpx handles list params by repeating keys: labels=...&labels=...
        response = self._client.get("/entries", params=params)
        response.raise_for_status()
        return [LogEntry.model_validate(e) for e in response.json()]

    def count_log_entries(self) -> int:
        response = self._client.get("/entries/count")
        response.raise_for_status()
        return response.json()["count"]

    def store_chunk(self, chunk: Chunk) -> None:
        response = self._client.post("/chunks", content=chunk.model_dump_json())
        response.raise_for_status()

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        response = self._client.post(
            "/chunks/exists",
            json={"content": content, "label": label.value},
        )
        response.raise_for_status()
        return response.json()["exists"]

    def count_chunks(self) -> int:
        response = self._client.get("/chunks/count")
        response.raise_for_status()
        return response.json()["count"]

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        response = self._client.get("/chunks/search", params={"q": query, "top_k": top_k})
        response.raise_for_status()
        return [Chunk.model_validate(c) for c in response.json()]

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        response = self._client.get("/chunks/embeddings")
        response.raise_for_status()
        return [Chunk.model_validate(c) for c in response.json()]

    def is_session_processed(self, source_session_id: str) -> bool:
        # We might want base64 or similar if session IDs contain slashes
        # But for now let's assume valid path chars or rely on query param if unsure
        # Query param is safer for arbitrary strings
        response = self._client.get(
            "/processed_sessions/check", params={"source_session_id": source_session_id}
        )
        response.raise_for_status()
        return response.json()["processed"]

    def mark_session_processed(self, source_session_id: str) -> None:
        response = self._client.put(
            "/processed_sessions", json={"source_session_id": source_session_id}
        )
        response.raise_for_status()

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        params = {}
        if source:
            params["source"] = source
        if source_session_id:
            params["source_session_id"] = source_session_id
        response = self._client.delete("/processed_sessions", params=params)
        response.raise_for_status()
        return response.json()["count"]

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        response = self._client.get("/checkpoints", params={"source_session_id": source_session_id})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return SessionCheckpoint.model_validate(response.json())

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        response = self._client.put("/checkpoints", content=checkpoint.model_dump_json())
        response.raise_for_status()

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        params = {}
        if source:
            params["source"] = source
        if source_session_id:
            params["source_session_id"] = source_session_id
        response = self._client.delete("/checkpoints", params=params)
        response.raise_for_status()
        return response.json()["count"]

    def get_stats(self) -> dict[str, int]:
        response = self._client.get("/stats")
        response.raise_for_status()
        return response.json()

    def get_last_processed_at(self) -> datetime | None:
        response = self._client.get("/stats/last_processed")
        response.raise_for_status()
        data = response.json()
        if data["last_processed_at"] is None:
            return None
        return datetime.fromisoformat(data["last_processed_at"])

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        response = self._client.get("/recent-sources", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        response = self._client.get("/background-sync/status")
        response.raise_for_status()
        return BackgroundSyncStatus.model_validate(response.json())

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        response = self._client.put("/background-sync/status", content=status.model_dump_json())
        response.raise_for_status()

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        response = self._client.post("/background-sync/start", json={"pid": pid})
        response.raise_for_status()
        return BackgroundSyncStatus.model_validate(response.json())

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        payload = {
            "sessions_processed": sessions_processed,
            "learnings_extracted": learnings_extracted,
            "error_message": error_message,
        }
        response = self._client.post("/background-sync/complete", json=payload)
        response.raise_for_status()
        return BackgroundSyncStatus.model_validate(response.json())


class RemoteStorage(Storage):
    """
    Storage implementation for shared single-tenant backends.

    This class acts as a client for a remote storage backend. It supports:
    - Shared filesystem backends via `file://` or `sqlite://` URLs.
    - Remote HTTP/HTTPS backends.
    """

    _delegate: Storage
    _local: Storage | None

    def __init__(self, config: SharedStorageConfig, local_db_path: Path | None = None) -> None:
        # Validate namespace before initializing
        validate_shared_namespace(config.tenant_id, config.project_id)

        self.config = config
        self._local = SQLiteStorage(local_db_path) if local_db_path else None

        resolved = resolve_shared_db_path(config.base_url)

        if isinstance(resolved, Path):
            # Filesystem shared backend - enable strict namespace validation
            self._delegate = SQLiteStorage(
                resolved,
                tenant_id=config.tenant_id,
                project_id=config.project_id,
                strict_namespace_validation=True,
            )
        else:
            # HTTP backend - client handles namespace via headers
            self._delegate = _HTTPClient(config)

    def _execute(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        attempts = max(1, self.config.retry_attempts or 1)
        last_error: Exception | None = None

        for i in range(attempts):
            try:
                method = getattr(self._delegate, method_name)
                return method(*args, **kwargs)
            except (
                httpx.TransportError,
                httpx.TimeoutException,
                sqlite3.OperationalError,
                OSError,
                SharedBackendUnavailableError,
            ) as e:
                last_error = e
                if i < attempts - 1:
                    time.sleep(0.5 * (2**i))
                    continue

        if self._local:
            try:
                method = getattr(self._local, method_name)
                return method(*args, **kwargs)
            except Exception:
                # If local also fails, ignore local error and raise the shared one
                pass

        if last_error:
            raise SharedBackendUnavailableError(
                f"Shared storage operation '{method_name}' failed after {attempts} attempts"
            ) from last_error

        raise SharedBackendUnavailableError(
            f"Shared storage operation '{method_name}' failed with unknown error"
        )

    def create_session(self, session: Session) -> None:
        return self._execute("create_session", session)

    def get_session(self, session_id: UUID) -> Session | None:
        return self._execute("get_session", session_id)

    def get_active_session(self) -> Session | None:
        return self._execute("get_active_session")

    def list_sessions(self, limit: int = 50, status: SessionStatus | None = None) -> list[Session]:
        return self._execute("list_sessions", limit=limit, status=status)

    def update_session(self, session: Session) -> None:
        return self._execute("update_session", session)

    def append_entry(self, entry: LogEntry) -> None:
        return self._execute("append_entry", entry)

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        return self._execute("get_entries", session_id)

    def get_entries_by_label(self, labels: list[SemanticLabel], limit: int = 100) -> list[LogEntry]:
        return self._execute("get_entries_by_label", labels=labels, limit=limit)

    def count_log_entries(self) -> int:
        return self._execute("count_log_entries")

    def store_chunk(self, chunk: Chunk) -> None:
        return self._execute("store_chunk", chunk)

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        return self._execute("has_chunk", content, label)

    def count_chunks(self) -> int:
        return self._execute("count_chunks")

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        return self._execute("search_chunks_fts", query, top_k=top_k)

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        return self._execute("list_chunks_with_embeddings")

    def is_session_processed(self, source_session_id: str) -> bool:
        return self._execute("is_session_processed", source_session_id)

    def mark_session_processed(self, source_session_id: str) -> None:
        return self._execute("mark_session_processed", source_session_id)

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        return self._execute(
            "clear_processed_sessions", source=source, source_session_id=source_session_id
        )

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        return self._execute("get_session_checkpoint", source_session_id)

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        return self._execute("save_session_checkpoint", checkpoint)

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        return self._execute(
            "clear_session_checkpoints", source=source, source_session_id=source_session_id
        )

    def get_stats(self) -> dict[str, int]:
        return self._execute("get_stats")

    def get_last_processed_at(self) -> datetime | None:
        return self._execute("get_last_processed_at")

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._execute("list_recent_source_sessions", limit=limit)

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        return self._execute("get_background_sync_status")

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        return self._execute("save_background_sync_status", status)

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        return self._execute("start_background_sync", pid)

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        return self._execute(
            "complete_background_sync",
            sessions_processed,
            learnings_extracted,
            error_message,
        )
