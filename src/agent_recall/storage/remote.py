from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID

import httpx

from agent_recall.storage.base import Storage
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
        raise ValueError(
            "Shared storage backend requires `storage.shared.base_url` to be set."
        )
    raw = base_url.strip()
    if not raw:
        raise ValueError(
            "Shared storage backend requires a non-empty `storage.shared.base_url`."
        )

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
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

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
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def list_sessions(
        self, limit: int = 50, status: SessionStatus | None = None
    ) -> list[Session]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def update_session(self, session: Session) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def append_entry(self, entry: LogEntry) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_entries_by_label(
        self, labels: list[SemanticLabel], limit: int = 100
    ) -> list[LogEntry]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def count_log_entries(self) -> int:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def store_chunk(self, chunk: Chunk) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def count_chunks(self) -> int:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def is_session_processed(self, source_session_id: str) -> bool:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def mark_session_processed(self, source_session_id: str) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_stats(self) -> dict[str, int]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_last_processed_at(self) -> datetime | None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        raise NotImplementedError("HTTP backend does not support this operation yet.")

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        raise NotImplementedError("HTTP backend does not support this operation yet.")


class RemoteStorage(Storage):
    """
    Storage implementation for shared single-tenant backends.

    This class acts as a client for a remote storage backend. It supports:
    - Shared filesystem backends via `file://` or `sqlite://` URLs.
    - Remote HTTP/HTTPS backends.
    """

    _delegate: Storage

    def __init__(self, config: SharedStorageConfig) -> None:
        self.config = config
        resolved = resolve_shared_db_path(config.base_url)

        if isinstance(resolved, Path):
            self._delegate = SQLiteStorage(resolved)
        else:
            self._delegate = _HTTPClient(config)

    def create_session(self, session: Session) -> None:
        return self._delegate.create_session(session)

    def get_session(self, session_id: UUID) -> Session | None:
        return self._delegate.get_session(session_id)

    def get_active_session(self) -> Session | None:
        return self._delegate.get_active_session()

    def list_sessions(
        self, limit: int = 50, status: SessionStatus | None = None
    ) -> list[Session]:
        return self._delegate.list_sessions(limit=limit, status=status)

    def update_session(self, session: Session) -> None:
        return self._delegate.update_session(session)

    def append_entry(self, entry: LogEntry) -> None:
        return self._delegate.append_entry(entry)

    def get_entries(self, session_id: UUID) -> list[LogEntry]:
        return self._delegate.get_entries(session_id)

    def get_entries_by_label(
        self, labels: list[SemanticLabel], limit: int = 100
    ) -> list[LogEntry]:
        return self._delegate.get_entries_by_label(labels=labels, limit=limit)

    def count_log_entries(self) -> int:
        return self._delegate.count_log_entries()

    def store_chunk(self, chunk: Chunk) -> None:
        return self._delegate.store_chunk(chunk)

    def has_chunk(self, content: str, label: SemanticLabel) -> bool:
        return self._delegate.has_chunk(content, label)

    def count_chunks(self) -> int:
        return self._delegate.count_chunks()

    def search_chunks_fts(self, query: str, top_k: int = 5) -> list[Chunk]:
        return self._delegate.search_chunks_fts(query, top_k=top_k)

    def list_chunks_with_embeddings(self) -> list[Chunk]:
        return self._delegate.list_chunks_with_embeddings()

    def is_session_processed(self, source_session_id: str) -> bool:
        return self._delegate.is_session_processed(source_session_id)

    def mark_session_processed(self, source_session_id: str) -> None:
        return self._delegate.mark_session_processed(source_session_id)

    def clear_processed_sessions(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        return self._delegate.clear_processed_sessions(
            source=source, source_session_id=source_session_id
        )

    def get_session_checkpoint(self, source_session_id: str) -> SessionCheckpoint | None:
        return self._delegate.get_session_checkpoint(source_session_id)

    def save_session_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        return self._delegate.save_session_checkpoint(checkpoint)

    def clear_session_checkpoints(
        self,
        source: str | None = None,
        source_session_id: str | None = None,
    ) -> int:
        return self._delegate.clear_session_checkpoints(
            source=source, source_session_id=source_session_id
        )

    def get_stats(self) -> dict[str, int]:
        return self._delegate.get_stats()

    def get_last_processed_at(self) -> datetime | None:
        return self._delegate.get_last_processed_at()

    def list_recent_source_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._delegate.list_recent_source_sessions(limit=limit)

    def get_background_sync_status(self) -> BackgroundSyncStatus:
        return self._delegate.get_background_sync_status()

    def save_background_sync_status(self, status: BackgroundSyncStatus) -> None:
        return self._delegate.save_background_sync_status(status)

    def start_background_sync(self, pid: int) -> BackgroundSyncStatus:
        return self._delegate.start_background_sync(pid)

    def complete_background_sync(
        self,
        sessions_processed: int,
        learnings_extracted: int,
        error_message: str | None = None,
    ) -> BackgroundSyncStatus:
        return self._delegate.complete_background_sync(
            sessions_processed, learnings_extracted, error_message
        )
