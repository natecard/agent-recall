from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from agent_recall.storage.models import SharedStorageConfig
from agent_recall.storage.sqlite import SQLiteStorage


def resolve_shared_db_path(base_url: str | None) -> Path:
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
        raise NotImplementedError(
            "HTTP shared storage service mode is not implemented yet. "
            "Use a shared filesystem URL (`file://...` or `sqlite://...`) for now."
        )

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
            "Use `file://`, `sqlite://`, or a filesystem path."
        )

    db_path = Path(path_value).expanduser()
    if not db_path.is_absolute():
        db_path = db_path.resolve()
    if raw.endswith(("/", "\\")) or not db_path.suffix:
        db_path = db_path / "state.db"
    return db_path


class RemoteStorage(SQLiteStorage):
    """
    Storage implementation for shared single-tenant backends.

    Tracer-bullet implementation:
    - Supports `file://` and `sqlite://` URLs that point at a shared filesystem.
    - Reuses SQLiteStorage so all core operations remain behavior-compatible.
    - Reserves HTTP(S) service-backed mode for the follow-up shared backend slice.
    """

    def __init__(self, config: SharedStorageConfig) -> None:
        self.config = config
        super().__init__(resolve_shared_db_path(config.base_url))
