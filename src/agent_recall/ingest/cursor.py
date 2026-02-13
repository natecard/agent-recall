from __future__ import annotations

import json
import os
import platform
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester


@dataclass(frozen=True)
class _CursorSessionRef:
    """Internal pointer to a specific Cursor conversation session."""

    workspace_db_path: Path
    composer_id: str | None = None
    created_at: datetime | None = None
    last_updated_at: datetime | None = None


class CursorIngester(SessionIngester):
    """Ingest Cursor conversation history from workspace SQLite storage."""

    def __init__(
        self,
        project_path: Path | None = None,
        cursor_db_path: Path | None = None,
        workspace_storage_dir: Path | None = None,
        global_storage_db_path: Path | None = None,
        include_all_workspaces: bool = False,
    ):
        self.project_path = (project_path or Path.cwd()).resolve()
        self.cursor_db_path = cursor_db_path.resolve() if cursor_db_path else None
        self.storage_dir = (
            workspace_storage_dir.resolve() if workspace_storage_dir else self._get_storage_dir()
        )
        self._global_storage_override = global_storage_db_path is not None
        self.global_storage_db_path = (
            global_storage_db_path.resolve()
            if global_storage_db_path
            else self._get_global_storage_db_path()
        )
        self.include_all_workspaces = include_all_workspaces
        self._session_refs: dict[Path, _CursorSessionRef] = {}
        self._global_db_paths_cache: list[Path] | None = None
        self._composer_db_cache: dict[str, Path] = {}

    @property
    def source_name(self) -> str:
        return "cursor"

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def _get_storage_dir(self) -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library/Application Support/Cursor/User/workspaceStorage"
        if system == "Linux":
            return Path.home() / ".config/Cursor/User/workspaceStorage"
        if system == "Windows":
            appdata = os.environ.get("APPDATA", "")
            return Path(appdata) / "Cursor/User/workspaceStorage"
        raise RuntimeError(f"Unsupported platform: {system}")

    def _get_global_storage_db_path(self) -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library/Application Support/Cursor/User/globalStorage/state.vscdb"
        if system == "Linux":
            return Path.home() / ".config/Cursor/User/globalStorage/state.vscdb"
        if system == "Windows":
            appdata = os.environ.get("APPDATA", "")
            return Path(appdata) / "Cursor/User/globalStorage/state.vscdb"
        raise RuntimeError(f"Unsupported platform: {system}")

    def _find_workspace_dbs(self) -> list[Path]:
        if self.cursor_db_path is not None:
            return [self.cursor_db_path] if self.cursor_db_path.exists() else []

        if not self.storage_dir.exists():
            return []

        found: list[Path] = []
        workspace_dirs: set[Path] = set()

        for metadata_path in self.storage_dir.rglob("workspace.json"):
            workspace_dirs.add(metadata_path.parent)

        if self.include_all_workspaces:
            for db_path in self.storage_dir.rglob("state.vscdb"):
                workspace_dirs.add(db_path.parent)

        for workspace_dir in sorted(workspace_dirs):
            db_path = workspace_dir / "state.vscdb"
            if not db_path.exists():
                continue

            if self.include_all_workspaces:
                found.append(db_path)
                continue

            metadata_path = workspace_dir / "workspace.json"
            if metadata_path.exists() and self._workspace_matches_project(metadata_path):
                found.append(db_path)
                break

        return found

    def _workspace_matches_project(self, metadata_path: Path) -> bool:
        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        if not isinstance(metadata, dict):
            return False

        folder_uri = metadata.get("folder")
        folder_path = self._decode_cursor_uri(folder_uri)
        if folder_path and folder_path == self.project_path:
            return True

        workspace_uri = metadata.get("workspace")
        workspace_path = self._decode_cursor_uri(workspace_uri)
        if not workspace_path:
            return False

        if workspace_path == self.project_path:
            return True

        if workspace_path.suffix != ".code-workspace" or not workspace_path.exists():
            return False

        return self._workspace_file_references_project(workspace_path)

    def _workspace_file_references_project(self, workspace_path: Path) -> bool:
        try:
            payload = json.loads(workspace_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        if not isinstance(payload, dict):
            return False

        folders = payload.get("folders", [])
        if not isinstance(folders, list):
            return False

        for folder in folders:
            if not isinstance(folder, dict):
                continue

            uri_candidate = folder.get("uri")
            if isinstance(uri_candidate, str):
                decoded = self._decode_cursor_uri(uri_candidate)
                if decoded and decoded == self.project_path:
                    return True

            path_candidate = folder.get("path")
            if isinstance(path_candidate, str):
                candidate = Path(path_candidate)
                if not candidate.is_absolute():
                    candidate = workspace_path.parent / candidate
                try:
                    if candidate.resolve() == self.project_path:
                        return True
                except OSError:
                    continue

        return False

    def _decode_cursor_uri(self, raw_value: Any) -> Path | None:
        if not isinstance(raw_value, str) or not raw_value:
            return None

        parsed = urlparse(raw_value)
        if parsed.scheme == "file":
            decoded_path = unquote(parsed.path)
            if len(decoded_path) > 2 and decoded_path[0] == "/" and decoded_path[2] == ":":
                decoded_path = decoded_path[1:]
            try:
                return Path(decoded_path).resolve()
            except OSError:
                return None

        if parsed.scheme:
            return None

        try:
            return Path(unquote(raw_value)).resolve()
        except OSError:
            return None

    @staticmethod
    def _try_parse_json(value: str | bytes) -> dict[str, Any] | list[Any] | None:
        decoded: str
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        else:
            decoded = value

        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    def _iter_global_storage_dbs(self) -> list[Path]:
        if self._global_db_paths_cache is not None:
            return list(self._global_db_paths_cache)

        if self._global_storage_override:
            paths = [self.global_storage_db_path] if self.global_storage_db_path.exists() else []
            self._global_db_paths_cache = paths
            return list(paths)

        candidates: list[Path] = []
        if self.global_storage_db_path.exists():
            candidates.append(self.global_storage_db_path)

        profiles_root = self.global_storage_db_path.parent.parent / "profiles"
        if profiles_root.exists():
            for profile_db in sorted(profiles_root.glob("*/globalStorage/state.vscdb")):
                if profile_db.exists():
                    candidates.append(profile_db)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            deduped.append(resolved)
            seen.add(resolved)

        self._global_db_paths_cache = deduped
        return list(deduped)

    @staticmethod
    def _session_handle_for_ref(ref: _CursorSessionRef) -> Path:
        if not ref.composer_id:
            return ref.workspace_db_path
        return ref.workspace_db_path.parent / (
            f"{ref.workspace_db_path.stem}.composer.{ref.composer_id}"
        )

    def _extract_workspace_composer_refs(self, db_path: Path) -> list[_CursorSessionRef]:
        refs: list[_CursorSessionRef] = []
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
            ).fetchone()
        except sqlite3.Error:
            return refs
        finally:
            conn.close()

        if not row:
            return refs

        parsed = self._try_parse_json(row[0])
        if not isinstance(parsed, dict):
            return refs

        all_composers = parsed.get("allComposers")
        if not isinstance(all_composers, list):
            return refs

        for composer in all_composers:
            if not isinstance(composer, dict):
                continue
            composer_id = composer.get("composerId")
            if not isinstance(composer_id, str) or not composer_id:
                continue

            created_at = self._parse_timestamp(composer.get("createdAt"))
            last_updated_at = self._parse_timestamp(composer.get("lastUpdatedAt")) or created_at
            refs.append(
                _CursorSessionRef(
                    workspace_db_path=db_path,
                    composer_id=composer_id,
                    created_at=created_at,
                    last_updated_at=last_updated_at,
                )
            )

        refs.sort(
            key=lambda ref: (
                (
                    ref.created_at.timestamp()
                    if ref.created_at
                    else (ref.last_updated_at.timestamp() if ref.last_updated_at else 0.0)
                ),
                ref.composer_id or "",
            )
        )
        return refs

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        db_paths = self._find_workspace_dbs()
        if not db_paths:
            return []

        self._session_refs.clear()
        session_handles: list[Path] = []
        normalized_since = self._normalize_dt(since) if since else None

        for db_path in db_paths:
            composer_refs = self._extract_workspace_composer_refs(db_path)
            if composer_refs:
                for ref in composer_refs:
                    if normalized_since:
                        session_time = ref.last_updated_at or ref.created_at
                        if session_time and session_time < normalized_since:
                            continue
                        if session_time is None:
                            mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC)
                            if mtime < normalized_since:
                                continue

                    handle = self._session_handle_for_ref(ref)
                    self._session_refs[handle] = ref
                    session_handles.append(handle)
                continue

            if normalized_since:
                mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC)
                if mtime < normalized_since:
                    continue

            fallback_ref = _CursorSessionRef(workspace_db_path=db_path)
            self._session_refs[db_path] = fallback_ref
            session_handles.append(db_path)

        return session_handles

    def _resolve_session_ref(self, path: Path) -> _CursorSessionRef:
        ref = self._session_refs.get(path)
        if ref:
            return ref
        return _CursorSessionRef(workspace_db_path=path)

    def get_session_id(self, path: Path) -> str:
        ref = self._resolve_session_ref(path)
        workspace_db = ref.workspace_db_path

        if ref.composer_id:
            timestamp_source = ref.last_updated_at or ref.created_at
            timestamp = int(timestamp_source.timestamp() * 1000) if timestamp_source else 0
            return f"cursor-{workspace_db.parent.name}-{ref.composer_id}-{timestamp}"

        mtime = int(workspace_db.stat().st_mtime)
        return f"cursor-{workspace_db.parent.name}-{mtime}"

    def parse_session(self, path: Path) -> RawSession:
        ref = self._resolve_session_ref(path)
        if ref.composer_id:
            return self._parse_composer_session(path, ref)
        return self._parse_workspace_session(path, ref.workspace_db_path)

    def _parse_workspace_session(self, session_handle: Path, db_path: Path) -> RawSession:
        records = self._extract_conversations(db_path)

        messages: list[RawMessage] = []
        earliest_time: datetime | None = None
        latest_time: datetime | None = None

        for record in records:
            table = str(record.get("table", ""))
            key = str(record.get("key", ""))
            data = record.get("data")

            if table == "ItemTable":
                parsed = self._parse_itemtable_payload(key, data)
            else:
                parsed = self._parse_generic_payload(data)

            for message in parsed:
                if len(message.content.strip()) < 3:
                    continue
                messages.append(message)
                if message.timestamp:
                    earliest_time = (
                        message.timestamp
                        if earliest_time is None
                        else min(earliest_time, message.timestamp)
                    )
                    latest_time = (
                        message.timestamp
                        if latest_time is None
                        else max(latest_time, message.timestamp)
                    )

        sorted_messages = sorted(
            enumerate(messages),
            key=lambda item: (
                item[1].timestamp.timestamp() if item[1].timestamp else 0.0,
                item[0],
            ),
        )

        deduped: list[RawMessage] = []
        for _, message in sorted_messages:
            if not deduped:
                deduped.append(message)
                continue
            previous = deduped[-1]
            if (message.role, message.content) != (previous.role, previous.content):
                deduped.append(message)

        inferred_project_path = self._project_path_from_workspace_metadata(db_path)
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(session_handle),
            title=self._infer_title_from_messages(deduped),
            project_path=inferred_project_path or self.project_path,
            started_at=earliest_time or datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC),
            ended_at=latest_time,
            messages=deduped,
        )

    def _parse_composer_session(self, session_handle: Path, ref: _CursorSessionRef) -> RawSession:
        composer_id = ref.composer_id
        if not composer_id:
            return self._parse_workspace_session(session_handle, ref.workspace_db_path)

        inferred_project_path = self._project_path_from_workspace_metadata(ref.workspace_db_path)
        messages, composer_title = self._parse_composer_bubbles(composer_id)
        if not messages:
            return self._parse_workspace_session(session_handle, ref.workspace_db_path)

        sorted_messages = sorted(
            enumerate(messages),
            key=lambda item: (
                item[1].timestamp.timestamp() if item[1].timestamp else 0.0,
                item[0],
            ),
        )

        deduped: list[RawMessage] = []
        earliest_time: datetime | None = ref.created_at
        latest_time: datetime | None = ref.last_updated_at

        for _, message in sorted_messages:
            if message.timestamp:
                earliest_time = (
                    message.timestamp
                    if earliest_time is None
                    else min(earliest_time, message.timestamp)
                )
                latest_time = (
                    message.timestamp
                    if latest_time is None
                    else max(latest_time, message.timestamp)
                )

            if not deduped:
                deduped.append(message)
                continue

            previous = deduped[-1]
            if (message.role, message.content) != (previous.role, previous.content):
                deduped.append(message)

        started_at = earliest_time or datetime.fromtimestamp(
            ref.workspace_db_path.stat().st_mtime,
            tz=UTC,
        )
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(session_handle),
            title=composer_title or self._infer_title_from_messages(deduped),
            project_path=inferred_project_path or self.project_path,
            started_at=started_at,
            ended_at=latest_time,
            messages=deduped,
        )

    def _parse_composer_bubbles(self, composer_id: str) -> tuple[list[RawMessage], str | None]:
        db_path = self._find_global_db_for_composer(composer_id)
        if db_path is None:
            return [], None

        conn = sqlite3.connect(str(db_path))
        try:
            composer_payload, kv_table = self._load_composer_payload(conn, composer_id)
            if not composer_payload or not kv_table:
                return [], None

            composer_title: str | None = None
            raw_title = composer_payload.get("name")
            if isinstance(raw_title, str):
                cleaned = raw_title.strip()
                if cleaned:
                    composer_title = cleaned

            headers_raw = composer_payload.get("fullConversationHeadersOnly")
            header_items: list[tuple[str, str]] = []
            if isinstance(headers_raw, list):
                for item in headers_raw:
                    if not isinstance(item, dict):
                        continue
                    bubble_id = item.get("bubbleId")
                    if not isinstance(bubble_id, str) or not bubble_id:
                        continue
                    bubble_type = item.get("type", 2)
                    role = "user" if bubble_type in {1, "1"} else "assistant"
                    header_items.append((bubble_id, role))

            bubble_payloads: dict[str, dict[str, Any]] = {}
            if header_items:
                bubble_keys = [
                    f"bubbleId:{composer_id}:{bubble_id}" for bubble_id, _ in header_items
                ]
                bubble_payloads = self._load_rows_by_keys(conn, kv_table, bubble_keys)
            else:
                prefix_rows = self._load_rows_by_prefix(conn, kv_table, f"bubbleId:{composer_id}:%")
                for key, value in prefix_rows.items():
                    bubble_id = key.rsplit(":", 1)[-1]
                    bubble_payloads[bubble_id] = value

                def _bubble_sort_key(item: tuple[str, dict[str, Any]]) -> float:
                    timestamp = self._extract_timestamp(item[1])
                    return timestamp.timestamp() if timestamp else 0.0

                ordered_bubbles = sorted(
                    bubble_payloads.items(),
                    key=_bubble_sort_key,
                )
                header_items = [(bubble_id, "assistant") for bubble_id, _ in ordered_bubbles]

            messages: list[RawMessage] = []
            for bubble_id, default_role in header_items:
                payload = bubble_payloads.get(bubble_id)
                if not payload:
                    continue

                role = default_role
                bubble_type = payload.get("type")
                if bubble_type in {1, "1"}:
                    role = "user"
                elif bubble_type in {2, "2"}:
                    role = "assistant"

                content = self._extract_content(
                    payload,
                    keys=[
                        "text",
                        "rawText",
                        "content",
                        "message",
                        "body",
                        "markdown",
                    ],
                ).strip()
                if not content:
                    content = self._extract_rich_text(payload.get("richText"))

                timestamp = self._extract_timestamp(payload)
                tool_calls = self._extract_tool_calls(payload)
                message = self._build_message(role, content, timestamp, tool_calls)
                if message:
                    messages.append(message)

            return messages, composer_title
        finally:
            conn.close()

    def _find_global_db_for_composer(self, composer_id: str) -> Path | None:
        cached = self._composer_db_cache.get(composer_id)
        if cached and cached.exists():
            return cached

        lookup_key = f"composerData:{composer_id}"
        for db_path in self._iter_global_storage_dbs():
            conn = sqlite3.connect(str(db_path))
            try:
                for table in ("cursorDiskKV", "ItemTable"):
                    try:
                        row = conn.execute(
                            f"SELECT 1 FROM {table} WHERE key = ? LIMIT 1",
                            (lookup_key,),
                        ).fetchone()
                    except sqlite3.Error:
                        continue
                    if row:
                        self._composer_db_cache[composer_id] = db_path
                        return db_path
            finally:
                conn.close()

        return None

    def _load_composer_payload(
        self,
        conn: sqlite3.Connection,
        composer_id: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        lookup_key = f"composerData:{composer_id}"
        for table in ("cursorDiskKV", "ItemTable"):
            try:
                row = conn.execute(
                    f"SELECT value FROM {table} WHERE key = ?",
                    (lookup_key,),
                ).fetchone()
            except sqlite3.Error:
                continue
            if not row:
                continue
            parsed = self._try_parse_json(row[0])
            if isinstance(parsed, dict):
                return parsed, table
        return None, None

    def _load_rows_by_prefix(
        self,
        conn: sqlite3.Connection,
        table: str,
        key_pattern: str,
    ) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        try:
            cursor = conn.execute(
                f"SELECT key, value FROM {table} WHERE key LIKE ?",
                (key_pattern,),
            )
        except sqlite3.Error:
            return rows

        for key_value, raw_value in cursor.fetchall():
            key = str(key_value)
            parsed = self._try_parse_json(raw_value)
            if isinstance(parsed, dict):
                rows[key] = parsed
        return rows

    def _load_rows_by_keys(
        self,
        conn: sqlite3.Connection,
        table: str,
        keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        if not keys:
            return rows

        # SQLite supports up to 999 bind parameters by default; keep margin for safety.
        chunk_size = 400
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT key, value FROM {table} WHERE key IN ({placeholders})"
            try:
                cursor = conn.execute(query, tuple(chunk))
            except sqlite3.Error:
                continue
            for key_value, raw_value in cursor.fetchall():
                key = str(key_value)
                parsed = self._try_parse_json(raw_value)
                if isinstance(parsed, dict):
                    rows[key] = parsed

        keyed_by_bubble_id: dict[str, dict[str, Any]] = {}
        for key, payload in rows.items():
            bubble_id = key.rsplit(":", 1)[-1]
            keyed_by_bubble_id[bubble_id] = payload
        return keyed_by_bubble_id

    def _project_path_from_workspace_metadata(self, db_path: Path) -> Path | None:
        metadata_path = db_path.parent / "workspace.json"
        if not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(metadata, dict):
            return None

        folder_uri = metadata.get("folder")
        folder_path = self._decode_cursor_uri(folder_uri)
        if folder_path:
            return folder_path

        workspace_uri = metadata.get("workspace")
        workspace_path = self._decode_cursor_uri(workspace_uri)
        if not workspace_path:
            return None
        if workspace_path.is_dir():
            return workspace_path
        if workspace_path.suffix != ".code-workspace" or not workspace_path.exists():
            return workspace_path

        try:
            payload = json.loads(workspace_path.read_text())
        except (json.JSONDecodeError, OSError):
            return workspace_path
        if not isinstance(payload, dict):
            return workspace_path

        folders = payload.get("folders", [])
        if not isinstance(folders, list):
            return workspace_path

        for folder in folders:
            if not isinstance(folder, dict):
                continue
            uri_candidate = folder.get("uri")
            if isinstance(uri_candidate, str):
                decoded = self._decode_cursor_uri(uri_candidate)
                if decoded:
                    return decoded
            path_candidate = folder.get("path")
            if isinstance(path_candidate, str):
                candidate = Path(path_candidate)
                if not candidate.is_absolute():
                    candidate = workspace_path.parent / candidate
                try:
                    return candidate.resolve()
                except OSError:
                    continue

        return workspace_path

    def _extract_conversations(self, db_path: Path) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conversations: list[dict[str, Any]] = []

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [str(row["name"]) for row in cursor.fetchall()]

            if "ItemTable" in tables:
                conversations.extend(self._extract_itemtable_rows(cursor))

            ai_keywords = ["chat", "conversation", "composer", "ai", "message", "copilot"]
            for table in tables:
                if table == "ItemTable":
                    continue
                if not any(keyword in table.lower() for keyword in ai_keywords):
                    continue
                conversations.extend(self._extract_table_rows(cursor, table))
        finally:
            conn.close()

        return conversations

    def _extract_itemtable_rows(self, cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            cursor.execute("SELECT key, value FROM ItemTable")
        except sqlite3.Error:
            return rows

        for row in cursor.fetchall():
            key = str(row["key"]) if row["key"] is not None else ""
            value = row["value"]
            if not isinstance(value, (str, bytes)):
                continue

            parsed = self._try_parse_json(value)
            if parsed is None:
                continue

            key_lower = key.lower()
            if (
                key_lower == "aiservice.prompts"
                or key_lower == "aiservice.generations"
                or "aichat" in key_lower
                or "composer" in key_lower
                or self._looks_like_message_payload(parsed)
            ):
                rows.append(
                    {
                        "table": "ItemTable",
                        "key": key,
                        "data": parsed,
                    }
                )

        return rows

    def _extract_table_rows(self, cursor: sqlite3.Cursor, table: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            cursor.execute(f"SELECT * FROM '{table}'")
        except sqlite3.Error:
            return rows

        for row in cursor.fetchall():
            rows.append({"table": table, "data": dict(row)})

        return rows

    @staticmethod
    def _looks_like_message_payload(parsed: dict[str, Any] | list[Any]) -> bool:
        text_blob = str(parsed).lower()
        return any(token in text_blob for token in ("message", "content", "chat", "prompt"))

    def _parse_itemtable_payload(self, key: str, data: Any) -> list[RawMessage]:
        key_lower = key.lower()

        if key_lower == "aiservice.prompts":
            return self._parse_prompts(data)
        if key_lower == "aiservice.generations":
            return self._parse_generations(data)
        if "aichat" in key_lower and "chatdata" in key_lower:
            return self._parse_chatdata(data)

        return self._parse_generic_payload(data)

    def _parse_prompts(self, data: Any) -> list[RawMessage]:
        messages: list[RawMessage] = []
        if not isinstance(data, list):
            return messages

        for item in data:
            if not isinstance(item, dict):
                continue
            content = self._extract_content(item, keys=["text", "content", "prompt"])
            timestamp = self._extract_timestamp(item)
            message = self._build_message("user", content, timestamp)
            if message:
                messages.append(message)

        return messages

    def _parse_generations(self, data: Any) -> list[RawMessage]:
        messages: list[RawMessage] = []
        if not isinstance(data, list):
            return messages

        for item in data:
            if not isinstance(item, dict):
                continue
            content = self._extract_content(
                item,
                keys=["textDescription", "text", "content", "message"],
            )
            timestamp = self._extract_timestamp(item)
            message = self._build_message("assistant", content, timestamp)
            if message:
                messages.append(message)

        return messages

    def _parse_chatdata(self, data: Any) -> list[RawMessage]:
        bubbles: list[dict[str, Any]] = []
        self._collect_chat_bubbles(data, bubbles)

        messages: list[RawMessage] = []
        for bubble in bubbles:
            role_raw = str(bubble.get("type", bubble.get("role", "assistant"))).lower()
            role = "user" if role_raw in {"user", "human"} else "assistant"
            content = self._extract_content(
                bubble,
                keys=["text", "rawText", "content", "message", "body", "markdown"],
            )
            timestamp = self._extract_timestamp(bubble)
            tool_calls = self._extract_tool_calls(bubble)
            message = self._build_message(role, content, timestamp, tool_calls)
            if message:
                messages.append(message)

        return messages

    def _collect_chat_bubbles(self, node: Any, bubbles: list[dict[str, Any]]) -> None:
        if isinstance(node, list):
            for item in node:
                self._collect_chat_bubbles(item, bubbles)
            return

        if not isinstance(node, dict):
            return

        bubble_list = node.get("bubbles")
        if isinstance(bubble_list, list):
            for bubble in bubble_list:
                if isinstance(bubble, dict):
                    bubbles.append(bubble)

        tabs = node.get("tabs")
        if isinstance(tabs, list):
            for tab in tabs:
                self._collect_chat_bubbles(tab, bubbles)

        # Cursor chat payloads sometimes nest by generated IDs.
        for key, value in node.items():
            if key in {"bubbles", "tabs"}:
                continue
            if isinstance(value, (dict, list)):
                self._collect_chat_bubbles(value, bubbles)

    def _parse_generic_payload(self, data: Any) -> list[RawMessage]:
        messages: list[RawMessage] = []

        if isinstance(data, dict):
            nested_messages = data.get("messages", data.get("conversation"))
            if isinstance(nested_messages, list):
                for item in nested_messages:
                    if isinstance(item, dict):
                        parsed = self._parse_message_dict(item)
                        if parsed:
                            messages.append(parsed)
                return messages

            parsed = self._parse_message_dict(data)
            if parsed:
                messages.append(parsed)
            return messages

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parsed = self._parse_message_dict(item)
                    if parsed:
                        messages.append(parsed)

        return messages

    def _parse_message_dict(self, payload: dict[str, Any]) -> RawMessage | None:
        role_value = payload.get(
            "role",
            payload.get("type", payload.get("author", payload.get("sender", ""))),
        )
        role_raw = str(role_value).lower()
        role = "user" if role_raw in {"user", "human", "prompt"} else "assistant"

        content = self._extract_content(
            payload,
            keys=[
                "content",
                "message",
                "text",
                "body",
                "rawText",
                "textDescription",
            ],
        )
        timestamp = self._extract_timestamp(payload)
        tool_calls = self._extract_tool_calls(payload)
        return self._build_message(role, content, timestamp, tool_calls)

    @staticmethod
    def _extract_content(payload: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for block in value:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content") or block.get("value")
                        if text is not None:
                            parts.append(str(text))
                    else:
                        parts.append(str(block))
                return " ".join(parts)
            if isinstance(value, dict):
                for nested_key in ["text", "content", "value", "markdown"]:
                    if nested_key in value and value[nested_key] is not None:
                        return str(value[nested_key])
        return ""

    @staticmethod
    def _extract_rich_text(raw_value: Any) -> str:
        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value

        text_parts: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "text" and isinstance(node.get("text"), str):
                    text_parts.append(node["text"])
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        visit(value)
                return
            if isinstance(node, list):
                for value in node:
                    visit(value)

        visit(parsed)
        return "\n".join(part for part in text_parts if part.strip()).strip()

    def _extract_timestamp(self, payload: dict[str, Any]) -> datetime | None:
        for key in ["unixMs", "timestamp", "created_at", "time", "ts", "createdAt", "unix_ms"]:
            if key in payload:
                parsed = self._parse_timestamp(payload[key])
                if parsed:
                    return parsed
        return None

    def _parse_timestamp(self, value: Any) -> datetime | None:
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 1e12:
                    timestamp /= 1000
                return datetime.fromtimestamp(timestamp, tz=UTC)
            if isinstance(value, str):
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return self._normalize_dt(parsed)
        except (OSError, TypeError, ValueError):
            return None

        return None

    def _extract_tool_calls(self, payload: dict[str, Any]) -> list[RawToolCall]:
        tool_calls: list[RawToolCall] = []
        for key in ["tool_calls", "toolCalls", "tools", "function_calls"]:
            if key not in payload:
                continue

            value = payload[key]
            if not isinstance(value, list):
                break

            for tool_call in value:
                if not isinstance(tool_call, dict):
                    continue
                args = tool_call.get("arguments", tool_call.get("args", {}))
                if not isinstance(args, dict):
                    args = {"raw": args}
                tool_calls.append(
                    RawToolCall(
                        tool=str(tool_call.get("name", tool_call.get("function", "unknown"))),
                        args=args,
                        result=(
                            str(tool_call.get("result"))
                            if tool_call.get("result") is not None
                            else None
                        ),
                        success=bool(tool_call.get("success", True)),
                    )
                )
            break

        tool_former = payload.get("toolFormerData")
        if isinstance(tool_former, dict):
            args_raw = tool_former.get("params", {})
            args: dict[str, Any]
            if isinstance(args_raw, str):
                parsed = self._try_parse_json(args_raw)
                args = parsed if isinstance(parsed, dict) else {"raw": args_raw}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {"raw": args_raw}

            status = str(tool_former.get("status", "")).strip().lower()
            success = status in {"", "completed", "success", "succeeded", "done"}
            result = tool_former.get("result")
            tool_calls.append(
                RawToolCall(
                    tool=str(tool_former.get("name", tool_former.get("tool", "unknown"))),
                    args=args,
                    result=str(result) if result is not None else None,
                    success=success,
                )
            )

        return tool_calls

    @staticmethod
    def _build_message(
        role: str,
        content: str,
        timestamp: datetime | None,
        tool_calls: list[RawToolCall] | None = None,
    ) -> RawMessage | None:
        if len(content.strip()) < 3:
            return None
        return RawMessage(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_calls=tool_calls or [],
        )

    @staticmethod
    def _infer_title_from_messages(messages: list[RawMessage]) -> str | None:
        for preferred_role in ("user", "assistant"):
            for message in messages:
                if message.role != preferred_role:
                    continue
                normalized = " ".join(message.content.split())
                if len(normalized) < 5 or normalized == "[tool-result]":
                    continue
                if len(normalized) > 96:
                    normalized = f"{normalized[:93].rstrip()}..."
                return normalized
        return None
