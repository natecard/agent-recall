from __future__ import annotations

import json
import os
import platform
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester


class CursorIngester(SessionIngester):
    """Ingest Cursor conversation history from workspace SQLite storage."""

    def __init__(
        self,
        project_path: Path | None = None,
        cursor_db_path: Path | None = None,
        workspace_storage_dir: Path | None = None,
        include_all_workspaces: bool = False,
    ):
        self.project_path = (project_path or Path.cwd()).resolve()
        self.cursor_db_path = cursor_db_path.resolve() if cursor_db_path else None
        self.storage_dir = (
            workspace_storage_dir.resolve() if workspace_storage_dir else self._get_storage_dir()
        )
        self.include_all_workspaces = include_all_workspaces

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

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        db_paths = self._find_workspace_dbs()
        if not db_paths:
            return []

        filtered: list[Path] = []
        if since:
            normalized_since = self._normalize_dt(since)
            for db_path in db_paths:
                mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=UTC)
                if mtime >= normalized_since:
                    filtered.append(db_path)
            return filtered

        return db_paths

    def get_session_id(self, path: Path) -> str:
        mtime = int(path.stat().st_mtime)
        return f"cursor-{path.parent.name}-{mtime}"

    def parse_session(self, path: Path) -> RawSession:
        records = self._extract_conversations(path)

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

        inferred_project_path = self._project_path_from_workspace_metadata(path)
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            project_path=inferred_project_path or self.project_path,
            started_at=earliest_time or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            ended_at=latest_time,
            messages=deduped,
        )

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
