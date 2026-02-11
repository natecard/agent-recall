from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester


class OpenCodeIngester(SessionIngester):
    """Ingest OpenCode sessions from local JSON storage."""

    def __init__(
        self,
        project_path: Path | None = None,
        opencode_dir: Path | None = None,
    ):
        self.project_path = (project_path or Path.cwd()).resolve()
        self.opencode_dir = (opencode_dir or self._default_opencode_dir()).expanduser().resolve()
        self.storage_dir = self.opencode_dir / "storage"
        self._project_worktree_cache: dict[str, Path | None] = {}

    @property
    def source_name(self) -> str:
        return "opencode"

    @staticmethod
    def _default_opencode_dir() -> Path:
        system = platform.system()
        if system == "Windows":
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata:
                return Path(local_appdata) / "opencode"
            return Path.home() / "AppData/Local/opencode"

        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / "opencode"
        return Path.home() / ".local/share/opencode"

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @classmethod
    def _parse_timestamp(cls, value: Any) -> datetime | None:
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 1e12:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp, tz=UTC)
            if isinstance(value, str):
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return cls._normalize_dt(parsed)
        except (OSError, TypeError, ValueError):
            return None

        return None

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _resolve_path(raw_value: Any) -> Path | None:
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        try:
            return Path(raw_value).expanduser().resolve()
        except OSError:
            return None

    def _project_worktree(self, project_id: Any) -> Path | None:
        if not isinstance(project_id, str) or not project_id:
            return None
        if project_id in self._project_worktree_cache:
            return self._project_worktree_cache[project_id]

        project_path = self.storage_dir / "project" / f"{project_id}.json"
        payload = self._read_json_dict(project_path) if project_path.exists() else None
        worktree = self._resolve_path(payload.get("worktree")) if payload else None
        self._project_worktree_cache[project_id] = worktree
        return worktree

    def _session_matches_project(self, payload: dict[str, Any]) -> bool:
        session_directory = self._resolve_path(payload.get("directory"))
        if session_directory and session_directory == self.project_path:
            return True

        project_worktree = self._project_worktree(payload.get("projectID"))
        if project_worktree and project_worktree == self.project_path:
            return True

        return False

    def _session_updated_at(self, payload: dict[str, Any], fallback_path: Path) -> datetime:
        if isinstance(payload.get("time"), dict):
            time_info = payload["time"]
            updated = self._parse_timestamp(time_info.get("updated"))
            if updated:
                return updated
            created = self._parse_timestamp(time_info.get("created"))
            if created:
                return created
        return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=UTC)

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        sessions_root = self.storage_dir / "session"
        if not sessions_root.exists():
            return []

        normalized_since = self._normalize_dt(since) if since else None
        discovered: list[tuple[float, Path]] = []

        for session_file in sessions_root.rglob("ses_*.json"):
            payload = self._read_json_dict(session_file)
            if not payload:
                continue
            if not self._session_matches_project(payload):
                continue

            updated_at = self._session_updated_at(payload, session_file)
            if normalized_since and updated_at < normalized_since:
                continue

            discovered.append((updated_at.timestamp(), session_file))

        discovered.sort(key=lambda item: (item[0], item[1].name))
        return [path for _, path in discovered]

    def get_session_id(self, path: Path) -> str:
        payload = self._read_json_dict(path)
        native_id = str(payload.get("id") or path.stem) if payload else path.stem
        return f"opencode-{native_id}"

    def _load_message_parts(self, message_id: str) -> list[dict[str, Any]]:
        part_dir = self.storage_dir / "part" / message_id
        if not part_dir.exists():
            return []

        parsed_parts: list[tuple[float | None, str, dict[str, Any]]] = []
        for part_file in part_dir.glob("*.json"):
            payload = self._read_json_dict(part_file)
            if not payload:
                continue
            timestamp = self._part_timestamp(payload)
            parsed_parts.append(
                (
                    timestamp.timestamp() if timestamp else None,
                    part_file.name,
                    payload,
                )
            )

        parsed_parts.sort(
            key=lambda item: (
                item[0] if item[0] is not None else float("inf"),
                item[1],
            )
        )
        return [payload for _, _, payload in parsed_parts]

    def _part_timestamp(self, payload: dict[str, Any]) -> datetime | None:
        time_candidates: list[Any] = []

        part_time = payload.get("time")
        if isinstance(part_time, dict):
            time_candidates.extend([part_time.get("start"), part_time.get("end")])

        state = payload.get("state")
        if isinstance(state, dict):
            state_time = state.get("time")
            if isinstance(state_time, dict):
                time_candidates.extend([state_time.get("start"), state_time.get("end")])

        for candidate in time_candidates:
            parsed = self._parse_timestamp(candidate)
            if parsed:
                return parsed
        return None

    @staticmethod
    def _normalize_role(raw_role: Any) -> str:
        role = str(raw_role or "assistant").strip().lower()
        if role in {"user", "human"}:
            return "user"
        if role in {"assistant", "ai", "model"}:
            return "assistant"
        return "assistant"

    @staticmethod
    def _is_generated_context_block(text: str) -> bool:
        lowered = text.strip().lower()
        if lowered.startswith("called the ") and " tool with the following input:" in lowered:
            return True
        if text.startswith("<file>") and "(End of file" in text:
            return True
        return False

    @staticmethod
    def _file_label(payload: dict[str, Any]) -> str | None:
        filename = payload.get("filename")
        if isinstance(filename, str) and filename.strip():
            return filename.strip()

        source = payload.get("source")
        if isinstance(source, dict):
            source_path = source.get("path")
            if isinstance(source_path, str) and source_path.strip():
                return source_path.strip()

        url = payload.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()

        return None

    def _extract_content(self, role: str, parts: list[dict[str, Any]]) -> str:
        text_parts: list[str] = []
        attachments: list[str] = []

        for part in parts:
            part_type = str(part.get("type", "")).lower()
            if part_type == "text":
                text = str(part.get("text", "")).strip()
                if not text:
                    continue
                if self._is_generated_context_block(text):
                    continue
                text_parts.append(text)
            elif part_type == "file":
                label = self._file_label(part)
                if label:
                    attachments.append(label)

        content = "\n\n".join(text_parts).strip()

        # Preserve user file references without inlining huge generated file dumps.
        if role == "user" and attachments:
            attachment_block = "Attached files:\n" + "\n".join(
                f"- {item}" for item in attachments
            )
            if content:
                content = f"{content}\n\n{attachment_block}"
            else:
                content = attachment_block

        return content

    @staticmethod
    def _stringify_result(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)

    def _extract_tool_calls(self, parts: list[dict[str, Any]]) -> list[RawToolCall]:
        tool_calls: list[RawToolCall] = []

        for part in parts:
            part_type = str(part.get("type", "")).lower()

            if part_type == "tool":
                state = part.get("state", {}) if isinstance(part.get("state"), dict) else {}
                raw_args = state.get("input", {})
                args = raw_args if isinstance(raw_args, dict) else {"raw": raw_args}
                status = str(state.get("status", "")).strip().lower()
                error_like = {"failed", "error", "cancelled", "canceled", "timed_out", "timeout"}
                success = (
                    status not in error_like
                    and not bool(state.get("error"))
                    and not bool(state.get("is_error"))
                )

                duration_ms: int | None = None
                state_time = state.get("time")
                if isinstance(state_time, dict):
                    start = self._parse_timestamp(state_time.get("start"))
                    end = self._parse_timestamp(state_time.get("end"))
                    if start and end:
                        duration_ms = max(0, int((end - start).total_seconds() * 1000))

                tool_calls.append(
                    RawToolCall(
                        tool=str(part.get("tool", state.get("title", "unknown"))),
                        args=args,
                        result=self._stringify_result(state.get("output")),
                        success=success,
                        duration_ms=duration_ms,
                    )
                )
                continue

            if part_type == "patch":
                files = part.get("files")
                args: dict[str, Any] = {}
                if isinstance(files, list):
                    args["files"] = [str(item) for item in files if isinstance(item, str)]
                tool_calls.append(
                    RawToolCall(
                        tool="patch",
                        args=args,
                        result=self._stringify_result(part.get("hash")),
                        success=True,
                    )
                )

        return tool_calls

    def _message_timestamp(
        self,
        message_payload: dict[str, Any],
        parts: list[dict[str, Any]],
    ) -> datetime | None:
        time_info = message_payload.get("time")
        if isinstance(time_info, dict):
            for field in ("created", "completed"):
                parsed = self._parse_timestamp(time_info.get(field))
                if parsed:
                    return parsed

        for part in parts:
            parsed = self._part_timestamp(part)
            if parsed:
                return parsed

        return None

    @staticmethod
    def _infer_title(messages: list[RawMessage], fallback: str) -> str:
        for message in messages:
            if message.role != "user":
                continue
            normalized = " ".join(message.content.split())
            if len(normalized) < 5 or normalized == "[tool-result]":
                continue
            if len(normalized) > 96:
                return f"{normalized[:93].rstrip()}..."
            return normalized

        cleaned = " ".join(fallback.replace("-", " ").replace("_", " ").split()).strip()
        return cleaned or fallback

    def parse_session(self, path: Path) -> RawSession:
        payload = self._read_json_dict(path)
        if not payload:
            raise ValueError(f"Invalid OpenCode session file: {path}")

        native_session_id = str(payload.get("id") or path.stem)
        session_id = self.get_session_id(path)
        message_dir = self.storage_dir / "message" / native_session_id

        messages: list[RawMessage] = []
        started_at = self._session_updated_at(payload, path)
        ended_at = started_at

        if isinstance(payload.get("time"), dict):
            time_info = payload["time"]
            created = self._parse_timestamp(time_info.get("created"))
            updated = self._parse_timestamp(time_info.get("updated"))
            if created:
                started_at = created
            if updated:
                ended_at = updated

        if message_dir.exists():
            message_rows: list[tuple[float, RawMessage]] = []
            for message_file in message_dir.glob("*.json"):
                message_payload = self._read_json_dict(message_file)
                if not message_payload:
                    continue

                message_id = str(message_payload.get("id") or message_file.stem)
                parts = self._load_message_parts(message_id)
                role = self._normalize_role(message_payload.get("role"))
                timestamp = self._message_timestamp(message_payload, parts)
                content = self._extract_content(role, parts)
                tool_calls = self._extract_tool_calls(parts)

                if len(content.strip()) < 3 and not tool_calls:
                    continue

                if timestamp:
                    started_at = min(started_at, timestamp)
                    ended_at = max(ended_at, timestamp)

                raw_message = RawMessage(
                    role=role,
                    content=content if content else "[tool-result]",
                    timestamp=timestamp,
                    tool_calls=tool_calls,
                )
                sort_key = timestamp.timestamp() if timestamp else float("inf")
                message_rows.append((sort_key, raw_message))

            message_rows.sort(key=lambda item: item[0])
            messages = [item[1] for item in message_rows]

        session_directory = self._resolve_path(payload.get("directory"))
        project_path = session_directory or self._project_worktree(payload.get("projectID"))
        if project_path is None:
            project_path = self.project_path

        title = str(payload.get("title") or "").strip()
        if not title:
            title = self._infer_title(messages, fallback=native_session_id)

        return RawSession(
            source=self.source_name,
            session_id=session_id,
            title=title,
            project_path=project_path,
            started_at=started_at,
            ended_at=ended_at,
            messages=messages,
        )
