from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester


class CodexIngester(SessionIngester):
    """Ingest OpenAI Codex session logs from local JSON/JSONL storage."""

    def __init__(
        self,
        project_path: Path | None = None,
        codex_dir: Path | None = None,
    ):
        self.project_path = (project_path or Path.cwd()).resolve()
        self.codex_dir = (codex_dir or self._default_codex_dir()).expanduser().resolve()
        self.sessions_dir = self.codex_dir / "sessions"
        self._session_meta_cache: dict[Path, dict[str, Any]] = {}

    @property
    def source_name(self) -> str:
        return "codex"

    @staticmethod
    def _default_codex_dir() -> Path:
        codex_home = os.environ.get("CODEX_HOME", "").strip()
        if codex_home:
            return Path(codex_home)
        return Path.home() / ".codex"

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

    def _read_session_meta(self, path: Path) -> dict[str, Any]:
        cached = self._session_meta_cache.get(path)
        if cached is not None:
            return dict(cached)

        session_id: str | None = None
        cwd: str | None = None
        started_at: datetime | None = None

        if path.suffix == ".json":
            payload = self._read_json_dict(path)
            if isinstance(payload, dict):
                session = payload.get("session")
                if isinstance(session, dict):
                    raw_id = session.get("id")
                    if isinstance(raw_id, str) and raw_id.strip():
                        session_id = raw_id.strip()
                    raw_cwd = session.get("cwd")
                    if isinstance(raw_cwd, str) and raw_cwd.strip():
                        cwd = raw_cwd
                    started_at = self._parse_timestamp(session.get("timestamp"))
        else:
            try:
                with path.open(encoding="utf-8") as file:
                    for raw_line in file:
                        line = raw_line.strip()
                        if not line:
                            continue

                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(event, dict):
                            continue

                        if started_at is None:
                            started_at = self._parse_timestamp(event.get("timestamp"))

                        event_type = str(event.get("type", "")).lower()
                        payload = event.get("payload")
                        if not isinstance(payload, dict):
                            continue

                        if event_type == "session_meta":
                            raw_id = payload.get("id")
                            if isinstance(raw_id, str) and raw_id.strip():
                                session_id = raw_id.strip()
                            raw_cwd = payload.get("cwd")
                            if isinstance(raw_cwd, str) and raw_cwd.strip():
                                cwd = raw_cwd
                            if started_at is None:
                                started_at = self._parse_timestamp(payload.get("timestamp"))
                        elif event_type == "turn_context" and not cwd:
                            raw_cwd = payload.get("cwd")
                            if isinstance(raw_cwd, str) and raw_cwd.strip():
                                cwd = raw_cwd

                        if session_id and cwd and started_at is not None:
                            break
            except OSError:
                pass

        metadata = {
            "session_id": session_id,
            "cwd": cwd,
            "started_at": started_at,
        }
        self._session_meta_cache[path] = metadata
        return dict(metadata)

    def _session_matches_project(self, path: Path) -> bool:
        metadata = self._read_session_meta(path)
        session_cwd = self._resolve_path(metadata.get("cwd"))
        if session_cwd is None:
            return False
        return session_cwd == self.project_path

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        if not self.sessions_dir.exists():
            return []

        normalized_since = self._normalize_dt(since) if since else None
        discovered: list[tuple[float, Path]] = []

        for session_file in self.sessions_dir.rglob("*"):
            if session_file.suffix not in {".json", ".jsonl"}:
                continue
            if not self._session_matches_project(session_file):
                continue

            metadata = self._read_session_meta(session_file)
            session_time = metadata.get("started_at")
            updated_at = (
                session_time
                if isinstance(session_time, datetime)
                else datetime.fromtimestamp(session_file.stat().st_mtime, tz=UTC)
            )
            if normalized_since and updated_at < normalized_since:
                continue

            discovered.append((updated_at.timestamp(), session_file))

        discovered.sort(key=lambda item: (item[0], item[1].name))
        return [path for _, path in discovered]

    def get_session_id(self, path: Path) -> str:
        metadata = self._read_session_meta(path)
        native_id = metadata.get("session_id")
        if not isinstance(native_id, str) or not native_id.strip():
            native_id = path.stem
        return f"codex-{native_id}"

    @staticmethod
    def _normalize_role(raw_role: Any) -> str | None:
        role = str(raw_role or "").strip().lower()
        if role in {"user", "human"}:
            return "user"
        if role in {"assistant", "ai", "model"}:
            return "assistant"
        return None

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        if isinstance(raw_content, str):
            return raw_content.strip()

        if isinstance(raw_content, list):
            parts: list[str] = []
            for block in raw_content:
                if isinstance(block, str):
                    text = block.strip()
                    if text:
                        parts.append(text)
                    continue
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n\n".join(parts).strip()

        if isinstance(raw_content, dict):
            text = raw_content.get("text")
            if isinstance(text, str):
                return text.strip()

        return ""

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

    @staticmethod
    def _parse_tool_arguments(raw_value: Any) -> dict[str, Any]:
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                return {}
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return {"raw": stripped}
            if isinstance(decoded, dict):
                return decoded
            return {"raw": decoded}
        if raw_value is None:
            return {}
        return {"raw": raw_value}

    @staticmethod
    def _stringify_result(raw_value: Any) -> str | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            return raw_value
        try:
            return json.dumps(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)

    def _parse_jsonl_session(self, path: Path) -> RawSession:
        metadata = self._read_session_meta(path)
        session_id = self.get_session_id(path)

        messages: list[RawMessage] = []
        message_rows: list[tuple[float, int, RawMessage]] = []
        pending_tool_calls: list[RawToolCall] = []
        pending_by_call_id: dict[str, RawToolCall] = {}

        started_at = metadata.get("started_at")
        if not isinstance(started_at, datetime):
            started_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        ended_at = started_at

        with path.open(encoding="utf-8") as file:
            for index, raw_line in enumerate(file):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(event, dict):
                    continue

                event_time = self._parse_timestamp(event.get("timestamp"))
                if event_time:
                    started_at = min(started_at, event_time)
                    ended_at = max(ended_at, event_time)

                if str(event.get("type", "")).lower() != "response_item":
                    continue

                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue

                payload_type = str(payload.get("type", "")).lower()
                if payload_type == "message":
                    role = self._normalize_role(payload.get("role"))
                    if role is None:
                        continue
                    content = self._extract_message_content(payload.get("content"))
                    tool_calls = list(pending_tool_calls)
                    pending_tool_calls.clear()
                    pending_by_call_id.clear()

                    if len(content.strip()) < 3 and not tool_calls:
                        continue

                    message = RawMessage(
                        role=role,
                        content=content if content else "[tool-result]",
                        timestamp=event_time,
                        tool_calls=tool_calls,
                    )
                    sort_key = event_time.timestamp() if event_time else float("inf")
                    message_rows.append((sort_key, index, message))
                    continue

                if payload_type in {"function_call", "custom_tool_call"}:
                    raw_arguments = payload.get("arguments")
                    if payload_type == "custom_tool_call":
                        raw_arguments = payload.get("input")
                    tool_call = RawToolCall(
                        tool=str(payload.get("name") or "tool"),
                        args=self._parse_tool_arguments(raw_arguments),
                    )
                    pending_tool_calls.append(tool_call)

                    call_id = payload.get("call_id")
                    if isinstance(call_id, str) and call_id.strip():
                        pending_by_call_id[call_id] = tool_call
                    continue

                if payload_type in {"function_call_output", "custom_tool_call_output"}:
                    output_text = self._stringify_result(payload.get("output"))
                    success = not bool(payload.get("is_error"))
                    call_id = payload.get("call_id")

                    matched: RawToolCall | None = None
                    if isinstance(call_id, str) and call_id.strip():
                        matched = pending_by_call_id.get(call_id)
                    if matched is None and pending_tool_calls:
                        matched = pending_tool_calls[-1]
                    if matched is not None:
                        matched.result = output_text
                        matched.success = success
                    else:
                        pending_tool_calls.append(
                            RawToolCall(
                                tool="tool_result",
                                result=output_text,
                                success=success,
                            )
                        )

        if pending_tool_calls:
            message_rows.append(
                (
                    ended_at.timestamp(),
                    len(message_rows),
                    RawMessage(
                        role="assistant",
                        content="[tool-result]",
                        timestamp=ended_at,
                        tool_calls=list(pending_tool_calls),
                    ),
                )
            )

        message_rows.sort(key=lambda item: (item[0], item[1]))
        messages = [item[2] for item in message_rows]

        native_session_id = str(metadata.get("session_id") or path.stem)
        title = self._infer_title(messages, fallback=native_session_id)
        session_project = self._resolve_path(metadata.get("cwd")) or self.project_path

        return RawSession(
            source=self.source_name,
            session_id=session_id,
            title=title,
            project_path=session_project,
            started_at=started_at,
            ended_at=ended_at,
            messages=messages,
        )

    def _parse_legacy_json_session(self, path: Path) -> RawSession:
        payload = self._read_json_dict(path)
        if not payload:
            raise ValueError(f"Invalid Codex session file: {path}")

        session_payload = payload.get("session")
        session_data = session_payload if isinstance(session_payload, dict) else {}
        items = payload.get("items")
        entries = items if isinstance(items, list) else []

        messages: list[RawMessage] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            role = self._normalize_role(entry.get("role"))
            if role is None:
                continue
            content = self._extract_message_content(entry.get("content"))
            if len(content.strip()) < 3:
                continue
            messages.append(
                RawMessage(
                    role=role,
                    content=content,
                )
            )

        started_at = self._parse_timestamp(session_data.get("timestamp"))
        if started_at is None:
            started_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

        native_session_id = str(session_data.get("id") or path.stem)
        title = self._infer_title(messages, fallback=native_session_id)
        session_project = self._resolve_path(session_data.get("cwd")) or self.project_path

        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            title=title,
            project_path=session_project,
            started_at=started_at,
            ended_at=None,
            messages=messages,
        )

    def parse_session(self, path: Path) -> RawSession:
        if path.suffix == ".jsonl":
            return self._parse_jsonl_session(path)
        if path.suffix == ".json":
            return self._parse_legacy_json_session(path)
        raise ValueError(f"Unsupported Codex session format: {path}")
