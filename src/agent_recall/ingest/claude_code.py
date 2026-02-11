from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester


class ClaudeCodeIngester(SessionIngester):
    """Ingest Claude Code session transcripts from JSONL storage."""

    def __init__(self, project_path: Path | None = None):
        self.project_path = (project_path or Path.cwd()).resolve()
        self.claude_dir = Path.home() / ".claude"

    @property
    def source_name(self) -> str:
        return "claude-code"

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def _find_project_dir(self) -> Path | None:
        projects_dir = self.claude_dir / "projects"
        if not projects_dir.exists():
            return None

        path_str = str(self.project_path)
        hash_inputs = [path_str, path_str.lower(), path_str.rstrip("/")]

        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            matched = self._is_path_mapped_project(project_dir)
            if matched:
                return project_dir

            if self._is_hashed_match(project_dir.name, hash_inputs):
                return project_dir

        return None

    def _is_path_mapped_project(self, project_dir: Path) -> bool:
        for config_name in ["project.json", "config.json", ".project", "settings.json"]:
            config_file = project_dir / config_name
            if not config_file.exists():
                continue

            try:
                config = json.loads(config_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            project_path = (
                config.get("path")
                or config.get("folder")
                or config.get("root")
                or config.get("projectPath")
                or config.get("workspacePath")
            )
            if not project_path:
                continue

            try:
                if Path(str(project_path)).resolve() == self.project_path:
                    return True
            except OSError:
                continue

        return False

    @staticmethod
    def _is_hashed_match(dir_name: str, hash_inputs: list[str]) -> bool:
        for hash_input in hash_inputs:
            encoded = hash_input.encode()
            for length in [16, 32, 40, 64]:
                sha_hash = hashlib.sha256(encoded).hexdigest()[:length]
                if dir_name == sha_hash:
                    return True
                md5_hash = hashlib.md5(encoded).hexdigest()[:length]
                if dir_name == md5_hash:
                    return True
        return False

    def _parse_timestamp(self, value: Any) -> datetime | None:
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
                return self._normalize_dt(parsed)
        except (OSError, TypeError, ValueError):
            return None

        return None

    def discover_sessions(self, since: datetime | None = None) -> list[Path]:
        project_dir = self._find_project_dir()
        if not project_dir:
            return []

        sessions_dir = project_dir / "sessions"
        if not sessions_dir.exists():
            return []

        sessions: list[Path] = []
        normalized_since = self._normalize_dt(since) if since else None

        for session_file in sessions_dir.glob("*.jsonl"):
            if normalized_since:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime, tz=UTC)
                if mtime < normalized_since:
                    continue
            sessions.append(session_file)

        return sorted(sessions, key=lambda session: session.stat().st_mtime)

    def get_session_id(self, path: Path) -> str:
        return f"claude-code-{path.stem}"

    def parse_session(self, path: Path) -> RawSession:
        messages: list[RawMessage] = []
        started_at: datetime | None = None
        ended_at: datetime | None = None

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

                timestamp = self._extract_timestamp(event)
                if timestamp:
                    started_at = timestamp if started_at is None else min(started_at, timestamp)
                    ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

                event_type = str(event.get("type", "message")).lower()
                if event_type not in {"message", "assistant", "user", "human", "text"}:
                    if event_type != "tool_result" and "tool_result" not in event:
                        continue

                role = self._extract_role(event)
                content = self._extract_content(event)
                tool_calls = self._extract_tool_calls(event)

                if len(content.strip()) < 3 and not tool_calls:
                    continue

                messages.append(
                    RawMessage(
                        role=role,
                        content=content if content else "[tool-result]",
                        timestamp=timestamp,
                        tool_calls=tool_calls,
                    )
                )

        messages.sort(
            key=lambda message: message.timestamp.timestamp() if message.timestamp else 0.0
        )

        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            title=self._infer_title(messages, fallback=path.stem),
            project_path=self.project_path,
            started_at=started_at
            or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
            ended_at=ended_at,
            messages=messages,
        )

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

        cleaned_fallback = " ".join(
            fallback.replace("-", " ").replace("_", " ").split()
        ).strip()
        return cleaned_fallback or fallback

    def _extract_timestamp(self, event: dict[str, Any]) -> datetime | None:
        for field in ["timestamp", "time", "ts", "created_at", "createdAt"]:
            if field in event:
                parsed = self._parse_timestamp(event[field])
                if parsed:
                    return parsed
        return None

    @staticmethod
    def _extract_role(event: dict[str, Any]) -> str:
        role = str(event.get("role", "assistant")).lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"assistant", "ai", "claude", "model"}:
            return "assistant"
        return "assistant"

    @staticmethod
    def _extract_content(event: dict[str, Any]) -> str:
        content = event.get("content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue
                block_type = str(block.get("type", "")).lower()
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    parts.append(f"[Using tool: {block.get('name', 'unknown')}]")
            return " ".join(parts)

        return ""

    def _extract_tool_calls(self, event: dict[str, Any]) -> list[RawToolCall]:
        tool_calls: list[RawToolCall] = []

        for field in ["tool_calls", "tool_use", "tools", "toolCalls"]:
            if field not in event:
                continue
            value = event[field]
            if not isinstance(value, list):
                break
            for tool_call in value:
                if not isinstance(tool_call, dict):
                    continue
                args = tool_call.get("input", tool_call.get("args", tool_call.get("arguments", {})))
                if not isinstance(args, dict):
                    args = {"raw": args}
                tool_calls.append(
                    RawToolCall(
                        tool=str(tool_call.get("name", tool_call.get("tool", "unknown"))),
                        args=args,
                        result=(
                            str(tool_call.get("result", tool_call.get("output")))
                            if tool_call.get("result", tool_call.get("output")) is not None
                            else None
                        ),
                        success=not bool(tool_call.get("error", tool_call.get("is_error", False))),
                    )
                )
            break

        if event.get("type") == "tool_result" or "tool_result" in event:
            result_data = event.get("tool_result", event)
            if isinstance(result_data, dict):
                result_content = result_data.get("content", result_data.get("output", ""))
                tool_calls.append(
                    RawToolCall(
                        tool=str(
                            result_data.get("tool_use_id", result_data.get("name", "unknown"))
                        ),
                        args={},
                        result=str(result_content),
                        success=not bool(result_data.get("is_error", False)),
                    )
                )

        return tool_calls
