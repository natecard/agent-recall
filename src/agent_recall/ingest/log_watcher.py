from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_recall.ingest.base import RawMessage
from agent_recall.ingest.claude_code import ClaudeCodeIngester


@dataclass
class _FileState:
    offset: int
    inode: int | None
    buffer: bytes


class LogWatcher:
    """Poll Claude Code JSONL session logs for new events."""

    def __init__(
        self,
        *,
        project_path: Path | None = None,
        sessions_dir: Path | None = None,
        poll_interval: float = 0.5,
        start_at_end: bool = True,
    ) -> None:
        self._ingester = ClaudeCodeIngester(project_path=project_path)
        self._sessions_dir = sessions_dir
        self.poll_interval = max(poll_interval, 0.05)
        self.start_at_end = start_at_end
        self._state: dict[Path, _FileState] = {}

    def poll(self) -> list[RawMessage]:
        sessions_dir = self._resolve_sessions_dir()
        if sessions_dir is None or not sessions_dir.exists():
            return []

        events: list[RawMessage] = []
        paths = sorted(sessions_dir.glob("*.jsonl"))
        active = set(paths)
        for path in list(self._state.keys()):
            if path not in active:
                self._state.pop(path, None)

        for path in paths:
            events.extend(self._read_new_events(path))
        return events

    def watch(
        self,
        *,
        on_message: Callable[[RawMessage], None] | None = None,
        max_events: int | None = None,
        max_seconds: float | None = None,
    ) -> int:
        start = time.monotonic()
        seen = 0

        while True:
            messages = self.poll()
            if messages:
                for message in messages:
                    if on_message is not None:
                        on_message(message)
                seen += len(messages)
                if max_events is not None and seen >= max_events:
                    break

            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                break
            time.sleep(self.poll_interval)

        return seen

    def _resolve_sessions_dir(self) -> Path | None:
        if self._sessions_dir is not None:
            return self._sessions_dir
        return self._ingester.get_sessions_dir()

    def _read_new_events(self, path: Path) -> list[RawMessage]:
        try:
            stat = path.stat()
        except OSError:
            self._state.pop(path, None)
            return []

        inode = getattr(stat, "st_ino", None)
        state = self._state.get(path)
        if state is None:
            offset = stat.st_size if self.start_at_end else 0
            state = _FileState(offset=offset, inode=inode, buffer=b"")
            self._state[path] = state
            if self.start_at_end:
                return []

        if inode is not None and state.inode is not None and inode != state.inode:
            state.offset = 0
            state.buffer = b""
        elif stat.st_size < state.offset:
            state.offset = 0
            state.buffer = b""

        state.inode = inode
        if stat.st_size == state.offset:
            return []

        read_offset = state.offset
        try:
            with path.open("rb") as handle:
                handle.seek(read_offset)
                chunk = handle.read()
                state.offset = handle.tell()
        except OSError:
            return []

        data = state.buffer + chunk
        if not data:
            return []

        lines = data.split(b"\n")
        if data.endswith(b"\n"):
            state.buffer = b""
        else:
            state.buffer = lines.pop() if lines else data

        messages: list[RawMessage] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            message = self._ingester.parse_event(payload)
            if message is not None:
                messages.append(message)

        # On ext4 (Ubuntu), inodes are reused when a file is deleted and recreated.
        # If we read from a non-zero offset and got 0 valid messages, we may be
        # in the middle of a rotated file—retry from the beginning.
        if not messages and read_offset > 0:
            state.offset = 0
            state.buffer = b""
            return self._read_new_events(path)

        return messages
