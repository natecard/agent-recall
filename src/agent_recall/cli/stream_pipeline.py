from __future__ import annotations

import codecs
import json
import os
import selectors
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

_STREAM_DEBUG_ENV = "AGENT_RECALL_RALPH_STREAM_DEBUG"
_STREAM_DEBUG_MAX_MB_ENV = "AGENT_RECALL_RALPH_STREAM_DEBUG_MAX_MB"
_DEFAULT_DEBUG_MAX_MB = 8.0
_MAX_BUNDLES = 25


def stream_debug_enabled() -> bool:
    value = os.getenv(_STREAM_DEBUG_ENV, "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def stream_debug_dir(cwd: Path) -> Path:
    return cwd / ".agent" / "ralph" / ".runtime" / "stream-debug"


def _parse_max_debug_bytes() -> int:
    raw = os.getenv(_STREAM_DEBUG_MAX_MB_ENV, str(_DEFAULT_DEBUG_MAX_MB)).strip()
    try:
        mb = float(raw)
    except ValueError:
        mb = _DEFAULT_DEBUG_MAX_MB
    if mb <= 0:
        mb = _DEFAULT_DEBUG_MAX_MB
    return int(mb * 1024 * 1024)


def _preview_text(value: str, *, max_chars: int = 200) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return f"{value[: max_chars - 3]}..."


class _StreamDebugSink:
    def __init__(self, *, cwd: Path, context: str, transport: str) -> None:
        self._enabled = stream_debug_enabled()
        self._context = context
        self._transport = transport
        self._event_fp = None
        self._raw_fp = None
        self._raw_written = 0
        self._raw_limit = _parse_max_debug_bytes()
        self._raw_truncated = False
        self.events_path: Path | None = None
        self.raw_path: Path | None = None
        if not self._enabled:
            return

        out_dir = stream_debug_dir(cwd)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"stream-{timestamp}-{os.getpid()}-{time.time_ns() % 1_000_000}"
        self.events_path = out_dir / f"{run_id}.jsonl"
        self.raw_path = out_dir / f"{run_id}.raw.log"
        self._event_fp = self.events_path.open("w", encoding="utf-8")
        self._raw_fp = self.raw_path.open("w", encoding="utf-8")
        self.emit_event(event="start", preview=run_id)

    def emit_event(
        self,
        *,
        event: str,
        bytes_count: int = 0,
        preview: str = "",
        emitted_fragment_len: int = 0,
        exit_code: int | None = None,
    ) -> None:
        if not self._enabled or self._event_fp is None:
            return
        payload = {
            "ts_ns": time.time_ns(),
            "context": self._context,
            "event": event,
            "transport": self._transport,
            "bytes": int(bytes_count),
            "preview": _preview_text(preview),
            "emitted_fragment_len": int(emitted_fragment_len),
            "exit_code": exit_code,
        }
        self._event_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._event_fp.flush()

    def append_raw(self, fragment: str) -> None:
        if not self._enabled or self._raw_fp is None:
            return
        encoded = fragment.encode("utf-8", errors="replace")
        remaining = self._raw_limit - self._raw_written
        if remaining <= 0:
            if not self._raw_truncated:
                self._raw_truncated = True
                self.emit_event(event="raw_truncated")
            return
        if len(encoded) > remaining:
            to_write = encoded[:remaining]
            self._raw_fp.write(to_write.decode("utf-8", errors="replace"))
            self._raw_fp.flush()
            self._raw_written += len(to_write)
            if not self._raw_truncated:
                self._raw_truncated = True
                self.emit_event(event="raw_truncated")
            return
        self._raw_fp.write(fragment)
        self._raw_fp.flush()
        self._raw_written += len(encoded)

    def close(self, *, exit_code: int) -> None:
        if not self._enabled:
            return
        self.emit_event(
            event="complete",
            exit_code=exit_code,
            preview=str(self.events_path) if self.events_path is not None else "",
        )
        if self._event_fp is not None:
            self._event_fp.close()
            self._event_fp = None
        if self._raw_fp is not None:
            self._raw_fp.close()
            self._raw_fp = None
        self._prune_old_bundles()

    def _prune_old_bundles(self) -> None:
        if self.events_path is None:
            return
        directory = self.events_path.parent
        bundles = sorted(
            directory.glob("stream-*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in bundles[_MAX_BUNDLES:]:
            raw = directory / f"{stale.stem}.raw.log"
            try:
                stale.unlink()
            except OSError:
                pass
            try:
                raw.unlink()
            except OSError:
                pass


def run_streaming_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    on_emit: Callable[[str], None],
    context: str,
    partial_flush_ms: int = 120,
    transport: str = "pipe",
) -> int:
    """Run a command and emit decoded stream fragments in-order.

    Fragments are emitted on newline or carriage return boundaries, with periodic
    partial flushes to avoid waiting indefinitely for delimiters.
    """

    flush_interval = max(int(partial_flush_ms), 1) / 1000.0
    sink = _StreamDebugSink(cwd=cwd, context=context, transport=transport)
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    assert process.stdout is not None  # noqa: S101

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    selector = selectors.DefaultSelector()
    stream_fd = process.stdout.fileno()
    os.set_blocking(stream_fd, False)
    selector.register(stream_fd, selectors.EVENT_READ)

    pending = ""
    eof = False
    last_emit_monotonic = time.monotonic()

    def emit_fragment(fragment: str, *, event: str) -> None:
        nonlocal last_emit_monotonic
        on_emit(fragment)
        sink.append_raw(fragment)
        sink.emit_event(
            event=event,
            bytes_count=len(fragment.encode("utf-8", errors="replace")),
            preview=fragment,
            emitted_fragment_len=len(fragment),
        )
        last_emit_monotonic = time.monotonic()

    def emit_delimited_fragments() -> None:
        nonlocal pending
        while pending:
            line_break = pending.find("\n")
            carriage = pending.find("\r")
            points = [idx for idx in (line_break, carriage) if idx != -1]
            if not points:
                return
            boundary = min(points)
            fragment = pending[: boundary + 1]
            pending = pending[boundary + 1 :]
            emit_fragment(fragment, event="emit_delimiter")

    def read_available_bytes() -> None:
        nonlocal pending, eof
        while True:
            try:
                chunk = os.read(stream_fd, 4096)
            except BlockingIOError:
                return
            if not chunk:
                eof = True
                sink.emit_event(event="eof")
                return
            sink.emit_event(
                event="read_chunk",
                bytes_count=len(chunk),
                preview=chunk.decode("utf-8", errors="replace"),
            )
            decoded = decoder.decode(chunk, final=False)
            if not decoded:
                continue
            pending += decoded
            emit_delimited_fragments()

    exit_code = 1
    try:
        while not eof:
            now = time.monotonic()
            timeout = flush_interval
            if pending:
                elapsed = now - last_emit_monotonic
                timeout = max(0.0, flush_interval - elapsed)
            events = selector.select(timeout)
            if events:
                read_available_bytes()
            elif pending and (time.monotonic() - last_emit_monotonic) >= flush_interval:
                fragment = pending
                pending = ""
                emit_fragment(fragment, event="emit_partial")
            if process.poll() is not None:
                read_available_bytes()

        final_fragment = decoder.decode(b"", final=True)
        if final_fragment:
            pending += final_fragment
        if pending:
            fragment = pending
            pending = ""
            emit_fragment(fragment, event="emit_eof_flush")
        exit_code = int(process.wait())
        return exit_code
    finally:
        selector.close()
        sink.close(exit_code=exit_code)
