from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RALPH_PRE_HOOK_NAME = "ralph-pre-tool-use"
RALPH_POST_HOOK_NAME = "ralph-post-tool-use"
RALPH_NOTIFICATION_HOOK_NAME = "ralph-notification"

DEFAULT_BLOCK_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+\*",
    r"drop\s+table",
    r"mkfs\b",
    r"dd\s+if=",
    r"shutdown\b",
    r"reboot\b",
    r"kill\s+-9\s+-1",
    r"format\s+[a-z]:",
]


@dataclass(frozen=True)
class HookPaths:
    hooks_dir: Path
    pre_tool_path: Path
    post_tool_path: Path
    notification_path: Path
    events_path: Path


def get_hook_paths(agent_dir: Path) -> HookPaths:
    hooks_dir = agent_dir / "ralph" / "hooks"
    return HookPaths(
        hooks_dir=hooks_dir,
        pre_tool_path=hooks_dir / "pre_tool_use.py",
        post_tool_path=hooks_dir / "post_tool_use.py",
        notification_path=hooks_dir / "notification.py",
        events_path=agent_dir / "ralph" / "tool_events.jsonl",
    )


def extract_guardrail_patterns(guardrails_text: str) -> list[str]:
    patterns: list[str] = []
    for raw_line in guardrails_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        for match in re.findall(r"`([^`]+)`", line):
            cleaned = match.strip()
            if cleaned:
                patterns.append(cleaned)
        match = re.search(r"(?i)\bblock(?:ed)?\b\s*[:\-]\s*(.+)$", line)
        if match:
            value = match.group(1).strip()
            if value:
                patterns.append(value)
    return patterns


def build_guardrail_patterns(guardrails_text: str) -> list[str]:
    combined = [*extract_guardrail_patterns(guardrails_text), *DEFAULT_BLOCK_PATTERNS]
    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in combined:
        if pattern not in seen:
            ordered.append(pattern)
            seen.add(pattern)
    return ordered


def _payload_to_text(payload: dict[str, Any]) -> str:
    tool = payload.get("tool") or payload.get("name") or payload.get("tool_name") or ""
    args = payload.get("arguments")
    if args is None:
        args = payload.get("input", payload.get("args", payload.get("arguments", {})))
    try:
        args_text = json.dumps(args, default=str)
    except TypeError:
        args_text = str(args)
    return f"{tool} {args_text}".strip()


def should_block_payload(payload: dict[str, Any], patterns: list[str]) -> tuple[bool, str | None]:
    text = _payload_to_text(payload)
    if not text:
        return False, None
    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True, pattern
        except re.error:
            continue
    return False, None


def build_hook_command(script_path: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))}"


def generate_pre_tool_script(
    guardrails_text: str,
    output_path: Path,
    *,
    patterns: list[str] | None = None,
) -> list[str]:
    guardrail_patterns = patterns or build_guardrail_patterns(guardrails_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_json = json.dumps(guardrail_patterns, indent=2)
    script = """#!/usr/bin/env python3
import json
import re
import sys

PATTERNS = __PATTERNS__


def _load_payload():
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _payload_to_text(payload):
    tool = payload.get("tool") or payload.get("name") or payload.get("tool_name") or ""
    args = payload.get("arguments")
    if args is None:
        args = payload.get("input", payload.get("args", payload.get("arguments", {})))
    try:
        args_text = json.dumps(args, default=str)
    except TypeError:
        args_text = str(args)
    return f"{tool} {args_text}".strip()


def main():
    payload = _load_payload()
    text = _payload_to_text(payload)
    if not text:
        return 0
    for pattern in PATTERNS:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                sys.stderr.write(f"Blocked by Ralph guardrails: {pattern}\n")
                return 2
        except re.error:
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    script = script.replace("__PATTERNS__", patterns_json)
    output_path.write_text(script, encoding="utf-8")
    return guardrail_patterns


def _summarize_result(result: Any, limit: int = 200) -> str:
    text = ""
    if result is None:
        text = ""
    elif isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, default=str)
        except TypeError:
            text = str(result)
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def build_tool_event(payload: dict[str, Any]) -> dict[str, Any]:
    tool = payload.get("tool") or payload.get("name") or payload.get("tool_name") or "unknown"
    args = payload.get("arguments")
    if args is None:
        args = payload.get("input", payload.get("args", payload.get("arguments", {})))
    result = payload.get("result", payload.get("output"))
    success = payload.get("success")
    if success is None:
        success = not bool(payload.get("error") or payload.get("is_error"))
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool,
        "arguments": args,
        "result_summary": _summarize_result(result),
        "success": bool(success),
    }
    if payload.get("error"):
        event["error"] = payload.get("error")
    return event


def append_tool_event(events_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = build_tool_event(payload)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str))
        handle.write("\n")
    return event


def generate_post_tool_script(output_path: Path, events_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = """#!/usr/bin/env python3
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

EVENTS_PATH = Path(__EVENTS_PATH__)


def _load_payload():
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _summarize_result(result, limit=200):
    if result is None:
        return ""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, default=str)
        except TypeError:
            text = str(result)
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _build_event(payload):
    tool = payload.get("tool") or payload.get("name") or payload.get("tool_name") or "unknown"
    args = payload.get("arguments")
    if args is None:
        args = payload.get("input", payload.get("args", payload.get("arguments", {})))
    result = payload.get("result", payload.get("output"))
    success = payload.get("success")
    if success is None:
        success = not bool(payload.get("error") or payload.get("is_error"))
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool,
        "arguments": args,
        "result_summary": _summarize_result(result),
        "success": bool(success),
    }
    if payload.get("error"):
        event["error"] = payload.get("error")
    return event


def main():
    payload = _load_payload()
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = _build_event(payload)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    script = script.replace("__EVENTS_PATH__", json.dumps(str(events_path)))
    output_path.write_text(script, encoding="utf-8")


def generate_notification_script(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = """#!/usr/bin/env python3
import json
import platform
import subprocess
import sys


def _load_payload():
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _notification_content(payload):
    title = payload.get("title") or payload.get("heading") or "Ralph notification"
    message = payload.get("message") or payload.get("content") or payload.get("text") or ""
    return str(title), str(message)


def _run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def _notify_macos(title, message):
    script = "display notification " + json.dumps(message) + " with title " + json.dumps(title)
    return _run_command(["osascript", "-e", script])


def _notify_linux(title, message):
    return _run_command(["notify-send", title, message])


def main():
    payload = _load_payload()
    title, message = _notification_content(payload)
    system = platform.system()
    if system == "Darwin":
        return 0 if _notify_macos(title, message) else 1
    if system == "Linux":
        return 0 if _notify_linux(title, message) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    output_path.write_text(script, encoding="utf-8")


def _load_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_settings(settings_path: Path, settings: dict[str, Any]) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _ensure_hook_list(settings: dict[str, Any], hook_name: str) -> list[dict[str, Any]]:
    hooks = settings.setdefault("hooks", {})
    hook_list = hooks.get(hook_name)
    if not isinstance(hook_list, list):
        hook_list = []
        hooks[hook_name] = hook_list
    return hook_list


def _upsert_hook_entry(hook_list: list[dict[str, Any]], entry: dict[str, Any], name: str) -> bool:
    updated = False
    for index, existing in enumerate(hook_list):
        if not isinstance(existing, dict):
            continue
        if existing.get("name") == name:
            hook_list[index] = entry
            updated = True
            break
    if not updated:
        hook_list.append(entry)
    return True


def install_hooks(
    settings_path: Path,
    pre_command: str,
    post_command: str,
    notification_command: str | None = None,
) -> bool:
    settings = _load_settings(settings_path)
    pre_entry = {
        "name": RALPH_PRE_HOOK_NAME,
        "type": "command",
        "command": pre_command,
    }
    post_entry = {
        "name": RALPH_POST_HOOK_NAME,
        "type": "command",
        "command": post_command,
    }
    pre_list = _ensure_hook_list(settings, "PreToolUse")
    post_list = _ensure_hook_list(settings, "PostToolUse")
    _upsert_hook_entry(pre_list, pre_entry, RALPH_PRE_HOOK_NAME)
    _upsert_hook_entry(post_list, post_entry, RALPH_POST_HOOK_NAME)
    if notification_command:
        notification_entry = {
            "name": RALPH_NOTIFICATION_HOOK_NAME,
            "type": "command",
            "command": notification_command,
        }
        notification_list = _ensure_hook_list(settings, "Notification")
        _upsert_hook_entry(notification_list, notification_entry, RALPH_NOTIFICATION_HOOK_NAME)
    _write_settings(settings_path, settings)
    return True


def uninstall_hooks(settings_path: Path) -> bool:
    settings = _load_settings(settings_path)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    changed = False
    for hook_name, target in [
        ("PreToolUse", RALPH_PRE_HOOK_NAME),
        ("PostToolUse", RALPH_POST_HOOK_NAME),
        ("Notification", RALPH_NOTIFICATION_HOOK_NAME),
    ]:
        hook_list = hooks.get(hook_name)
        if not isinstance(hook_list, list):
            continue
        original = len(hook_list)
        hook_list[:] = [
            item
            for item in hook_list
            if not (isinstance(item, dict) and item.get("name") == target)
        ]
        if len(hook_list) != original:
            changed = True
    if changed:
        _write_settings(settings_path, settings)
    return changed
