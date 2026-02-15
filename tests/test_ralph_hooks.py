from __future__ import annotations

import json
from pathlib import Path

from agent_recall.ralph.hooks import (
    RALPH_NOTIFICATION_HOOK_NAME,
    RALPH_POST_HOOK_NAME,
    RALPH_PRE_HOOK_NAME,
    append_tool_event,
    build_guardrail_patterns,
    build_hook_command,
    build_tool_event,
    generate_notification_script,
    generate_post_tool_script,
    generate_pre_tool_script,
    install_hooks,
    should_block_payload,
    uninstall_hooks,
)


def test_build_guardrail_patterns_defaults_and_dedup() -> None:
    guardrails = "- Blocked: `rm -rf /`\n- Other: `rm -rf /`\n"
    patterns = build_guardrail_patterns(guardrails)
    assert "rm -rf /" in patterns
    assert patterns.count("rm -rf /") == 1


def test_build_hook_command_uses_python_executable(tmp_path: Path) -> None:
    script_path = tmp_path / "hook.py"
    command = build_hook_command(script_path)
    assert str(script_path) in command


def test_should_block_payload_matches_patterns() -> None:
    payload = {"tool": "shell", "arguments": {"cmd": "rm -rf /"}}
    blocked, pattern = should_block_payload(payload, [r"rm\s+-rf\s+/"])
    assert blocked is True
    assert pattern == r"rm\s+-rf\s+/"


def test_generate_pre_tool_script_writes_patterns(tmp_path: Path) -> None:
    output = tmp_path / "pre.py"
    patterns = generate_pre_tool_script("- Blocked: `rm -rf /`", output)
    assert output.exists()
    contents = output.read_text(encoding="utf-8")
    assert "rm -rf /" in contents
    assert "PATTERNS" in contents
    assert "rm -rf /" in patterns


def test_generate_post_tool_script_writes_events_path(tmp_path: Path) -> None:
    output = tmp_path / "post.py"
    events = tmp_path / "events.jsonl"
    generate_post_tool_script(output, events)
    assert output.exists()
    assert str(events) in output.read_text(encoding="utf-8")


def test_generate_notification_script_writes_payload(tmp_path: Path) -> None:
    output = tmp_path / "notify.py"
    generate_notification_script(output)
    assert output.exists()
    contents = output.read_text(encoding="utf-8")
    assert "Ralph notification" in contents


def test_build_tool_event_summarizes_result() -> None:
    payload = {"tool": "test", "arguments": {"a": 1}, "result": "ok", "success": True}
    event = build_tool_event(payload)
    assert event["tool"] == "test"
    assert event["arguments"] == {"a": 1}
    assert event["result_summary"] == "ok"
    assert event["success"] is True


def test_append_tool_event_writes_jsonl(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    event = append_tool_event(events_path, {"tool": "test", "arguments": {"x": 2}})
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["tool"] == event["tool"]


def test_install_and_uninstall_hooks(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    install_hooks(
        settings_path,
        "python pre.py",
        "python post.py",
        "python notify.py",
    )
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_hooks = payload["hooks"]["PreToolUse"]
    post_hooks = payload["hooks"]["PostToolUse"]
    notification_hooks = payload["hooks"]["Notification"]
    assert any(entry.get("name") == RALPH_PRE_HOOK_NAME for entry in pre_hooks)
    assert any(entry.get("name") == RALPH_POST_HOOK_NAME for entry in post_hooks)
    assert any(entry.get("name") == RALPH_NOTIFICATION_HOOK_NAME for entry in notification_hooks)
    removed = uninstall_hooks(settings_path)
    assert removed is True
    updated = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_hooks = updated["hooks"]["PreToolUse"]
    post_hooks = updated["hooks"]["PostToolUse"]
    notification_hooks = updated["hooks"].get("Notification", [])
    assert not any(entry.get("name") == RALPH_PRE_HOOK_NAME for entry in pre_hooks)
    assert not any(entry.get("name") == RALPH_POST_HOOK_NAME for entry in post_hooks)
    assert not any(
        entry.get("name") == RALPH_NOTIFICATION_HOOK_NAME for entry in notification_hooks
    )


def test_install_hooks_is_idempotent(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    install_hooks(
        settings_path,
        "python pre.py",
        "python post.py",
        "python notify.py",
    )
    install_hooks(
        settings_path,
        "python pre.py",
        "python post.py",
        "python notify.py",
    )
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_hooks = payload["hooks"]["PreToolUse"]
    post_hooks = payload["hooks"]["PostToolUse"]
    notification_hooks = payload["hooks"]["Notification"]
    assert len([entry for entry in pre_hooks if entry.get("name") == RALPH_PRE_HOOK_NAME]) == 1
    assert len([entry for entry in post_hooks if entry.get("name") == RALPH_POST_HOOK_NAME]) == 1
    assert (
        len(
            [
                entry
                for entry in notification_hooks
                if entry.get("name") == RALPH_NOTIFICATION_HOOK_NAME
            ]
        )
        == 1
    )
