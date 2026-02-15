from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass

from agent_recall.storage.models import RalphNotificationEvent


@dataclass(frozen=True)
class NotificationEventInfo:
    title: str
    message: str


def _supports_event(
    event: RalphNotificationEvent, enabled_events: Iterable[RalphNotificationEvent]
) -> bool:
    return event in set(enabled_events)


def build_notification_content(
    event: RalphNotificationEvent, *, iteration: int | None = None
) -> NotificationEventInfo:
    if event == RalphNotificationEvent.ITERATION_COMPLETE:
        title = "Ralph iteration complete"
        message = "Iteration completed."
        if iteration is not None:
            message = f"Iteration {iteration} completed."
        return NotificationEventInfo(title=title, message=message)
    if event == RalphNotificationEvent.VALIDATION_FAILED:
        title = "Ralph validation failed"
        message = "Validation failed."
        if iteration is not None:
            message = f"Iteration {iteration} validation failed."
        return NotificationEventInfo(title=title, message=message)
    if event == RalphNotificationEvent.BUDGET_EXCEEDED:
        return NotificationEventInfo(
            title="Ralph budget exceeded",
            message="Budget limit exceeded. Ralph loop paused.",
        )
    return NotificationEventInfo(
        title="Ralph loop finished",
        message="Ralph loop finished.",
    )


def dispatch_notification(
    event: RalphNotificationEvent,
    *,
    enabled: bool,
    enabled_events: Iterable[RalphNotificationEvent],
    iteration: int | None = None,
) -> bool:
    if not enabled or not _supports_event(event, enabled_events):
        return False

    info = build_notification_content(event, iteration=iteration)
    return _dispatch_payload(info)


def dispatch_claude_notification(payload: dict[str, object]) -> bool:
    info = _build_info_from_payload(payload)
    return _dispatch_payload(info)


def _build_info_from_payload(payload: dict[str, object]) -> NotificationEventInfo:
    raw_message = payload.get("message") or payload.get("content") or payload.get("text")
    raw_title = payload.get("title") or payload.get("heading") or "Ralph notification"
    title = str(raw_title) if raw_title is not None else "Ralph notification"
    message = ""
    if raw_message is not None:
        message = str(raw_message)
    return NotificationEventInfo(title=title, message=message)


def _dispatch_payload(info: NotificationEventInfo) -> bool:
    system = platform.system()
    if system == "Darwin":
        return _notify_macos(info)
    if system == "Linux":
        return _notify_linux(info)
    return False


def _notify_macos(info: NotificationEventInfo) -> bool:
    script = f"display notification {json.dumps(info.message)} with title {json.dumps(info.title)}"
    return _run_command(["osascript", "-e", script])


def _notify_linux(info: NotificationEventInfo) -> bool:
    return _run_command(["notify-send", info.title, info.message])


def _run_command(cmd: list[str]) -> bool:
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
