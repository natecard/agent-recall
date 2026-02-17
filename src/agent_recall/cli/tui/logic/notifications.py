from __future__ import annotations

from typing import Any

from agent_recall.storage.models import RalphNotificationEvent


def _read_notification_config(
    ralph_config: dict[str, Any],
) -> tuple[bool, list[RalphNotificationEvent]]:
    notifications = ralph_config.get("notifications")
    if not isinstance(notifications, dict):
        return False, []
    enabled = bool(notifications.get("enabled"))
    events_raw = notifications.get("events")
    if not isinstance(events_raw, list):
        return enabled, []
    parsed: list[RalphNotificationEvent] = []
    for value in events_raw:
        if isinstance(value, RalphNotificationEvent):
            parsed.append(value)
            continue
        if not isinstance(value, str):
            continue
        try:
            parsed.append(RalphNotificationEvent(value))
        except ValueError:
            continue
    return enabled, parsed
