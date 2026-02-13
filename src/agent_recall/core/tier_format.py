"""Tier file format detection helpers for Ralph and bullet formats."""

from __future__ import annotations

import re
from enum import StrEnum


class EntryFormat(StrEnum):
    BULLET = "bullet"
    RALPH = "ralph"
    RECENT_BULLET = "recent_bullet"
    PREAMBLE = "preamble"
    UNKNOWN = "unknown"


BULLET_RE = re.compile(r"^\s*-\s*\[(?P<kind>[A-Z_]+)\]\s*(?P<text>.+?)\s*$")
RALPH_ENTRY_RE = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?:HARD FAILURE\s+)?"
    r"Iteration\s+(\d+)\s+\(([^)]+)\)"
)
RECENT_BULLET_RE = re.compile(r"^\s*\*\*(?P<date>\d{4}-\d{2}-\d{2})\*\*:\s*(?P<summary>.+)\s*$")


def detect_line_format(line: str) -> EntryFormat:
    cleaned = line.rstrip()
    if RALPH_ENTRY_RE.match(cleaned):
        return EntryFormat.RALPH
    if BULLET_RE.match(cleaned):
        return EntryFormat.BULLET
    if RECENT_BULLET_RE.match(cleaned):
        return EntryFormat.RECENT_BULLET
    return EntryFormat.UNKNOWN


def is_ralph_entry_start(line: str) -> bool:
    return bool(RALPH_ENTRY_RE.match(line))


def is_bullet_entry(line: str) -> bool:
    return bool(BULLET_RE.match(line))


detect_entry_format = detect_line_format
is_ralph_entry_line = is_ralph_entry_start
is_bullet_entry_line = is_bullet_entry
