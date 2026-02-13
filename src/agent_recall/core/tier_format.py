"""Tier file format detection helpers for Ralph and bullet formats."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ParsedEntry:
    format: EntryFormat
    raw_content: str
    timestamp: str | None = None
    iteration: int | None = None
    item_id: str | None = None
    kind: str | None = None
    text: str | None = None


@dataclass
class TierContent:
    preamble: list[str] = field(default_factory=list)
    bullet_entries: list[ParsedEntry] = field(default_factory=list)
    ralph_entries: list[ParsedEntry] = field(default_factory=list)
    unknown_lines: list[str] = field(default_factory=list)


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


def parse_ralph_header(line: str) -> tuple[str, int, str] | None:
    match = RALPH_ENTRY_RE.match(line.rstrip())
    if not match:
        return None
    timestamp, iteration, item_id = match.groups()
    return timestamp, int(iteration), item_id


def parse_bullet_entry(line: str) -> tuple[str, str] | None:
    match = BULLET_RE.match(line.rstrip())
    if not match:
        return None
    return match.group("kind"), match.group("text")


def parse_tier_content(content: str) -> TierContent:
    tier_content = TierContent()
    in_ralph_block = False
    current_ralph_lines: list[str] = []
    current_ralph_header: tuple[str, int, str] | None = None
    current_ralph_empty_count = 0
    found_first_entry = False
    had_header_termination = False

    def finalize_ralph_block() -> None:
        nonlocal current_ralph_lines
        nonlocal current_ralph_header
        nonlocal in_ralph_block
        nonlocal current_ralph_empty_count
        nonlocal had_header_termination
        if current_ralph_lines and current_ralph_header is not None:
            timestamp, iteration, item_id = current_ralph_header
            tier_content.ralph_entries.append(
                ParsedEntry(
                    format=EntryFormat.RALPH,
                    raw_content="\n".join(current_ralph_lines),
                    timestamp=timestamp,
                    iteration=iteration,
                    item_id=item_id,
                )
            )
        current_ralph_lines = []
        current_ralph_header = None
        current_ralph_empty_count = 0
        in_ralph_block = False
        had_header_termination = False

    for line in content.split("\n"):
        if in_ralph_block:
            if is_ralph_entry_start(line):
                finalize_ralph_block()
                header = parse_ralph_header(line)
                if header is not None:
                    current_ralph_header = header
                    current_ralph_lines = [line]
                    in_ralph_block = True
                    found_first_entry = True
                continue
            if line.startswith("## "):
                finalize_ralph_block()
                had_header_termination = True
            elif line.strip() == "":
                current_ralph_empty_count += 1
                if current_ralph_empty_count >= 2:
                    finalize_ralph_block()
                else:
                    current_ralph_lines.append(line)
                    continue
            else:
                current_ralph_empty_count = 0
                current_ralph_lines.append(line)
                continue

        if had_header_termination:
            had_header_termination = False
            if line.strip():
                if not found_first_entry:
                    tier_content.preamble.append(line)
                else:
                    tier_content.unknown_lines.append(line)
            continue

        if is_ralph_entry_start(line):
            header = parse_ralph_header(line)
            if header is not None:
                current_ralph_header = header
                current_ralph_lines = [line]
                in_ralph_block = True
                found_first_entry = True
            continue

        bullet = parse_bullet_entry(line)
        if bullet is not None:
            kind, text = bullet
            tier_content.bullet_entries.append(
                ParsedEntry(
                    format=EntryFormat.BULLET,
                    raw_content=line,
                    kind=kind,
                    text=text,
                )
            )
            found_first_entry = True
            continue

        recent_match = RECENT_BULLET_RE.match(line.rstrip())
        if recent_match is not None:
            tier_content.bullet_entries.append(
                ParsedEntry(
                    format=EntryFormat.RECENT_BULLET,
                    raw_content=line,
                    timestamp=recent_match.group("date"),
                    text=recent_match.group("summary"),
                )
            )
            found_first_entry = True
            continue

        if not found_first_entry:
            tier_content.preamble.append(line)
        elif line.strip():
            tier_content.unknown_lines.append(line)

    if current_ralph_lines:
        finalize_ralph_block()

    return tier_content


detect_entry_format = detect_line_format
is_ralph_entry_line = is_ralph_entry_start
is_bullet_entry_line = is_bullet_entry
