"""Structured tier-file writing with policies, duplicate guards, and validation.

This module provides:
- Canonical schemas for GUARDRAILS.md, STYLE.md, and RECENT.md
- Write policies with section-targeted updates
- Duplicate detection and guards
- Bounded append behavior
- Validation and lint checks
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_recall.storage.files import FileStorage, KnowledgeTier


class WriteMode(StrEnum):
    """Write modes for tier file updates."""

    APPEND = "append"
    REPLACE_SECTION = "replace-section"


class TierValidationError(Exception):
    """Raised when tier file validation fails."""

    pass


@dataclass(frozen=True)
class SectionSchema:
    """Schema for a tier file section."""

    name: str
    pattern: re.Pattern[str]
    required: bool = False
    max_entries: int | None = None


@dataclass
class WritePolicy:
    """Policy configuration for tier file writes."""

    mode: WriteMode = WriteMode.APPEND
    deduplicate: bool = True
    max_entries: int | None = None
    section_target: str | None = None


# Canonical section schemas for each tier
GUARDRAILS_SCHEMA = [
    SectionSchema(
        name="header",
        pattern=re.compile(r"^#\s+Guardrails", re.IGNORECASE),
        required=True,
    ),
    SectionSchema(
        name="hard_failure",
        pattern=re.compile(r"^##\s+.*HARD FAILURE.*", re.IGNORECASE),
        max_entries=50,
    ),
    SectionSchema(
        name="general",
        pattern=re.compile(r"^##\s+.*", re.IGNORECASE),
        max_entries=100,
    ),
]

STYLE_SCHEMA = [
    SectionSchema(
        name="header",
        pattern=re.compile(r"^#\s+Style", re.IGNORECASE),
        required=True,
    ),
    SectionSchema(
        name="iteration",
        pattern=re.compile(r"^##\s+.*Iteration.*", re.IGNORECASE),
        max_entries=100,
    ),
    SectionSchema(
        name="general",
        pattern=re.compile(r"^##\s+.*", re.IGNORECASE),
        max_entries=100,
    ),
]

RECENT_SCHEMA = [
    SectionSchema(
        name="header",
        pattern=re.compile(r"^#\s+Recent", re.IGNORECASE),
        required=True,
    ),
    SectionSchema(
        name="iteration",
        pattern=re.compile(r"^##\s+.*Iteration.*", re.IGNORECASE),
        max_entries=50,
    ),
]

TIER_SCHEMAS: dict[KnowledgeTier, list[SectionSchema]] = {
    KnowledgeTier.GUARDRAILS: GUARDRAILS_SCHEMA,
    KnowledgeTier.STYLE: STYLE_SCHEMA,
    KnowledgeTier.RECENT: RECENT_SCHEMA,
}

# Default headers for tier files
TIER_HEADERS: dict[KnowledgeTier, str] = {
    KnowledgeTier.GUARDRAILS: "# Guardrails\n\nRules and warnings learned during development.\n",
    KnowledgeTier.STYLE: "# Style\n\nPatterns and preferences learned during development.\n",
    KnowledgeTier.RECENT: "# Recent\n\nRecent development activity summaries.\n",
}

# Content hash normalization for deduplication
_DUPLICATE_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for duplicate detection."""
    return _DUPLICATE_RE.sub(" ", text.strip().lower())


def _compute_content_hash(text: str) -> str:
    """Compute a hash for content deduplication."""
    normalized = _normalize_for_dedup(text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class TierEntry:
    """A parsed entry from a tier file."""

    section: str
    content: str
    line_start: int
    line_end: int
    timestamp: datetime | None = None


class TierWriter:
    """Writer for structured tier file updates with policies and validation."""

    def __init__(
        self,
        files: FileStorage,
        policy: WritePolicy | None = None,
    ):
        self.files = files
        self.policy = policy or WritePolicy()

    def write_guardrails_entry(
        self,
        iteration: int,
        item_id: str,
        item_title: str,
        reason: str,
        validation_hint: str = "",
        timestamp: datetime | None = None,
    ) -> bool:
        """Write a guardrails entry following the canonical schema.

        Returns True if content was written, False if skipped (duplicate or policy violation).
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "",
            f"## {ts_str} Iteration {iteration} ({item_id})",
            f"- Scope item: {item_title}",
        ]

        if reason == "validation_failed":
            lines.append("- Do not move to a new PRD item while validation is red.")
        elif reason == "agent_timeout":
            lines.append(
                "- Agent exceeded iteration timeout; reduce scope and keep commits smaller."
            )
        elif reason == "abort":
            lines.append("- Abort means scope exceeded safety; reduce change size next iteration.")
        else:
            lines.append("- Keep changes isolated and verifiable before commit.")

        if validation_hint:
            lines.append(f"- Runtime validation signal: {validation_hint}")

        lines.append(
            f"- Runtime logs: agent_recall/ralph/.runtime/agent-{iteration}.log, "
            f"agent_recall/ralph/.runtime/validate-{iteration}.log"
        )

        content = "\n".join(lines)

        return self._write_with_policy(
            tier=KnowledgeTier.GUARDRAILS,
            content=content,
            section_hint="general",
        )

    def write_guardrails_hard_failure(
        self,
        iteration: int,
        item_id: str,
        item_title: str,
        validation_errors: list[str],
        validation_hint: str = "",
        timestamp: datetime | None = None,
    ) -> bool:
        """Write a hard failure guardrails entry.

        Returns True if content was written, False if skipped.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "",
            f"## {ts_str} HARD FAILURE Iteration {iteration} ({item_id})",
            f"- Item: {item_title}",
            "- Validation command: uv run pytest && uv run ruff check . && uv run ty check",
        ]

        if validation_errors:
            lines.append("- Top validation errors:")
            for error in validation_errors[:6]:
                lines.append(f"  - {error}")
        else:
            lines.append("- Validation failed without captured output.")

        if validation_hint:
            lines.append(f"- Primary actionable signal: {validation_hint}")

        lines.append(
            f"- Runtime logs: agent_recall/ralph/.runtime/agent-{iteration}.log, "
            f"agent_recall/ralph/.runtime/validate-{iteration}.log"
        )

        content = "\n".join(lines)

        return self._write_with_policy(
            tier=KnowledgeTier.GUARDRAILS,
            content=content,
            section_hint="hard_failure",
        )

    def write_style_entry(
        self,
        iteration: int,
        item_id: str,
        validation_hint: str = "",
        timestamp: datetime | None = None,
    ) -> bool:
        """Write a style entry following the canonical schema.

        Returns True if content was written, False if skipped.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "",
            f"## {ts_str} Iteration {iteration} ({item_id})",
            "- Prefer one logical change per commit.",
            "- Keep validation command green before committing: "
            "uv run pytest && uv run ruff check . && uv run ty check",
        ]

        if validation_hint:
            lines.append(
                f"- Start debugging from the first actionable validation line: {validation_hint}"
            )
        else:
            lines.append(
                "- Keep runtime validate logs concise so the first actionable line is obvious."
            )

        lines.append(
            f"- Runtime logs: agent_recall/ralph/.runtime/agent-{iteration}.log, "
            f"agent_recall/ralph/.runtime/validate-{iteration}.log"
        )

        content = "\n".join(lines)

        return self._write_with_policy(
            tier=KnowledgeTier.STYLE,
            content=content,
            section_hint="iteration",
        )

    def write_recent_entry(
        self,
        iteration: int,
        item_id: str,
        item_title: str,
        work_mode: str,
        agent_exit: int,
        validate_status: str,
        outcome: str,
        validation_hint: str = "",
        timestamp: datetime | None = None,
    ) -> bool:
        """Write a recent entry following the canonical schema.

        Returns True if content was written, False if skipped.
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "",
            f"## {ts_str} Iteration {iteration}",
            f"- Item: {item_id} - {item_title}",
            f"- Mode: {work_mode}",
            f"- Agent exit code: {agent_exit}",
            f"- Validation: {validate_status}",
            f"- Outcome: {outcome}",
        ]

        if validation_hint:
            lines.append(f"- Validation signal: {validation_hint}")

        lines.append(
            f"- Runtime logs: agent_recall/ralph/.runtime/agent-{iteration}.log, "
            f"agent_recall/ralph/.runtime/validate-{iteration}.log"
        )

        content = "\n".join(lines)

        return self._write_with_policy(
            tier=KnowledgeTier.RECENT,
            content=content,
            section_hint="iteration",
        )

    def _ensure_header(self, tier: KnowledgeTier, content: str) -> str:
        """Ensure tier content has proper header if empty or missing."""
        if not content.strip():
            return TIER_HEADERS[tier]

        # Check if header exists
        header_pattern = TIER_HEADERS[tier].split("\n")[0]
        if not content.strip().startswith(header_pattern):
            return TIER_HEADERS[tier] + "\n" + content

        return content

    def _write_with_policy(
        self,
        tier: KnowledgeTier,
        content: str,
        section_hint: str | None = None,
    ) -> bool:
        """Write content to tier following the configured policy.

        Returns True if written, False if skipped.
        """
        current = self.files.read_tier(tier)

        # Ensure header exists
        current = self._ensure_header(tier, current)

        # Check for duplicates if enabled
        if self.policy.deduplicate and self._is_duplicate(current, content):
            return False

        # Apply write mode
        if self.policy.mode == WriteMode.REPLACE_SECTION and section_hint:
            updated = self._replace_section(current, content, section_hint)
        else:
            # Default: append with bounds checking
            updated = self._bounded_append(
                current,
                content,
                tier,
                self.policy.max_entries,
            )

        # Validate before writing
        validation_errors = self.validate_tier_content(tier, updated)
        if validation_errors:
            raise TierValidationError(
                f"Validation failed for {tier.value}: {', '.join(validation_errors)}"
            )

        self.files.write_tier(tier, updated)
        return True

    def _is_duplicate(self, current: str, new_content: str) -> bool:
        """Check if content is a duplicate of existing content.

        Duplicates are detected by:
        1. Exact hash match of the entire entry
        2. Same iteration number and item ID (for Ralph loop entries)
        """
        # Check for exact hash match
        new_hash = _compute_content_hash(new_content)

        # Extract iteration number and item ID from new content
        new_iteration = None
        new_item_id = None
        for line in new_content.split("\n"):
            if line.startswith("## "):
                # Parse: ## TIMESTAMP Iteration N (ITEM-ID)
                match = re.search(r"Iteration\s+(\d+)\s+\(([^)]+)\)", line)
                if match:
                    new_iteration = match.group(1)
                    new_item_id = match.group(2)
                break

        # Parse existing entries
        for line in current.split("\n"):
            # Check hash match
            if _compute_content_hash(line) == new_hash:
                return True

            # Check iteration/item_id match for Ralph loop entries
            if new_iteration and new_item_id and line.startswith("## "):
                match = re.search(r"Iteration\s+(\d+)\s+\(([^)]+)\)", line)
                if match:
                    existing_iteration = match.group(1)
                    existing_item_id = match.group(2)
                    if existing_iteration == new_iteration and existing_item_id == new_item_id:
                        return True

        return False

    def _replace_section(self, current: str, new_content: str, section_hint: str) -> str:
        """Replace a specific section in the tier file."""
        lines = current.split("\n")
        section_start = None
        section_end = None

        # Find section start
        for i, line in enumerate(lines):
            if re.match(rf"^##\s+.*{section_hint}.*", line, re.IGNORECASE):
                section_start = i
                break

        if section_start is None:
            # Section not found, append
            return current + "\n" + new_content

        # Find section end (start of next section or end of file)
        for i in range(section_start + 1, len(lines)):
            if lines[i].startswith("## "):
                section_end = i
                break

        if section_end is None:
            section_end = len(lines)

        # Replace section
        new_lines = lines[:section_start] + new_content.split("\n") + lines[section_end:]
        return "\n".join(new_lines)

    def _bounded_append(
        self,
        current: str,
        new_content: str,
        tier: KnowledgeTier,
        max_entries: int | None,
    ) -> str:
        """Append content with bounds checking to prevent unbounded growth."""
        if max_entries is None:
            # Use tier-specific defaults
            schema = TIER_SCHEMAS.get(tier, [])
            for section in schema:
                if section.max_entries:
                    max_entries = section.max_entries
                    break
            else:
                max_entries = 100  # Global default

        # Count current entries (lines starting with "## ")
        lines = current.split("\n")
        entry_count = sum(1 for line in lines if line.startswith("## "))

        if entry_count >= max_entries:
            # Remove oldest entries to make room
            entries_to_remove = entry_count - max_entries + 1
            new_lines = []
            removed = 0
            skip_until_next_header = False

            for line in lines:
                if line.startswith("## "):
                    if removed < entries_to_remove:
                        removed += 1
                        skip_until_next_header = True
                        continue
                    else:
                        skip_until_next_header = False

                if not skip_until_next_header:
                    new_lines.append(line)

            lines = new_lines

        return "\n".join(lines) + "\n" + new_content

    def validate_tier_content(
        self,
        tier: KnowledgeTier,
        content: str,
    ) -> list[str]:
        """Validate tier content against its schema.

        Returns list of validation error messages (empty if valid).
        """
        errors: list[str] = []
        schema = TIER_SCHEMAS.get(tier, [])

        if not schema:
            return errors

        lines = content.split("\n")

        # Check required sections
        for section in schema:
            if section.required:
                found = any(section.pattern.match(line) for line in lines)
                if not found:
                    errors.append(f"Missing required section: {section.name}")

        # Check for malformed sections
        for i, line in enumerate(lines):
            if line.startswith("## ") and not re.match(r"^##\s+\S", line):
                errors.append(f"Line {i + 1}: Malformed section header (missing content)")

        # Check entry count bounds
        entry_count = sum(1 for line in lines if line.startswith("## "))
        for section in schema:
            if section.max_entries and entry_count > section.max_entries:
                errors.append(
                    f"Entry count ({entry_count}) exceeds maximum ({section.max_entries}) "
                    f"for section {section.name}"
                )
                break

        return errors


def lint_tier_file(
    tier: KnowledgeTier,
    content: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Lint a tier file and return (errors, warnings).

    Args:
        tier: The tier type being linted
        content: The content to lint
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (errors, warnings) lists
    """
    errors: list[str] = []
    warnings: list[str] = []

    lines = content.split("\n")

    # Check for empty content
    if not content.strip():
        errors.append("Tier file is empty")
        return errors, warnings

    # Check for required header
    header_pattern = re.compile(r"^#\s+\w+", re.IGNORECASE)
    if not any(header_pattern.match(line) for line in lines):
        errors.append("Missing required header (e.g., '# Guardrails')")

    # Check for low-signal entries (very short)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            # Look ahead for content
            content_length = 0
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].startswith("## "):
                    break
                content_length += len(lines[j].strip())

            if content_length < 20:
                warnings.append(f"Line {i + 1}: Low-signal entry (very short content)")

    # Check for malformed timestamps
    timestamp_pattern = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
    for i, line in enumerate(lines):
        match = timestamp_pattern.match(line)
        if match:
            ts_str = match.group(1)
            try:
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                warnings.append(f"Line {i + 1}: Malformed timestamp: {ts_str}")

    # Check for excessive duplication
    writer = TierWriter(FileStorage(Path(".")))
    validation_errors = writer.validate_tier_content(tier, content)
    for error in validation_errors:
        if "Duplicate" in error:
            errors.append(error)
        elif strict:
            errors.append(error)
        else:
            warnings.append(error)

    # In strict mode, convert all warnings to errors
    if strict and warnings:
        errors.extend(warnings)
        warnings = []

    return errors, warnings


def get_tier_statistics(content: str) -> dict[str, Any]:
    """Get statistics about a tier file's content."""
    lines = content.split("\n")

    entries = [line for line in lines if line.startswith("## ")]
    entry_count = len(entries)

    # Calculate total content size (excluding headers)
    content_lines = [line for line in lines if line.strip() and not line.startswith("#")]
    content_size = sum(len(line) for line in content_lines)

    # Find date range if timestamps present
    timestamps: list[datetime] = []
    timestamp_pattern = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")
    for line in lines:
        match = timestamp_pattern.match(line)
        if match:
            try:
                ts = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
                timestamps.append(ts)
            except ValueError:
                pass

    return {
        "entry_count": entry_count,
        "content_size": content_size,
        "line_count": len(lines),
        "date_range": {
            "earliest": min(timestamps).isoformat() if timestamps else None,
            "latest": max(timestamps).isoformat() if timestamps else None,
        },
    }
