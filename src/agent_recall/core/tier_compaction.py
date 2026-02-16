"""Tier compaction hook for normalizing, deduplicating, and summarizing tier files.

This module provides:
- Post-loop tier compaction to keep GUARDRAILS/STYLE/RECENT files concise
- Size budget enforcement
- Duplicate detection and removal
- Entry summarization when content grows too large
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_recall.storage.files import FileStorage, KnowledgeTier


@dataclass
class TierCompactionResult:
    """Result of tier compaction for a single tier."""

    tier: KnowledgeTier
    entries_before: int
    entries_after: int
    bytes_before: int
    bytes_after: int
    duplicates_removed: int
    entries_summarized: int


@dataclass
class TierCompactionSummary:
    """Summary of tier compaction across all tiers."""

    results: list[TierCompactionResult]
    auto_run: bool = False

    @property
    def total_entries_before(self) -> int:
        return sum(r.entries_before for r in self.results)

    @property
    def total_entries_after(self) -> int:
        return sum(r.entries_after for r in self.results)

    @property
    def total_bytes_before(self) -> int:
        return sum(r.bytes_before for r in self.results)

    @property
    def total_bytes_after(self) -> int:
        return sum(r.bytes_after for r in self.results)

    @property
    def total_duplicates_removed(self) -> int:
        return sum(r.duplicates_removed for r in self.results)

    @property
    def total_entries_summarized(self) -> int:
        return sum(r.entries_summarized for r in self.results)


class TierCompactionConfig:
    """Configuration for tier compaction."""

    def __init__(
        self,
        auto_run: bool = True,
        max_entries_per_tier: int = 50,
        strict_deduplication: bool = False,
        summary_threshold_entries: int = 40,
        summary_max_entries: int = 20,
    ):
        self.auto_run = auto_run
        self.max_entries_per_tier = max_entries_per_tier
        self.strict_deduplication = strict_deduplication
        self.summary_threshold_entries = summary_threshold_entries
        self.summary_max_entries = summary_max_entries

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TierCompactionConfig:
        """Create config from parsed YAML config dict."""
        tier_cfg = config.get("tier_compaction", {})
        if not isinstance(tier_cfg, dict):
            tier_cfg = {}

        return cls(
            auto_run=bool(tier_cfg.get("auto_run", True)),
            max_entries_per_tier=int(tier_cfg.get("max_entries_per_tier", 50)),
            strict_deduplication=bool(tier_cfg.get("strict_deduplication", False)),
            summary_threshold_entries=int(tier_cfg.get("summary_threshold_entries", 40)),
            summary_max_entries=int(tier_cfg.get("summary_max_entries", 20)),
        )


# Entry header pattern: ## TIMESTAMP Iteration N (ITEM-ID)
_ENTRY_HEADER_RE = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+Iteration\s+(\d+)\s+\(([^)]+)\)"
)


class TierCompactionHook:
    """Hook for compacting tier files after Ralph loop iterations."""

    def __init__(self, files: FileStorage, config: TierCompactionConfig | None = None):
        self.files = files
        self.config = config or TierCompactionConfig()

    def compact_all(self) -> TierCompactionSummary:
        """Compact all tier files and return summary."""
        results: list[TierCompactionResult] = []

        for tier in [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]:
            result = self.compact_tier(tier)
            results.append(result)

        return TierCompactionSummary(results=results, auto_run=self.config.auto_run)

    def compact_tier(self, tier: KnowledgeTier) -> TierCompactionResult:
        """Compact a single tier file and return result."""
        return self._compact_tier(tier)

    def _compact_tier(self, tier: KnowledgeTier) -> TierCompactionResult:
        """Compact a single tier file."""
        content = self.files.read_tier(tier)
        bytes_before = len(content.encode("utf-8"))

        if not content.strip():
            return TierCompactionResult(
                tier=tier,
                entries_before=0,
                entries_after=0,
                bytes_before=bytes_before,
                bytes_after=0,
                duplicates_removed=0,
                entries_summarized=0,
            )

        # Parse entries
        entries = self._parse_entries(content)
        entries_before = len(entries)

        # Remove duplicates
        entries, duplicates_removed = self._remove_duplicates(entries)

        # Apply size budget (remove oldest if over limit)
        entries, entries_dropped = self._apply_size_budget(entries)

        # Summarize if still over threshold
        entries, entries_summarized = self._maybe_summarize(entries)

        # Reconstruct content
        new_content = self._reconstruct_content(tier, entries)
        bytes_after = len(new_content.encode("utf-8"))

        # Write back if changed
        if new_content != content:
            self.files.write_tier(tier, new_content)

        return TierCompactionResult(
            tier=tier,
            entries_before=entries_before,
            entries_after=len(entries),
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            duplicates_removed=duplicates_removed,
            entries_summarized=entries_summarized,
        )

    def _parse_entries(self, content: str) -> list[dict[str, Any]]:
        """Parse content into structured entries."""
        entries: list[dict[str, Any]] = []
        lines = content.split("\n")

        current_entry: dict[str, Any] | None = None
        current_content_lines: list[str] = []

        for line in lines:
            match = _ENTRY_HEADER_RE.match(line)
            if match:
                # Save previous entry
                if current_entry is not None:
                    current_entry["content"] = "\n".join(current_content_lines)
                    entries.append(current_entry)

                # Start new entry
                timestamp = match.group(1)
                iteration = int(match.group(2))
                item_id = match.group(3)
                current_entry = {
                    "timestamp": timestamp,
                    "iteration": iteration,
                    "item_id": item_id,
                    "header_line": line,
                    "content_lines": [],
                }
                current_content_lines = []
            elif current_entry is not None:
                current_content_lines.append(line)

        # Save final entry
        if current_entry is not None:
            current_entry["content"] = "\n".join(current_content_lines)
            entries.append(current_entry)

        return entries

    def _remove_duplicates(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Remove duplicate entries based on iteration + item_id."""
        seen_keys: set[str] = set()
        unique_entries: list[dict[str, Any]] = []
        duplicates_removed = 0

        for entry in entries:
            # Use iteration + item_id as dedup key (keep newest/timestamp-based)
            key = f"{entry['iteration']}:{entry['item_id']}"

            if self.config.strict_deduplication:
                # Also consider content hash
                content_hash = hash(entry["content"])
                key = f"{key}:{content_hash}"

            if key in seen_keys:
                duplicates_removed += 1
                continue

            seen_keys.add(key)
            unique_entries.append(entry)

        return unique_entries, duplicates_removed

    def _apply_size_budget(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Apply size budget by removing oldest entries if over limit."""
        max_entries = self.config.max_entries_per_tier

        if len(entries) <= max_entries:
            return entries, 0

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e["timestamp"])

        # Keep only the most recent max_entries
        entries_to_keep = sorted_entries[-max_entries:]
        entries_dropped = len(entries) - len(entries_to_keep)

        # Re-sort to maintain original order (newest first)
        entries_to_keep.sort(key=lambda e: e["timestamp"], reverse=True)

        return entries_to_keep, entries_dropped

    def _maybe_summarize(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Summarize entries if still over threshold."""
        threshold = self.config.summary_threshold_entries
        max_summary_entries = self.config.summary_max_entries

        if len(entries) <= threshold:
            return entries, 0

        # Group entries by item_id for summarization
        by_item: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            item_id = entry["item_id"]
            if item_id not in by_item:
                by_item[item_id] = []
            by_item[item_id].append(entry)

        # Create summary entries for items with multiple entries
        summarized_entries: list[dict[str, Any]] = []
        entries_summarized = 0

        for item_id, item_entries in by_item.items():
            if len(item_entries) > 1 and len(summarized_entries) < max_summary_entries:
                # Create a summary entry
                latest = max(item_entries, key=lambda e: e["timestamp"])
                summary_entry = {
                    "timestamp": latest["timestamp"],
                    "iteration": latest["iteration"],
                    "item_id": item_id,
                    "header_line": latest["header_line"],
                    "content": f"(Summarized {len(item_entries)} entries for this item)",
                    "is_summary": True,
                    "summarized_count": len(item_entries),
                }
                summarized_entries.append(summary_entry)
                entries_summarized += len(item_entries) - 1  # Count merged entries
            else:
                # Keep individual entries
                summarized_entries.extend(item_entries)

        # Re-sort by timestamp (newest first)
        summarized_entries.sort(key=lambda e: e["timestamp"], reverse=True)

        # Apply final size limit
        if len(summarized_entries) > self.config.max_entries_per_tier:
            summarized_entries = summarized_entries[: self.config.max_entries_per_tier]

        return summarized_entries, entries_summarized

    def _reconstruct_content(self, tier: KnowledgeTier, entries: list[dict[str, Any]]) -> str:
        """Reconstruct tier file content from entries."""
        # Get appropriate header
        headers = {
            KnowledgeTier.GUARDRAILS: (
                "# Guardrails\n\nRules and warnings learned during development.\n"
            ),
            KnowledgeTier.STYLE: (
                "# Style\n\nPatterns and preferences learned during development.\n"
            ),
            KnowledgeTier.RECENT: "# Recent\n\nRecent development activity summaries.\n",
        }

        lines = [headers[tier]]

        for entry in entries:
            lines.append(entry["header_line"])
            if entry.get("is_summary"):
                lines.append(f"- {entry['content']}")
            else:
                lines.append(entry["content"])

        return "\n".join(lines) + "\n"


def format_compaction_summary(summary: TierCompactionSummary) -> str:
    """Format a compaction summary for display."""
    lines = ["Tier Compaction Summary", "=" * 40]

    for result in summary.results:
        tier_name = result.tier.value.upper()
        lines.append(f"\n{tier_name}:")
        lines.append(f"  Entries: {result.entries_before} → {result.entries_after}")
        lines.append(f"  Size: {result.bytes_before} → {result.bytes_after} bytes")
        if result.duplicates_removed:
            lines.append(f"  Duplicates removed: {result.duplicates_removed}")
        if result.entries_summarized:
            lines.append(f"  Entries summarized: {result.entries_summarized}")

    lines.append(f"\n{'=' * 40}")
    lines.append(f"Total entries: {summary.total_entries_before} → {summary.total_entries_after}")
    lines.append(f"Total size: {summary.total_bytes_before} → {summary.total_bytes_after} bytes")

    if summary.total_duplicates_removed:
        lines.append(f"Total duplicates removed: {summary.total_duplicates_removed}")
    if summary.total_entries_summarized:
        lines.append(f"Total entries summarized: {summary.total_entries_summarized}")

    return "\n".join(lines)


def estimate_token_count(content: str) -> int:
    """Estimate token count from text length (chars/4 heuristic)."""
    if not content:
        return 0
    return (len(content) + 3) // 4


def should_compact_for_tokens(content: str, max_tokens: int) -> bool:
    """Return True if content exceeds token threshold."""
    if max_tokens <= 0:
        return False
    return estimate_token_count(content) > max_tokens


def _resolve_max_tier_tokens(config: dict[str, Any]) -> int:
    compaction_cfg = config.get("compaction", {}) if isinstance(config, dict) else {}
    if not isinstance(compaction_cfg, dict):
        compaction_cfg = {}
    raw_value = compaction_cfg.get("max_tier_tokens", 10000)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = 10000
    return max(parsed, 0)


def compact_if_over_tokens(
    *,
    files: FileStorage,
    tier: KnowledgeTier,
    content: str,
    max_tokens: int | None = None,
) -> bool:
    """Run tier compaction if content exceeds token threshold."""
    config = files.read_config()
    effective_max = max_tokens if max_tokens is not None else _resolve_max_tier_tokens(config)
    if not should_compact_for_tokens(content, effective_max):
        return False
    hook = TierCompactionHook(files, TierCompactionConfig.from_config(config))
    hook.compact_tier(tier)
    return True
