from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_recall.core.tier_notes import normalize_tier_line
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import Chunk, SemanticLabel


class MemoryStore(ABC):
    """Backend-agnostic memory interface for tier and retrieval operations."""

    @abstractmethod
    def write_tier_note(self, *, tier: KnowledgeTier, note_line: str) -> bool:
        """Write a single note line to a tier representation."""
        ...

    @abstractmethod
    def read_tier(self, tier: KnowledgeTier) -> str:
        """Read raw tier content."""
        ...

    @abstractmethod
    def search(self, *, query: str, top_k: int = 5) -> list[Chunk]:
        """Search memory and return ranked chunks."""
        ...

    @abstractmethod
    def export_snapshot(self) -> dict[str, Any]:
        """Export memory payload for migration/inspection."""
        ...


class MarkdownMemoryStore(MemoryStore):
    """Markdown-first adapter preserving existing tier-file behavior."""

    def __init__(self, storage: Storage, files: FileStorage) -> None:
        self.storage = storage
        self.files = files

    def write_tier_note(self, *, tier: KnowledgeTier, note_line: str) -> bool:
        line = note_line.strip()
        if not line:
            return False
        content = self.files.read_tier(tier)
        normalized_line = line if line.startswith("- ") else f"- {line}"
        incoming_key = normalize_tier_line(normalized_line)
        existing_keys = {normalize_tier_line(item) for item in content.splitlines() if item.strip()}
        if incoming_key in existing_keys:
            return False
        separator = "\n" if content.endswith("\n") or not content else "\n\n"
        updated = f"{content}{separator}{normalized_line}\n"
        self.files.write_tier(tier, updated)
        return True

    def read_tier(self, tier: KnowledgeTier) -> str:
        return self.files.read_tier(tier)

    def search(self, *, query: str, top_k: int = 5) -> list[Chunk]:
        return self.storage.search_chunks_fts(query=query, top_k=max(1, int(top_k)))

    def export_snapshot(self) -> dict[str, Any]:
        return {
            "tiers": {
                tier.value: self.files.read_tier(tier)
                for tier in (KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT)
            },
            "chunks": [chunk.model_dump(mode="json") for chunk in self.storage.list_chunks()],
            "stats": self.storage.get_stats(),
        }


def infer_tier_from_label(label: SemanticLabel) -> KnowledgeTier:
    if label in {SemanticLabel.HARD_FAILURE, SemanticLabel.GOTCHA, SemanticLabel.CORRECTION}:
        return KnowledgeTier.GUARDRAILS
    if label in {SemanticLabel.PREFERENCE, SemanticLabel.PATTERN, SemanticLabel.DECISION_RATIONALE}:
        return KnowledgeTier.STYLE
    return KnowledgeTier.RECENT
