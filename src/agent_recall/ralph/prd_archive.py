from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from agent_recall.core.embeddings import cosine_similarity, generate_embedding
from agent_recall.storage.base import Storage
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


@dataclass
class ArchivedPRDItem:
    id: str
    title: str
    description: str
    user_story: str
    steps: list[str]
    acceptance_criteria: list[str]
    validation_commands: list[str]
    completed_at: datetime
    completion_iteration: int
    archive_id: UUID = field(default_factory=uuid4)
    key_decisions: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    commit_hashes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "archive_id": str(self.archive_id),
            "title": self.title,
            "description": self.description,
            "user_story": self.user_story,
            "steps": list(self.steps),
            "acceptance_criteria": list(self.acceptance_criteria),
            "validation_commands": list(self.validation_commands),
            "completed_at": self.completed_at.astimezone(UTC).isoformat(),
            "completion_iteration": self.completion_iteration,
            "key_decisions": list(self.key_decisions),
            "lessons_learned": list(self.lessons_learned),
            "related_files": list(self.related_files),
            "commit_hashes": list(self.commit_hashes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchivedPRDItem:
        acceptance = data.get("acceptance_criteria")
        if acceptance is None:
            acceptance = data.get("acceptance")
        acceptance_list = list(acceptance) if isinstance(acceptance, list) else []

        validation = data.get("validation_commands")
        if validation is None:
            validation = data.get("validation")
        validation_list = list(validation) if isinstance(validation, list) else []

        completed_raw = data.get("completed_at")
        completed_at = _parse_datetime(completed_raw)

        archive_id_raw = data.get("archive_id")
        archive_id = UUID(str(archive_id_raw)) if archive_id_raw else uuid4()

        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            user_story=str(data.get("user_story", "")),
            steps=list(data.get("steps") or []),
            acceptance_criteria=acceptance_list,
            validation_commands=validation_list,
            completed_at=completed_at,
            completion_iteration=int(data.get("completion_iteration") or 0),
            archive_id=archive_id,
            key_decisions=list(data.get("key_decisions") or []),
            lessons_learned=list(data.get("lessons_learned") or []),
            related_files=list(data.get("related_files") or []),
            commit_hashes=list(data.get("commit_hashes") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_searchable_text(self) -> str:
        sections: list[str] = [
            f"ID: {self.id}",
            f"Title: {self.title}",
            f"Description: {self.description}",
            f"User Story: {self.user_story}",
        ]
        if self.steps:
            steps_text = "\n".join(f"- {step}" for step in self.steps)
            sections.append(f"Steps:\n{steps_text}")
        if self.acceptance_criteria:
            acceptance_text = "\n".join(f"- {item}" for item in self.acceptance_criteria)
            sections.append(f"Acceptance:\n{acceptance_text}")
        if self.key_decisions:
            decisions_text = "\n".join(f"- {item}" for item in self.key_decisions)
            sections.append(f"Decisions:\n{decisions_text}")
        if self.lessons_learned:
            lessons_text = "\n".join(f"- {item}" for item in self.lessons_learned)
            sections.append(f"Lessons:\n{lessons_text}")
        return "\n".join(sections).strip()


class PRDArchive:
    def __init__(self, agent_dir: Path, storage: Storage | None = None):
        self.agent_dir = agent_dir
        self.storage = storage
        self.archive_path = agent_dir / "ralph" / "prd_archive.json"

    def _load_archive(self) -> list[ArchivedPRDItem]:
        if not self.archive_path.exists():
            return []
        try:
            payload = json.loads(self.archive_path.read_text())
        except json.JSONDecodeError:
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        return [ArchivedPRDItem.from_dict(item) for item in items if isinstance(item, dict)]

    def _save_archive(self, items: list[ArchivedPRDItem]) -> None:
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "items": [item.to_dict() for item in items],
        }
        self.archive_path.write_text(json.dumps(payload, indent=2))

    def archive_item(self, prd_item: dict[str, Any], *, iteration: int = 0) -> ArchivedPRDItem:
        archived = ArchivedPRDItem(
            id=str(prd_item.get("id", "")),
            title=str(prd_item.get("title", "")),
            description=str(prd_item.get("description", "")),
            user_story=str(prd_item.get("user_story", "")),
            steps=list(prd_item.get("steps") or []),
            acceptance_criteria=list(prd_item.get("acceptance") or []),
            validation_commands=list(prd_item.get("validation") or []),
            completed_at=datetime.now(UTC),
            completion_iteration=int(iteration),
            key_decisions=list(prd_item.get("key_decisions") or []),
            lessons_learned=list(prd_item.get("lessons_learned") or []),
            related_files=list(prd_item.get("related_files") or []),
            commit_hashes=list(prd_item.get("commit_hashes") or []),
            metadata=dict(prd_item.get("metadata") or {}),
        )
        items = self._load_archive()
        filtered = [item for item in items if item.id != archived.id]
        filtered.append(archived)
        self._save_archive(filtered)
        if self.storage is not None:
            self._index_archived_item(archived)
        return archived

    def archive_completed_from_prd(
        self, prd_path: Path, iteration: int = 0
    ) -> list[ArchivedPRDItem]:
        payload = json.loads(prd_path.read_text())
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []
        existing_ids = {item.id for item in self._load_archive()}
        completed_item_ids: set[str] = set()
        archived_items: list[ArchivedPRDItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("passes"):
                continue
            item_id = str(item.get("id", ""))
            if not item_id or item_id in existing_ids:
                if item_id and item_id in existing_ids:
                    completed_item_ids.add(item_id)
                continue
            archived_items.append(self.archive_item(item, iteration=iteration))
            existing_ids.add(item_id)
            completed_item_ids.add(item_id)

        if completed_item_ids:
            self._prune_prd_items(prd_path=prd_path, archived_ids=completed_item_ids)
        return archived_items

    def prune_archived_from_prd(self, prd_path: Path) -> int:
        """Remove from the PRD any passing items that are already in the archive.

        Use this to prune older PRDs that were not pruned previously (e.g. before
        pruning was added, or when archiving happened outside the loop).
        Returns the number of items pruned from the PRD.
        """
        archived_ids = {item.id for item in self._load_archive()}
        if not archived_ids:
            return 0

        payload = json.loads(prd_path.read_text())
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return 0

        to_prune: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("passes"):
                continue
            item_id = str(item.get("id", ""))
            if item_id and item_id in archived_ids:
                to_prune.add(item_id)

        if to_prune:
            return self._prune_prd_items(prd_path=prd_path, archived_ids=to_prune)
        return 0

    def _prune_prd_items(self, prd_path: Path, archived_ids: set[str]) -> int:
        """Remove passing items whose IDs are in archived_ids from the PRD.
        Returns the number of items removed.
        """
        if not archived_ids:
            return 0

        payload = json.loads(prd_path.read_text())
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return 0

        filtered_items: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                filtered_items.append(item)
                continue
            item_id = str(item.get("id", ""))
            should_remove = bool(item.get("passes")) and item_id in archived_ids
            if should_remove:
                continue
            filtered_items.append(item)

        removed = len(items) - len(filtered_items)
        if removed > 0:
            payload["items"] = filtered_items
            prd_path.write_text(json.dumps(payload, indent=2))
        return removed

    def get_by_id(self, item_id: str) -> ArchivedPRDItem | None:
        for item in self._load_archive():
            if item.id == item_id:
                return item
        return None

    def list_all(self) -> list[ArchivedPRDItem]:
        return self._load_archive()

    def _index_archived_item(self, item: ArchivedPRDItem) -> None:
        if self.storage is None:
            return
        text = item.to_searchable_text()
        embedding = generate_embedding(text, dimensions=64)
        chunk = Chunk(
            source=ChunkSource.IMPORT,
            content=text,
            label=SemanticLabel.DECISION_RATIONALE,
            tags=["prd", "archived", item.id.lower()],
            embedding=embedding,
        )
        self.storage.store_chunk(chunk)

    def search(
        self, query: str, top_k: int = 5, item_ids: list[str] | None = None
    ) -> list[tuple[ArchivedPRDItem, float]]:
        items = self._load_archive()
        if not items:
            return []

        query_embedding = generate_embedding(query, dimensions=64)
        allowed_ids = {item_id.lower() for item_id in item_ids} if item_ids else None
        scored: list[tuple[ArchivedPRDItem, float]] = []
        for item in items:
            if allowed_ids is not None and item.id.lower() not in allowed_ids:
                continue
            item_embedding = generate_embedding(item.to_searchable_text(), dimensions=64)
            score = cosine_similarity(query_embedding, item_embedding)
            scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(0, top_k)]

    def match_knowledge_to_prd(
        self, knowledge: str, top_k: int = 5, item_ids: list[str] | None = None
    ) -> list[tuple[ArchivedPRDItem, float]]:
        return self.search(knowledge, top_k=top_k, item_ids=item_ids)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = datetime.now(UTC)
    else:
        parsed = datetime.now(UTC)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
