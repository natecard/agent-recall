from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CurationQueueStatus(str):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class CurationQueueItem:
    chunk_id: str
    source: str
    timestamp: datetime
    content_preview: str
    proposed_label: str
    status: str = CurationQueueStatus.PENDING
    entry_id: str | None = None
    content: str = ""
    label_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "content_preview": self.content_preview,
            "proposed_label": self.proposed_label,
            "status": self.status,
            "entry_id": self.entry_id,
            "content": self.content,
            "label_confidence": self.label_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurationQueueItem:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = _utcnow()
        return cls(
            chunk_id=data.get("chunk_id", str(uuid4())),
            source=data.get("source", "unknown"),
            timestamp=timestamp,
            content_preview=data.get("content_preview", "")[:200],
            proposed_label=data.get("proposed_label", "pattern"),
            status=data.get("status", CurationQueueStatus.PENDING),
            entry_id=data.get("entry_id"),
            content=data.get("content", ""),
            label_confidence=data.get("label_confidence", 1.0),
        )


class CurationQueueStore:
    QUEUE_FILE = "curation_queue.json"

    def __init__(self, agent_dir: Path):
        self._agent_dir = agent_dir
        self._queue_path = agent_dir / self.QUEUE_FILE

    def load(self) -> list[CurationQueueItem]:
        if not self._queue_path.exists():
            return []
        try:
            data = json.loads(self._queue_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return []
            items = data.get("items", [])
            if not isinstance(items, list):
                return []
            return [CurationQueueItem.from_dict(item) for item in items]
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, items: list[CurationQueueItem]) -> None:
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _utcnow().isoformat(),
            "items": [item.to_dict() for item in items],
        }
        self._queue_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, item: CurationQueueItem) -> None:
        items = self.load()
        for existing in items:
            if existing.chunk_id == item.chunk_id:
                return
        items.append(item)
        self.save(items)

    def remove(self, chunk_id: str) -> bool:
        items = self.load()
        original_len = len(items)
        items = [item for item in items if item.chunk_id != chunk_id]
        if len(items) < original_len:
            self.save(items)
            return True
        return False

    def update_status(self, chunk_id: str, status: str) -> bool:
        items = self.load()
        for item in items:
            if item.chunk_id == chunk_id:
                item.status = status
                self.save(items)
                return True
        return False

    def update_label(self, chunk_id: str, label: str) -> bool:
        items = self.load()
        for item in items:
            if item.chunk_id == chunk_id:
                item.proposed_label = label
                self.save(items)
                return True
        return False

    def get_pending(self) -> list[CurationQueueItem]:
        return [item for item in self.load() if item.status == CurationQueueStatus.PENDING]

    def count_pending(self) -> int:
        return len(self.get_pending())

    def approve_all(self) -> int:
        items = self.load()
        count = 0
        for item in items:
            if item.status == CurationQueueStatus.PENDING:
                item.status = CurationQueueStatus.APPROVED
                count += 1
        if count > 0:
            self.save(items)
        return count

    def clear(self) -> None:
        if self._queue_path.exists():
            self._queue_path.unlink()
