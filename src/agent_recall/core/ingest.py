from __future__ import annotations

import json
from pathlib import Path

from agent_recall.core.log import LogWriter
from agent_recall.storage.models import LogSource, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


class TranscriptIngestor:
    """Ingest native session transcripts into raw immutable log entries."""

    def __init__(self, storage: SQLiteStorage):
        self.storage = storage
        self.log_writer = LogWriter(storage)

    def ingest_jsonl(
        self,
        path: Path,
        source_session_id: str | None = None,
        default_label: SemanticLabel = SemanticLabel.NARRATIVE,
    ) -> int:
        if source_session_id and self.storage.is_session_processed(source_session_id):
            return 0

        count = 0
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = self._extract_content(payload)
            if not content:
                continue

            self.log_writer.log(
                content=content,
                label=default_label,
                source=LogSource.INGESTED,
            )
            count += 1

        if source_session_id:
            self.storage.mark_session_processed(source_session_id)

        return count

    @staticmethod
    def _extract_content(payload: dict) -> str | None:
        # Handles basic transcript shapes from tool logs.
        if isinstance(payload.get("content"), str):
            return payload["content"].strip() or None

        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip() or None
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        text_parts.append(item["text"].strip())
                joined = "\n".join(part for part in text_parts if part)
                return joined or None

        return None
