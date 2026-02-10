from __future__ import annotations

from agent_recall.storage.models import Chunk, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


class Retriever:
    def __init__(self, storage: SQLiteStorage):
        self.storage = storage

    def search(
        self,
        query: str,
        top_k: int = 5,
        labels: list[SemanticLabel] | None = None,
    ) -> list[Chunk]:
        chunks = self.storage.search_chunks_fts(query=query, top_k=top_k)
        if not labels:
            return chunks
        allowed = {label.value for label in labels}
        return [chunk for chunk in chunks if chunk.label.value in allowed]
