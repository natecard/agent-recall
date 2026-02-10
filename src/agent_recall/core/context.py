from __future__ import annotations

from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.sqlite import SQLiteStorage


class ContextAssembler:
    def __init__(self, storage: SQLiteStorage, files: FileStorage):
        self.storage = storage
        self.files = files

    def assemble(self, task: str | None = None, include_retrieval: bool = True) -> str:
        """Assemble full context for an agent."""
        parts: list[str] = []

        guardrails = self.files.read_tier(KnowledgeTier.GUARDRAILS)
        if guardrails.strip():
            parts.append(f"## Guardrails\n\n{guardrails.strip()}")

        style = self.files.read_tier(KnowledgeTier.STYLE)
        if style.strip():
            parts.append(f"## Style\n\n{style.strip()}")

        recent = self.files.read_tier(KnowledgeTier.RECENT)
        if recent.strip():
            parts.append(f"## Recent Sessions\n\n{recent.strip()}")

        if task and include_retrieval:
            chunks = self.storage.search_chunks_fts(task, top_k=5)
            if chunks:
                relevant = "\n".join(f"- {chunk.content}" for chunk in chunks)
                parts.append(f"## Relevant to \"{task}\"\n\n{relevant}")

        return "\n\n---\n\n".join(parts) if parts else "No context available yet."
