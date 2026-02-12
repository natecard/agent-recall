from __future__ import annotations

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.sqlite import SQLiteStorage


class ContextAssembler:
    def __init__(
        self,
        storage: SQLiteStorage,
        files: FileStorage,
        retriever: Retriever | None = None,
        retrieval_top_k: int = 5,
    ):
        self.storage = storage
        self.files = files
        self.retriever = retriever
        self.retrieval_top_k = max(1, retrieval_top_k)

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
            retriever = self.retriever or Retriever(self.storage)
            chunks = retriever.search(task, top_k=self.retrieval_top_k)
            if chunks:
                relevant = "\n".join(f"- {chunk.content}" for chunk in chunks)
                parts.append(f"## Relevant to \"{task}\"\n\n{relevant}")

        return "\n\n---\n\n".join(parts) if parts else "No context available yet."
