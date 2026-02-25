from __future__ import annotations

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier


class ContextAssembler:
    def __init__(
        self,
        storage: Storage,
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

        rules_path = self.files.agent_dir / "RULES.md"
        if rules_path.exists():
            try:
                rules = rules_path.read_text(encoding="utf-8")
            except OSError:
                rules = ""
            if rules.strip():
                parts.append(f"## Rules\n\n{rules.strip()}")

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
            if hasattr(retriever, "search_hybrid"):
                chunks = retriever.search_hybrid(
                    query=task,
                    top_k=self.retrieval_top_k,
                    fts_weight=0.4,
                    semantic_weight=0.6,
                )
            else:
                chunks = retriever.search(task, top_k=self.retrieval_top_k, backend="hybrid")
            if chunks:
                relevant = "\n".join(f"- {chunk.content}" for chunk in chunks)
                parts.append(f'## Relevant to "{task}"\n\n{relevant}')

        return "\n\n---\n\n".join(parts) if parts else "No context available yet."
