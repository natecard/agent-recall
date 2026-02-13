from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_recall.core.adapters import write_adapter_payloads
from agent_recall.core.context import ContextAssembler
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage


class ContextRefreshHook:
    """Hook for refreshing context and adapter payloads between iterations."""

    def __init__(self, agent_dir: Path, storage: Storage, files: FileStorage):
        self.agent_dir = agent_dir
        self.storage = storage
        self.files = files
        self.assembler = ContextAssembler(storage, files)

    def refresh(
        self,
        task: str | None = None,
        item_id: str | None = None,
        iteration: int | None = None,
        token_budget: int | None = None,
    ) -> dict[str, object]:
        task_text = task

        if item_id or iteration is not None:
            if not task_text:
                task_text = "Continue work"
            if item_id and iteration is not None:
                task_text = f"[Ralph Iteration {iteration}] {item_id}: {task_text}"
            elif item_id:
                task_text = f"{item_id}: {task_text}"
            elif iteration is not None:
                task_text = f"[Ralph Iteration {iteration}] {task_text}"

        context = self.assembler.assemble(task=task_text, include_retrieval=True)
        refreshed_at = datetime.now(UTC)
        active_session_id = f"ralph-iteration-{iteration}" if iteration is not None else None
        written = write_adapter_payloads(
            context=context,
            task=task_text,
            active_session_id=active_session_id,
            repo_path=self.agent_dir.parent,
            refreshed_at=refreshed_at,
            output_dir=self.agent_dir,
            token_budget=token_budget,
        )

        return {
            "context_length": len(context),
            "adapters_written": list(written.keys()),
            "refreshed_at": refreshed_at.isoformat(),
            "task": task_text,
        }

    def refresh_for_prd_item(
        self,
        prd_item: dict[str, object],
        *,
        iteration: int | None = None,
        token_budget: int | None = None,
    ) -> dict[str, object]:
        item_id = prd_item.get("id")
        title = prd_item.get("title")
        description = prd_item.get("description")
        task_parts: list[str] = []

        if isinstance(title, str) and title.strip():
            task_parts.append(title.strip())
        if isinstance(description, str) and description.strip():
            task_parts.append(description.strip())
        task = " - ".join(task_parts) if task_parts else None

        return self.refresh(
            task=task,
            item_id=str(item_id) if item_id is not None else None,
            iteration=iteration,
            token_budget=token_budget,
        )
