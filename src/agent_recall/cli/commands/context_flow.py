from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from agent_recall.core.adapters import ContextAdapter, get_default_adapters, write_adapter_payloads
from agent_recall.core.context import ContextAssembler
from agent_recall.core.pr_context import (
    build_pr_context_output,
    extract_git_diff_scope,
    filter_chunks_for_scope,
)
from agent_recall.memory.agent_memory import (
    build_agent_memory_bundle,
    write_agent_memory_bundle,
)
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage


@dataclass(frozen=True)
class ContextRequest:
    task: str | None
    for_pr: bool
    base_ref: str
    head_ref: str
    max_diff_files: int
    top_k: int | None


@dataclass(frozen=True)
class ContextResult:
    task: str | None
    for_pr: bool
    context: str
    scope_payload: dict[str, Any] | None


@dataclass(frozen=True)
class ContextBundleWriteRequest:
    context: str
    task: str | None
    active_session_id: str | None
    repo_path: Path
    output_dir: Path
    refreshed_at: datetime
    storage: Storage
    files: FileStorage
    adapter_payloads: bool = False
    adapter_output_dir: Path | None = None
    adapters: list[ContextAdapter] | None = None
    token_budget: int | None = None
    per_adapter_budgets: dict[str, int] | None = None
    per_provider_budgets: dict[str, int] | None = None
    per_model_budgets: dict[str, int] | None = None
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ContextBundleWriteResult:
    markdown_path: Path
    json_path: Path
    agent_memory_path: Path
    refreshed_at: datetime
    adapters_written: dict[str, Path]


def assemble_standard_context(
    *,
    storage: Storage,
    files: FileStorage,
    retriever,
    retrieval_top_k: int,
    task: str | None,
) -> str:
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=retrieval_top_k,
    )
    return context_asm.assemble(task=task)


def write_context_bundle(request: ContextBundleWriteRequest) -> ContextBundleWriteResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = request.output_dir / "context.md"
    json_path = request.output_dir / "context.json"
    markdown_path.write_text(request.context)

    refreshed_at = (
        request.refreshed_at.replace(tzinfo=UTC)
        if request.refreshed_at.tzinfo is None
        else request.refreshed_at.astimezone(UTC)
    )
    payload = {
        "task": request.task,
        "active_session_id": request.active_session_id,
        "repo_path": str(request.repo_path),
        "refreshed_at": refreshed_at.isoformat(),
        "context": request.context,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    agent_memory_bundle = build_agent_memory_bundle(
        storage=request.storage,
        files=request.files,
        task=request.task,
        active_session_id=request.active_session_id,
        repo_path=request.repo_path,
        refreshed_at=refreshed_at,
    )
    agent_memory_path = write_agent_memory_bundle(request.output_dir, agent_memory_bundle)

    adapters_written: dict[str, Path] = {}
    if request.adapter_payloads:
        adapter_dir = request.adapter_output_dir or request.output_dir
        adapters_written = write_adapter_payloads(
            context=request.context,
            task=request.task,
            active_session_id=request.active_session_id,
            repo_path=request.repo_path,
            refreshed_at=refreshed_at,
            output_dir=adapter_dir,
            adapters=request.adapters or get_default_adapters(),
            token_budget=request.token_budget,
            per_adapter_budgets=request.per_adapter_budgets,
            per_provider_budgets=request.per_provider_budgets,
            per_model_budgets=request.per_model_budgets,
            provider=request.provider,
            model=request.model,
            agent_memory=agent_memory_bundle,
            agent_memory_path=agent_memory_path,
        )

    return ContextBundleWriteResult(
        markdown_path=markdown_path,
        json_path=json_path,
        agent_memory_path=agent_memory_path,
        refreshed_at=refreshed_at,
        adapters_written=adapters_written,
    )


def execute_context_request(
    *,
    request: ContextRequest,
    storage: Storage,
    files: FileStorage,
    retriever,
    retrieval_top_k: int,
) -> ContextResult:
    effective_top_k = request.top_k if request.top_k is not None else retrieval_top_k

    if request.for_pr:
        scope = extract_git_diff_scope(
            repo_root=Path.cwd(),
            base_ref=request.base_ref,
            head_ref=request.head_ref,
            max_files=request.max_diff_files,
        )
        inferred_query = request.task
        if inferred_query is None and scope.modules:
            inferred_query = " ".join(scope.modules[:5])
        if inferred_query is None and scope.files:
            inferred_query = " ".join(Path(path).stem for path in scope.files[:5])

        raw_chunks = (
            retriever.search(query=inferred_query, top_k=effective_top_k)
            if inferred_query is not None and inferred_query.strip()
            else []
        )
        scoped_chunks = filter_chunks_for_scope(raw_chunks, scope)[:effective_top_k]
        output = build_pr_context_output(
            files=files,
            scope=scope,
            chunks=scoped_chunks,
            query=inferred_query,
        )
        return ContextResult(
            task=request.task,
            for_pr=True,
            context=output,
            scope_payload=scope.to_dict(),
        )

    output = assemble_standard_context(
        storage=storage,
        files=files,
        retriever=retriever,
        retrieval_top_k=effective_top_k,
        task=request.task,
    )
    return ContextResult(
        task=request.task,
        for_pr=False,
        context=output,
        scope_payload=None,
    )


def render_context_result_markdown(result: ContextResult, *, console: Console) -> None:
    console.print(result.context)


def context_result_json_payload(result: ContextResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": result.task,
        "for_pr": result.for_pr,
        "context": result.context,
    }
    if result.scope_payload is not None:
        payload["scope"] = result.scope_payload
    return payload
