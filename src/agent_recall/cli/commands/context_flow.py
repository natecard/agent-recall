from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from agent_recall.core.context import ContextAssembler
from agent_recall.core.pr_context import (
    build_pr_context_output,
    extract_git_diff_scope,
    filter_chunks_for_scope,
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


def execute_context_request(
    *,
    request: ContextRequest,
    storage: Storage,
    files: FileStorage,
    retriever,
    retrieval_top_k: int,
) -> ContextResult:
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=request.top_k if request.top_k is not None else retrieval_top_k,
    )
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

    output = context_asm.assemble(task=request.task)
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
