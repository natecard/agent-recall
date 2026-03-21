from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import Chunk


@dataclass(frozen=True)
class DiffScope:
    base_ref: str
    head_ref: str
    files: list[str]
    modules: list[str]
    renamed: list[tuple[str, str]]
    added: int
    modified: int
    deleted: int
    truncated: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "files": list(self.files),
            "modules": list(self.modules),
            "renamed": [[old, new] for old, new in self.renamed],
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "truncated": self.truncated,
            "error": self.error,
        }


def parse_name_status_lines(
    *,
    lines: list[str],
    base_ref: str,
    head_ref: str,
    max_files: int = 200,
) -> DiffScope:
    files: list[str] = []
    modules: list[str] = []
    seen_files: set[str] = set()
    seen_modules: set[str] = set()
    renamed: list[tuple[str, str]] = []
    added = 0
    modified = 0
    deleted = 0

    def _add_file(path: str) -> None:
        normalized = path.strip()
        if not normalized or normalized in seen_files:
            return
        seen_files.add(normalized)
        files.append(normalized)
        module = normalized.split("/", 1)[0]
        if module and module not in seen_modules:
            seen_modules.add(module)
            modules.append(module)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0].strip().upper()
        prefix = status[:1]
        if prefix == "A":
            added += 1
        elif prefix == "D":
            deleted += 1
        else:
            modified += 1

        if status.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1].strip(), parts[2].strip()
            if old_path and new_path:
                renamed.append((old_path, new_path))
                _add_file(new_path)
            continue

        if len(parts) >= 2:
            _add_file(parts[1])

    limit = max(1, int(max_files))
    truncated = len(files) > limit
    if truncated:
        files = files[:limit]

    return DiffScope(
        base_ref=base_ref,
        head_ref=head_ref,
        files=files,
        modules=modules,
        renamed=renamed,
        added=added,
        modified=modified,
        deleted=deleted,
        truncated=truncated,
        error=None,
    )


def extract_git_diff_scope(
    *,
    repo_root: Path,
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    max_files: int = 200,
) -> DiffScope:
    command = [
        "git",
        "-C",
        str(repo_root),
        "diff",
        "--name-status",
        "--find-renames",
        f"{base_ref}...{head_ref}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return DiffScope(
            base_ref=base_ref,
            head_ref=head_ref,
            files=[],
            modules=[],
            renamed=[],
            added=0,
            modified=0,
            deleted=0,
            truncated=False,
            error=str(exc),
        )

    if completed.returncode != 0:
        return DiffScope(
            base_ref=base_ref,
            head_ref=head_ref,
            files=[],
            modules=[],
            renamed=[],
            added=0,
            modified=0,
            deleted=0,
            truncated=False,
            error=completed.stderr.strip() or "git diff failed",
        )

    lines = completed.stdout.splitlines()
    return parse_name_status_lines(
        lines=lines,
        base_ref=base_ref,
        head_ref=head_ref,
        max_files=max_files,
    )


def filter_chunks_for_scope(chunks: list[Chunk], scope: DiffScope) -> list[Chunk]:
    if not chunks or (not scope.files and not scope.modules):
        return chunks
    file_tokens = {path.lower() for path in scope.files}
    module_tokens = {module.lower() for module in scope.modules}
    filtered: list[Chunk] = []
    for chunk in chunks:
        haystack = f"{chunk.content} {' '.join(chunk.tags)}".lower()
        if any(token in haystack for token in file_tokens):
            filtered.append(chunk)
            continue
        if any(f"{module}/" in haystack or f" {module} " in haystack for module in module_tokens):
            filtered.append(chunk)
    return filtered if filtered else chunks


def build_pr_context_output(
    *,
    files: FileStorage,
    scope: DiffScope,
    chunks: list[Chunk],
    query: str | None = None,
) -> str:
    parts: list[str] = []

    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS).strip()
    style = files.read_tier(KnowledgeTier.STYLE).strip()
    recent = files.read_tier(KnowledgeTier.RECENT).strip()
    if guardrails:
        parts.append(f"## Guardrails\n\n{guardrails}")
    if style:
        parts.append(f"## Style\n\n{style}")
    if recent:
        parts.append(f"## Recent Sessions\n\n{recent}")

    scope_lines = [
        f"- Base: `{scope.base_ref}`",
        f"- Head: `{scope.head_ref}`",
        f"- Changed files: {len(scope.files)}",
        f"- Added: {scope.added}, Modified: {scope.modified}, Deleted: {scope.deleted}",
    ]
    if scope.modules:
        scope_lines.append(f"- Modules: {', '.join(scope.modules[:12])}")
    if scope.truncated:
        scope_lines.append("- Diff scope truncated to configured max files.")
    if scope.error:
        scope_lines.append(f"- Diff extraction warning: {scope.error}")
    if scope.files:
        scoped_file_lines = "\n".join(f"  - `{path}`" for path in scope.files[:30])
        scope_lines.append("  Files:\n" + scoped_file_lines)
    if scope.renamed:
        renamed_lines = "\n".join(f"  - `{old}` -> `{new}`" for old, new in scope.renamed[:20])
        scope_lines.append("  Renames:\n" + renamed_lines)
    parts.append("## PR Scope\n\n" + "\n".join(scope_lines))

    focus = [
        "## Review Focus",
        "",
        "1. Validate behavior changes and edge-case handling in changed modules.",
        "2. Check for migration, API contract, and backward-compatibility risks.",
        "3. Confirm tests cover renamed paths and affected integration boundaries.",
    ]
    parts.append("\n".join(focus))

    if chunks:
        title = f"## Relevant to PR Scope ({query})" if query else "## Relevant to PR Scope"
        chunk_lines = "\n".join(f"- ({chunk.label.value}) {chunk.content}" for chunk in chunks)
        parts.append(f"{title}\n\n{chunk_lines}")

    return "\n\n---\n\n".join(parts) if parts else "No PR context available."
