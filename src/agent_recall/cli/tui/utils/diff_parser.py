from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePath


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class DiffFile:
    path: str
    old_path: str | None = None
    new_path: str | None = None
    added: int = 0
    deleted: int = 0
    hunks: list[DiffHunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def display_name(self) -> str:
        if self.path:
            return self.path
        if self.old_path and self.new_path:
            return f"{self.old_path} → {self.new_path}"
        return self.old_path or self.new_path or "unknown"


@dataclass
class DiffTreeNode:
    name: str
    path: str
    is_dir: bool = False
    added: int = 0
    deleted: int = 0
    children: dict[str, DiffTreeNode] = field(default_factory=dict)
    diff_file: DiffFile | None = None


def parse_diff_files(diff_text: str) -> list[DiffFile]:
    if not diff_text or not diff_text.strip():
        return []

    files: list[DiffFile] = []
    current_file: DiffFile | None = None
    current_hunk: DiffHunk | None = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if current_file is not None:
                if current_hunk is not None:
                    current_file.hunks.append(current_hunk)
                files.append(current_file)

            match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
            if match:
                old_path = match.group(1)
                new_path = match.group(2)
                current_file = DiffFile(
                    path=new_path,
                    old_path=old_path,
                    new_path=new_path,
                )
            else:
                match = re.match(r"^diff --git (.+?) (.+)$", line)
                if match:
                    current_file = DiffFile(
                        path=match.group(2),
                        old_path=match.group(1),
                        new_path=match.group(2),
                    )
                else:
                    current_file = DiffFile(path="unknown")
            current_hunk = None
            continue

        if current_file is None:
            continue

        if line.startswith("Binary files "):
            current_file.is_binary = True
            continue

        if line.startswith("--- "):
            old = line[4:].strip()
            if old.startswith("a/"):
                old = old[2:]
            if old != "/dev/null":
                current_file.old_path = old
            continue

        if line.startswith("+++ "):
            new = line[4:].strip()
            if new.startswith("b/"):
                new = new[2:]
            if new != "/dev/null":
                current_file.new_path = new
                current_file.path = new
            continue

        if line.startswith("@@ "):
            if current_hunk is not None:
                current_file.hunks.append(current_hunk)

            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=[line],
                )
            continue

        if current_hunk is not None:
            current_hunk.lines.append(line)

            if line.startswith("+") and not line.startswith("+++"):
                current_file.added += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_file.deleted += 1

    if current_file is not None:
        if current_hunk is not None:
            current_file.hunks.append(current_hunk)
        files.append(current_file)

    return files


def build_diff_tree(files: list[DiffFile]) -> DiffTreeNode:
    root = DiffTreeNode(name="root", path="", is_dir=True)

    for diff_file in files:
        path = diff_file.path
        parts = PurePath(path).parts

        current = root
        current_path = ""

        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            current_path = f"{current_path}/{part}" if current_path else part

            if part not in current.children:
                node = DiffTreeNode(
                    name=part,
                    path=current_path,
                    is_dir=not is_last,
                    added=0 if not is_last else diff_file.added,
                    deleted=0 if not is_last else diff_file.deleted,
                    diff_file=diff_file if is_last else None,
                )
                current.children[part] = node
            else:
                node = current.children[part]
                if is_last:
                    node.added = diff_file.added
                    node.deleted = diff_file.deleted
                    node.diff_file = diff_file

            current = node

    _aggregate_stats(root)

    return root


def _aggregate_stats(node: DiffTreeNode) -> tuple[int, int]:
    if not node.is_dir:
        return node.added, node.deleted

    total_added = 0
    total_deleted = 0

    for child in node.children.values():
        child_added, child_deleted = _aggregate_stats(child)
        total_added += child_added
        total_deleted += child_deleted

    node.added = total_added
    node.deleted = total_deleted

    return total_added, total_deleted


def get_hunk_diff_text(diff_file: DiffFile) -> str:
    lines = []
    lines.append(f"--- {diff_file.old_path or '/dev/null'}")
    lines.append(f"+++ {diff_file.new_path or '/dev/null'}")

    for hunk in diff_file.hunks:
        lines.extend(hunk.lines)

    return "\n".join(lines)
