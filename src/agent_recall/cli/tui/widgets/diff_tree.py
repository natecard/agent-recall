from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, cast

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Static, Tree
from textual.widgets.tree import TreeNode

from agent_recall.cli.tui.utils.diff_parser import DiffFile


@dataclass
class DiffTreeData:
    name: str
    path: str
    is_dir: bool
    added: int
    deleted: int
    diff_file: DiffFile | None = None


class DiffTreeViewer(Vertical):
    DEFAULT_CSS = """
    DiffTreeViewer {
        height: 1fr;
        width: 100%;
        overflow: hidden;
    }
    DiffTreeViewer > Static {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    DiffTreeViewer > Input {
        dock: bottom;
        margin: 1;
        height: 3;
    }
    DiffTreeViewer > Tree {
        height: 1fr;
        overflow: auto;
    }
    """

    BINDINGS = [
        Binding("e", "expand_all", "Expand All"),
        Binding("c", "collapse_all", "Collapse All"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear", show=False),
    ]

    class FileSelected(Message):
        def __init__(self, diff_file: DiffFile) -> None:
            super().__init__()
            self.diff_file = diff_file

    class StatsChanged(Message):
        def __init__(self, total_files: int, total_added: int, total_deleted: int) -> None:
            super().__init__()
            self.total_files = total_files
            self.total_added = total_added
            self.total_deleted = total_deleted

    def __init__(
        self,
        diff_files: list[DiffFile] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._diff_files = diff_files or []
        self._filter_text = ""

    def set_diff_files(self, diff_files: list[DiffFile]) -> None:
        self._diff_files = diff_files
        self._rebuild_tree()
        self._emit_stats()

    def compose(self) -> ComposeResult:
        yield Static("Changed Files", id="diff_tree_header")
        yield Tree[DiffTreeData]("Files", id="diff_tree")
        yield Input(placeholder="Filter files...", id="diff_tree_filter")

    def on_mount(self) -> None:
        self._rebuild_tree()
        self._emit_stats()

    @on(Input.Changed, "#diff_tree_filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        self._filter_text = event.value.lower().strip()
        self._rebuild_tree()

    @on(Tree.NodeSelected)
    def _on_tree_node_selected(self, event: Tree.NodeSelected[DiffTreeData]) -> None:
        node = event.node
        if node.data and node.data.diff_file:
            self.post_message(self.FileSelected(node.data.diff_file))

    def action_expand_all(self) -> None:
        tree = self.query_one(Tree)
        tree.root.expand_all()

    def action_collapse_all(self) -> None:
        tree = self.query_one(Tree)
        tree.root.collapse_all()
        tree.root.expand()

    def action_focus_search(self) -> None:
        self.query_one("#diff_tree_filter", Input).focus()

    def action_clear_search(self) -> None:
        filter_input = self.query_one("#diff_tree_filter", Input)
        filter_input.value = ""
        self.query_one(Tree).focus()

    def _rebuild_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.clear()

        if not self._diff_files:
            tree.root.add_leaf("[dim]No changes[/dim]", data=None)
            return

        filtered_files = self._filter_files()
        if not filtered_files:
            tree.root.add_leaf("[dim]No matching files[/dim]", data=None)
            return

        self._build_tree_nodes(tree.root, filtered_files)
        tree.root.expand()

    def _filter_files(self) -> list[DiffFile]:
        if not self._filter_text:
            return self._diff_files

        return [f for f in self._diff_files if self._filter_text in f.path.lower()]

    def _build_tree_nodes(
        self,
        parent: TreeNode[DiffTreeData],
        files: list[DiffFile],
    ) -> None:
        tree_structure: dict[str, Any] = {}

        for diff_file in files:
            parts = list(PurePath(diff_file.path).parts)
            current = tree_structure

            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                if is_last:
                    current[part] = diff_file
                else:
                    if part not in current:
                        current[part] = {}
                    next_level = current.get(part)
                    if isinstance(next_level, dict):
                        current = next_level
                    else:
                        current = {}

        self._add_nodes(parent, tree_structure, "")

    def _add_nodes(
        self,
        parent: TreeNode[DiffTreeData],
        structure: dict[str, Any],
        current_path: str,
    ) -> None:
        for name, value in sorted(structure.items(), key=self._sort_key):
            full_path = f"{current_path}/{name}" if current_path else name

            if isinstance(value, DiffFile):
                data = DiffTreeData(
                    name=name,
                    path=full_path,
                    is_dir=False,
                    added=value.added,
                    deleted=value.deleted,
                    diff_file=value,
                )
                label = self._render_file_label(name, value)
                parent.add_leaf(label, data=data)
            else:
                nested: dict[str, Any] = (
                    cast(dict[str, Any], value) if isinstance(value, dict) else {}
                )
                dir_data = self._compute_dir_stats(nested)
                data = DiffTreeData(
                    name=name,
                    path=full_path,
                    is_dir=True,
                    added=dir_data["added"],
                    deleted=dir_data["deleted"],
                )
                label = self._render_dir_label(name, data)
                node = parent.add(label, data=data)
                self._add_nodes(node, nested, full_path)
                node.expand()

    def _sort_key(self, item: tuple[str, object]) -> tuple[int, str]:
        name, value = item
        is_dir = not isinstance(value, DiffFile)
        return (0 if is_dir else 1, name.lower())

    def _compute_dir_stats(self, structure: dict[str, Any]) -> dict:
        total_added = 0
        total_deleted = 0

        def recurse(s: dict[str, Any]) -> None:
            nonlocal total_added, total_deleted
            for v in s.values():
                if isinstance(v, DiffFile):
                    total_added += v.added
                    total_deleted += v.deleted
                elif isinstance(v, dict):
                    recurse(v)

        recurse(structure)
        return {"added": total_added, "deleted": total_deleted}

    def _render_file_label(self, name: str, diff_file: DiffFile) -> Text:
        text = Text()
        text.append("  ")
        text.append(name)

        if diff_file.added > 0 or diff_file.deleted > 0:
            text.append(" ")
            if diff_file.added > 0:
                text.append(f"+{diff_file.added}", style="green")
            if diff_file.deleted > 0:
                if diff_file.added > 0:
                    text.append(" ")
                text.append(f"-{diff_file.deleted}", style="red")

        return text

    def _render_dir_label(self, name: str, data: DiffTreeData) -> Text:
        text = Text()
        text.append(" ")
        text.append(name, style="bold")

        if data.added > 0 or data.deleted > 0:
            text.append(" ")
            if data.added > 0:
                text.append(f"+{data.added}", style="green")
            if data.deleted > 0:
                if data.added > 0:
                    text.append(" ")
                text.append(f"-{data.deleted}", style="red")

        return text

    def _emit_stats(self) -> None:
        total_files = len(self._diff_files)
        total_added = sum(f.added for f in self._diff_files)
        total_deleted = sum(f.deleted for f in self._diff_files)
        self.post_message(self.StatsChanged(total_files, total_added, total_deleted))
