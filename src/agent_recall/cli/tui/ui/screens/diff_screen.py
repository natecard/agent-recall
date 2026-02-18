from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from agent_recall.cli.tui.utils.diff_parser import DiffFile, parse_diff_files
from agent_recall.cli.tui.widgets.diff_content import DiffContentViewer, DiffMode
from agent_recall.cli.tui.widgets.diff_tree import DiffTreeViewer


@dataclass
class IterationMetadata:
    """Metadata for historical iteration diff display."""

    iteration: int
    item_id: str | None = None
    item_title: str | None = None
    commit_hash: str | None = None
    completed_at: datetime | None = None
    outcome: str | None = None


class DiffScreen(Screen[None]):
    DEFAULT_CSS = """
    DiffScreen {
        overflow: hidden;
    }
    DiffScreen > Container#diff_main {
        height: 1fr;
        width: 100%;
        overflow: hidden;
    }
    DiffScreen > Container#diff_main > Vertical#diff_tree_panel {
        width: 32;
        min-width: 20;
        max-width: 50;
        height: 1fr;
        border-right: solid $accent;
        background: $panel;
    }
    DiffScreen > Container#diff_main > Vertical#diff_content_panel {
        width: 1fr;
        height: 1fr;
        overflow: hidden;
    }
    DiffScreen #diff_header {
        height: auto;
        padding: 0 1;
        border-bottom: solid $accent;
    }
    DiffScreen #diff_header > Static#header_title {
        height: 1;
    }
    DiffScreen #diff_header > Static#header_meta {
        height: 1;
        color: $text-muted;
    }
    DiffScreen #diff_tree_panel > Static#tree_title {
        height: 1;
        padding: 0 1;
        text-style: bold;
        border-bottom: solid $accent;
        margin-bottom: 1;
    }
    DiffScreen #diff_content_panel > Static#content_title {
        height: 1;
        padding: 0 1;
        border-bottom: solid $accent;
    }
    DiffScreen #diff_content_panel > DiffContentViewer {
        height: 1fr;
        overflow: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("tab", "toggle_mode", "Toggle Mode"),
        Binding("s", "set_mode_side_by_side", "Side-by-Side"),
        Binding("u", "set_mode_unified", "Unified"),
        Binding("left", "focus_tree", "Tree"),
        Binding("right", "focus_content", "Content"),
    ]

    def __init__(
        self,
        diff_text: str,
        *,
        repo_dir: Path | None = None,
        title: str = "Diff Viewer",
        subtitle: str | None = None,
        iteration_meta: IterationMetadata | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._diff_text = diff_text
        self._diff_files: list[DiffFile] = []
        self._repo_dir = repo_dir
        self._title = title
        self._subtitle = subtitle
        self._iteration_meta = iteration_meta
        self._current_mode = DiffMode.SIDE_BY_SIDE

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="diff_main"):
            with Vertical(id="diff_tree_panel"):
                yield Static("Changed Files", id="tree_title")
                yield DiffTreeViewer(id="diff_tree_widget")

            with Vertical(id="diff_content_panel"):
                yield Static(self._render_mode_header(), id="content_title")
                yield DiffContentViewer(id="diff_viewer_widget")

        yield Footer()

    def on_mount(self) -> None:
        self._diff_files = parse_diff_files(self._diff_text)

        tree = self.query_one("#diff_tree_widget", DiffTreeViewer)
        tree.set_diff_files(self._diff_files)

        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.set_repo_dir(self._repo_dir)

        if self._iteration_meta:
            tree_title = self.query_one("#tree_title", Static)
            tree_title.update(self._render_tree_title())

        if self._diff_files:
            tree.focus()
        else:
            viewer.focus()

    @on(DiffTreeViewer.FileSelected)
    def _on_file_selected(self, event: DiffTreeViewer.FileSelected) -> None:
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.set_diff_file(event.diff_file)
        self._update_content_title(event.diff_file)

    @on(DiffTreeViewer.StatsChanged)
    def _on_stats_changed(self, event: DiffTreeViewer.StatsChanged) -> None:
        tree_title = self.query_one("#tree_title", Static)
        tree_title.update(self._render_tree_title(event))

    @on(DiffContentViewer.ModeChanged)
    def _on_mode_changed(self, event: DiffContentViewer.ModeChanged) -> None:
        self._current_mode = event.mode
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        if viewer._diff_file:
            self._update_content_title(viewer._diff_file)

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_toggle_mode(self) -> None:
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.action_toggle_mode()

    def action_set_mode_side_by_side(self) -> None:
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.mode = DiffMode.SIDE_BY_SIDE

    def action_set_mode_unified(self) -> None:
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.mode = DiffMode.UNIFIED

    def action_focus_tree(self) -> None:
        self.query_one("#diff_tree_widget", DiffTreeViewer).focus()

    def action_focus_content(self) -> None:
        self.query_one("#diff_viewer_widget", DiffContentViewer).focus()

    def _render_tree_title(self, stats: DiffTreeViewer.StatsChanged | None = None) -> Text:
        text = Text()
        meta = self._iteration_meta

        if meta:
            if meta.commit_hash:
                text.append("Diff (committed) ", style="bold green")
            else:
                text.append("Diff ", style="bold")
            text.append(f"Iteration {meta.iteration:03d}", style="cyan")
            if stats:
                text.append(f"  ({stats.total_files} files")
                if stats.total_added > 0:
                    text.append(f" +{stats.total_added}", style="green")
                if stats.total_deleted > 0:
                    text.append(f" -{stats.total_deleted}", style="red")
                text.append(")")
        else:
            text.append("Changed Files")
            if stats:
                text.append(f" ({stats.total_files})  ")
                if stats.total_added > 0:
                    text.append(f"+{stats.total_added}", style="green")
                    text.append(" ")
                if stats.total_deleted > 0:
                    text.append(f"-{stats.total_deleted}", style="red")

        return text

    def _render_mode_header(self) -> Text:
        text = Text()
        meta = self._iteration_meta

        if meta:
            if meta.commit_hash:
                text.append("Diff (committed) ", style="green")
            else:
                text.append("Diff ")
            text.append(f"Iteration {meta.iteration:03d}", style="bold cyan")
            if meta.commit_hash:
                text.append(f"  {meta.commit_hash[:7]}", style="dim")
            if meta.outcome:
                outcome_style = "green" if meta.outcome == "COMPLETED" else "yellow"
                text.append(f"  {meta.outcome}", style=outcome_style)
            text.append("  ")
        else:
            text.append("Diff  ")

        s_style = "accent" if self._current_mode == DiffMode.SIDE_BY_SIDE else "dim"
        text.append("[S]", style=s_style)
        text.append(" Side-by-side  ")
        u_style = "accent" if self._current_mode == DiffMode.UNIFIED else "dim"
        text.append("[U]", style=u_style)
        text.append(" Unified")
        return text

    def _update_content_title(self, diff_file: DiffFile | None) -> None:
        title = self.query_one("#content_title", Static)
        if not diff_file:
            title.update(self._render_mode_header())
            return

        text = Text()
        text.append(diff_file.display_name, style="bold")
        text.append("  ")
        if diff_file.added > 0:
            text.append(f"+{diff_file.added}", style="green")
            text.append(" ")
        if diff_file.deleted > 0:
            text.append(f"-{diff_file.deleted}", style="red")
        text.append("  ")
        s_style = "accent" if self._current_mode == DiffMode.SIDE_BY_SIDE else "dim"
        text.append("[S]", style=s_style)
        text.append(" Side-by-side  ")
        u_style = "accent" if self._current_mode == DiffMode.UNIFIED else "dim"
        text.append("[U]", style=u_style)
        text.append(" Unified")
        title.update(text)
