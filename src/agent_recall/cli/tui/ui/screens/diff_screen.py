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

from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.cli.tui.utils.diff_parser import DiffFile, parse_diff_files
from agent_recall.cli.tui.widgets.diff_content import DiffContentViewer, DiffMode
from agent_recall.cli.tui.widgets.diff_tree import DiffTreeViewer
from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore


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
        layout: horizontal;
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
    }
    DiffScreen #diff_content_panel > Static#content_title {
        height: 1;
        padding: 0 1;
        border-bottom: solid $accent;
    }
    DiffScreen #diff_content_panel > DiffContentViewer {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("tab", "toggle_mode", "Toggle Mode"),
        Binding("s", "set_mode_side_by_side", "Side-by-Side"),
        Binding("u", "set_mode_unified", "Unified"),
        Binding("left,h", "focus_tree", "Tree"),
        Binding("right,l", "focus_content", "Content"),
        Binding("[", "prev_iteration", "Prev Iter", show=False),
        Binding("]", "next_iteration", "Next Iter", show=False),
    ]

    def __init__(
        self,
        diff_text: str,
        *,
        repo_dir: Path | None = None,
        agent_dir: Path | None = None,
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
        self._agent_dir = agent_dir
        self._title = title
        self._subtitle = subtitle
        self._iteration_meta = iteration_meta
        self._current_mode = DiffMode.SIDE_BY_SIDE
        # Iteration navigation state (populated on mount when agent_dir is set)
        self._all_reports: list[IterationReport] = []
        self._current_index: int = 0

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

        # Load iteration list for navigation when agent_dir is provided
        if self._agent_dir and self._iteration_meta:
            try:
                store = IterationReportStore(self._agent_dir / "ralph")
                self._all_reports = store.load_all()
                # Find index of the currently displayed iteration
                target = self._iteration_meta.iteration
                for i, r in enumerate(self._all_reports):
                    if r.iteration == target:
                        self._current_index = i
                        break
            except Exception:
                self._all_reports = []

        tree = self.query_one("#diff_tree_widget", DiffTreeViewer)
        tree.set_diff_files(self._diff_files)

        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.set_repo_dir(self._repo_dir)

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
        self.query_one("#tree_title", Static).update(self._render_tree_title(event))

    @on(DiffContentViewer.ModeChanged)
    def _on_mode_changed(self, event: DiffContentViewer.ModeChanged) -> None:
        self._current_mode = event.mode
        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        self._update_content_title(viewer._diff_file)

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_toggle_mode(self) -> None:
        self.query_one("#diff_viewer_widget", DiffContentViewer).action_toggle_mode()

    def action_set_mode_side_by_side(self) -> None:
        self.query_one("#diff_viewer_widget", DiffContentViewer).mode = DiffMode.SIDE_BY_SIDE

    def action_set_mode_unified(self) -> None:
        self.query_one("#diff_viewer_widget", DiffContentViewer).mode = DiffMode.UNIFIED

    def action_focus_tree(self) -> None:
        self.query_one("#diff_tree_widget", DiffTreeViewer).focus()

    def action_focus_content(self) -> None:
        self.query_one("#diff_viewer_widget", DiffContentViewer).focus()

    def action_prev_iteration(self) -> None:
        self._navigate_iteration(-1)

    def action_next_iteration(self) -> None:
        self._navigate_iteration(1)

    def _navigate_iteration(self, delta: int) -> None:
        if len(self._all_reports) < 2 or not self._agent_dir:
            return
        new_index = self._current_index + delta
        if not (0 <= new_index < len(self._all_reports)):
            return
        self._current_index = new_index
        report = self._all_reports[new_index]

        try:
            store = IterationReportStore(self._agent_dir / "ralph")
            self._diff_text = store.load_diff_for_iteration(report.iteration) or ""
        except Exception:
            self._diff_text = ""

        self._iteration_meta = IterationMetadata(
            iteration=report.iteration,
            item_id=report.item_id or None,
            item_title=report.item_title or None,
            commit_hash=report.commit_hash,
            completed_at=report.completed_at,
            outcome=report.outcome.value if report.outcome else None,
        )

        self._diff_files = parse_diff_files(self._diff_text)

        tree = self.query_one("#diff_tree_widget", DiffTreeViewer)
        tree.set_diff_files(self._diff_files)

        viewer = self.query_one("#diff_viewer_widget", DiffContentViewer)
        viewer.set_diff_file(None)

        self.query_one("#tree_title", Static).update(self._render_tree_title())
        self.query_one("#content_title", Static).update(self._render_mode_header())

        tree.focus()

    def _nav_label(self) -> str:
        if len(self._all_reports) < 2:
            return ""
        return f"  [{self._current_index + 1}/{len(self._all_reports)}]  ← [ / ] →"

    def _render_tree_title(self, stats: DiffTreeViewer.StatsChanged | None = None) -> Text:
        text = Text()
        meta = self._iteration_meta

        if meta:
            if meta.commit_hash:
                text.append("Diff (committed) ", style="bold green")
            else:
                text.append("Diff ", style="bold")
            text.append(f"#{meta.iteration:03d}", style="cyan")
            nav = self._nav_label()
            if nav:
                text.append(nav, style="dim")
            if stats:
                text.append(f"  {stats.total_files}f")
                if stats.total_added > 0:
                    text.append(f" +{stats.total_added}", style="green")
                if stats.total_deleted > 0:
                    text.append(f" -{stats.total_deleted}", style="red")
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
                text.append("committed ", style="green")
            text.append(f"#{meta.iteration:03d}", style="bold cyan")
            if meta.commit_hash:
                text.append(f"  {meta.commit_hash[:7]}", style="dim")
            if meta.outcome:
                outcome_style = "green" if meta.outcome == "COMPLETED" else "yellow"
                text.append(f"  {meta.outcome}", style=outcome_style)
            if meta.item_title:
                text.append(f"  {meta.item_title}", style="dim")
            text.append("   ")
        else:
            text.append("Diff   ")

        accent_style = self._get_accent_style()
        s_style = accent_style if self._current_mode == DiffMode.SIDE_BY_SIDE else "dim"
        text.append("[S]", style=s_style)
        text.append(" Side-by-side  ")
        u_style = accent_style if self._current_mode == DiffMode.UNIFIED else "dim"
        text.append("[U]", style=u_style)
        text.append(" Unified")
        return text

    def _get_accent_style(self) -> str:
        """Get 'bold {accent}' from the current theme."""
        try:
            theme_name = getattr(self.app, "_active_theme_name", None) or DEFAULT_THEME
            if theme_name == "__initial__":
                theme_name = DEFAULT_THEME
            colors = ThemeManager.get_theme_colors(theme_name)
            accent = colors.get("accent", "cyan")
            return f"bold {accent}"
        except Exception:
            return "bold cyan"

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
        text.append("   ")
        accent_style = self._get_accent_style()
        s_style = accent_style if self._current_mode == DiffMode.SIDE_BY_SIDE else "dim"
        text.append("[S]", style=s_style)
        text.append(" Side-by-side  ")
        u_style = accent_style if self._current_mode == DiffMode.UNIFIED else "dim"
        text.append("[U]", style=u_style)
        text.append(" Unified")
        title.update(text)
