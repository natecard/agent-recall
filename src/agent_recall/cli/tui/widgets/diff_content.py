from __future__ import annotations

import difflib
import subprocess
from enum import StrEnum
from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from agent_recall.cli.tui.delta import get_delta_path
from agent_recall.cli.tui.utils.diff_parser import DiffFile, get_hunk_diff_text


class DiffMode(StrEnum):
    UNIFIED = "unified"
    SIDE_BY_SIDE = "side-by-side"


class DiffContentViewer(Vertical):
    DEFAULT_CSS = """
    DiffContentViewer {
        height: 1fr;
        width: 1fr;
        overflow: hidden;
    }
    DiffContentViewer > Static#diff_header {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        border-bottom: solid $accent;
    }
    DiffContentViewer > Vertical#diff_scroll {
        height: 1fr;
        overflow: auto;
    }
    DiffContentViewer > Vertical#diff_scroll > Static {
        height: auto;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("tab", "toggle_mode", "Toggle Mode"),
        Binding("s", "set_mode_side_by_side", "Side-by-Side"),
        Binding("u", "set_mode_unified", "Unified"),
    ]

    mode: reactive[DiffMode] = reactive(DiffMode.SIDE_BY_SIDE, init=False)

    class ModeChanged(Message):
        def __init__(self, mode: DiffMode) -> None:
            super().__init__()
            self.mode = mode

    def __init__(
        self,
        *,
        repo_dir: Path | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._diff_file: DiffFile | None = None
        self._repo_dir = repo_dir
        self._old_content: str | None = None
        self._new_content: str | None = None

    def set_diff_file(self, diff_file: DiffFile | None) -> None:
        self._diff_file = diff_file
        self._load_file_contents()
        self._render_diff()

    def set_repo_dir(self, repo_dir: Path | None) -> None:
        self._repo_dir = repo_dir

    def compose(self) -> ComposeResult:
        yield Static("", id="diff_header")
        with Vertical(id="diff_scroll"):
            yield Static("", id="diff_content")

    def watch_mode(self, old_mode: DiffMode, new_mode: DiffMode) -> None:
        if old_mode != new_mode:
            self._render_diff()
            self.post_message(self.ModeChanged(new_mode))

    def action_toggle_mode(self) -> None:
        if self.mode == DiffMode.UNIFIED:
            self.mode = DiffMode.SIDE_BY_SIDE
        else:
            self.mode = DiffMode.UNIFIED

    def action_set_mode_side_by_side(self) -> None:
        self.mode = DiffMode.SIDE_BY_SIDE

    def action_set_mode_unified(self) -> None:
        self.mode = DiffMode.UNIFIED

    def _load_file_contents(self) -> None:
        self._old_content = None
        self._new_content = None

        if not self._diff_file or not self._repo_dir:
            return

        old_path = self._diff_file.old_path
        new_path = self._diff_file.new_path

        if old_path and old_path != "/dev/null":
            try:
                result = subprocess.run(
                    ["git", "show", f"HEAD:{old_path}"],
                    cwd=self._repo_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if result.returncode == 0:
                    self._old_content = result.stdout
            except (OSError, subprocess.TimeoutExpired):
                pass

        if new_path and new_path != "/dev/null":
            try:
                file_path = self._repo_dir / new_path
                if file_path.exists():
                    self._new_content = file_path.read_text()
            except OSError:
                pass

    def _render_diff(self) -> None:
        header = self.query_one("#diff_header", Static)
        content = self.query_one("#diff_content", Static)

        if not self._diff_file:
            header.update("[dim]Select a file to view diff[/dim]")
            content.update("")
            return

        header_text = self._render_header()
        header.update(header_text)

        if self.mode == DiffMode.SIDE_BY_SIDE and self._can_show_side_by_side():
            content.update(self._render_side_by_side())
        else:
            content.update(self._render_unified())

    def _can_show_side_by_side(self) -> bool:
        return bool(self._old_content or self._new_content)

    def _render_header(self) -> Text:
        text = Text()
        if self._diff_file:
            text.append(self._diff_file.display_name, style="bold")
            text.append("  ")
            if self._diff_file.added > 0:
                text.append(f"+{self._diff_file.added}", style="green")
                text.append(" ")
            if self._diff_file.deleted > 0:
                text.append(f"-{self._diff_file.deleted}", style="red")
            text.append("  ")
            mode_style = "accent" if self.mode == DiffMode.SIDE_BY_SIDE else "dim"
            text.append("[S]", style=mode_style)
            text.append(" Side-by-side  ")
            mode_style = "accent" if self.mode == DiffMode.UNIFIED else "dim"
            text.append("[U]", style=mode_style)
            text.append(" Unified")
        return text

    def _render_with_delta(self, diff_text: str, side_by_side: bool = False) -> str | None:
        """Render diff with delta binary. Returns None on failure."""
        delta_path = get_delta_path()
        if not delta_path or not diff_text:
            return None
        try:
            args = [str(delta_path)]
            if side_by_side:
                args.append("--side-by-side")
            result = subprocess.run(
                args,
                input=diff_text,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def _render_unified(self) -> str | Syntax:
        diff_text = get_hunk_diff_text(self._diff_file) if self._diff_file else ""
        delta_output = self._render_with_delta(diff_text, side_by_side=False)
        if delta_output is not None:
            return delta_output
        return Syntax(
            diff_text,
            lexer="diff",
            word_wrap=False,
            theme="monokai",
        )

    def _render_side_by_side(self) -> str:
        diff_text = get_hunk_diff_text(self._diff_file) if self._diff_file else ""
        delta_output = self._render_with_delta(diff_text, side_by_side=True)
        if delta_output is not None:
            return delta_output
        return self._render_stdlib_side_by_side()

    def _render_stdlib_side_by_side(self) -> str:
        old_lines = (self._old_content or "").splitlines()
        new_lines = (self._new_content or "").splitlines()

        if not old_lines and not new_lines:
            return ""

        col_width = 80
        sep = " │ "
        output_lines: list[str] = []

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        groups = list(matcher.get_grouped_opcodes(3))

        for group in groups:
            if output_lines:
                output_lines.append("─" * (col_width + len(sep) + col_width // 2))
            for tag, i1, i2, j1, j2 in group:
                old_chunk = old_lines[i1:i2]
                new_chunk = new_lines[j1:j2]
                n = max(len(old_chunk), len(new_chunk))
                for k in range(n):
                    left = old_chunk[k] if k < len(old_chunk) else ""
                    right = new_chunk[k] if k < len(new_chunk) else ""
                    if tag == "equal":
                        marker_l = marker_r = " "
                    elif tag == "replace":
                        marker_l, marker_r = "-", "+"
                    elif tag == "delete":
                        marker_l, marker_r = "-", " "
                    else:  # insert
                        marker_l, marker_r = " ", "+"
                    left_col = (marker_l + " " + left)[:col_width].ljust(col_width)
                    right_col = marker_r + " " + right
                    output_lines.append(left_col + sep + right_col)

        return "\n".join(output_lines) if output_lines else "(no differences)"
