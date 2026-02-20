from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table

from agent_recall.ralph.iteration_store import IterationReportStore


@dataclass(frozen=True)
class DiffSummaryWidget:
    agent_dir: Path

    def render(self) -> Panel:
        diff_text = self._load_latest_diff()
        if not diff_text:
            return self._render_placeholder()
        total_added, total_deleted, per_file = parse_diff_stats(diff_text)
        return self._render_panel(total_added, total_deleted, per_file)

    def _load_latest_diff(self) -> str | None:
        store = IterationReportStore(self.agent_dir / "ralph")
        reports = store.load_recent(count=1)
        if not reports:
            return None
        latest = reports[0]
        return store.load_diff_for_iteration(latest.iteration)

    def _render_placeholder(self) -> Panel:
        return Panel(
            "[dim]No diff available — run Ralph loop to see changes[/dim]",
            title="Diff Summary",
            border_style="accent",
        )

    def _render_panel(
        self,
        total_added: int,
        total_deleted: int,
        per_file: list[tuple[str, int, int]],
    ) -> Panel:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("File", overflow="fold")
        table.add_column("Added", justify="right", width=6)
        table.add_column("Deleted", justify="right", width=6)

        for filename, added, deleted in per_file:
            truncated = self._truncate_filename(filename)
            file_signifier = "[dim cyan][F][/dim cyan]"
            added_str = f"[success]+{added}[/success]" if added > 0 else ""
            deleted_str = f"[error]-{deleted}[/error]" if deleted > 0 else ""
            table.add_row(f"{file_signifier} {truncated}", added_str, deleted_str)

        header = (
            f"[bold][success]+{total_added}[/success]  "
            f"[error]-{total_deleted}[/error]  "
            f"{len(per_file)} files changed[/bold]"
        )

        store = IterationReportStore(self.agent_dir / "ralph")
        reports = store.load_recent(count=1)
        footer = ""
        if reports:
            latest = reports[0]
            footer = f"[dim]Iteration {latest.iteration} · {latest.item_id}[/dim]"

        content_lines = [header]
        if table.row_count > 0:
            content_lines.append(table)
        if footer:
            content_lines.append(footer)

        from rich.console import Group

        return Panel(
            Group(*content_lines),
            title="Diff Summary",
            border_style="accent",
        )

    @staticmethod
    def _truncate_filename(filename: str, max_len: int = 40) -> str:
        if len(filename) <= max_len:
            return filename
        return "…" + filename[-(max_len - 1) :]


def parse_diff_stats(
    diff_text: str,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    total_added = 0
    total_deleted = 0
    file_stats: dict[str, tuple[int, int]] = {}

    current_file: str | None = None
    file_added = 0
    file_deleted = 0

    for line in diff_text.splitlines():
        file_match = re.match(r"^diff --git a/(.+?) b/.+$", line)
        if file_match:
            if current_file is not None:
                file_stats[current_file] = (file_added, file_deleted)
            current_file = file_match.group(1)
            file_added = 0
            file_deleted = 0
            continue

        header_match = re.match(r"^--- (?:a/)?(.+)$", line)
        if header_match and current_file is None:
            candidate = header_match.group(1)
            if candidate != "/dev/null":
                current_file = candidate
                file_added = 0
                file_deleted = 0
            continue

        if line.startswith("+++"):
            continue
        if line.startswith("---"):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            total_added += 1
            if current_file is not None:
                file_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            total_deleted += 1
            if current_file is not None:
                file_deleted += 1

    if current_file is not None:
        file_stats[current_file] = (file_added, file_deleted)

    per_file: list[tuple[str, int, int]] = [
        (filename, stats[0], stats[1]) for filename, stats in file_stats.items()
    ]

    return total_added, total_deleted, per_file
