from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console


@dataclass(frozen=True)
class DashboardRenderContext:
    console: Console
    theme_manager: Any
    agent_dir: Path
    ralph_max_iterations: int | None
    get_storage: Callable[[], Any]
    get_files: Callable[[], Any]
    get_repo_selected_sources: Callable[[Any], list[str] | None]
    resolve_repo_root_for_display: Callable[[], Path]
    filter_ingesters_by_sources: Callable[[Any, list[str] | None], Any]
    get_default_ingesters: Callable[..., Any]
    render_iteration_timeline: Callable[..., list[str]]
    summarize_costs: Callable[[Any], Any]
    format_usd: Callable[[float], str]
    is_interactive_terminal: Callable[[], bool]
    help_lines_provider: Callable[[], list[str]]
