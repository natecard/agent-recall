from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console, Group

from agent_recall.cli.tui.views import (
    DashboardRenderContext,
    build_dashboard_panels,
    build_tui_dashboard,
)


class _ThemeManagerStub:
    def get_theme_name(self) -> str:
        return "dark+"


class _StorageStub:
    def get_stats(self) -> dict[str, int]:
        return {
            "processed_sessions": 7,
            "log_entries": 42,
            "chunks": 120,
        }

    def get_last_processed_at(self) -> datetime:
        return datetime(2026, 2, 1, 12, 0, tzinfo=UTC)


class _FilesStub:
    def read_tier(self, _tier: object) -> str:
        return "content"

    def read_config(self) -> dict[str, object]:
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-test",
                "temperature": 0.2,
                "max_tokens": 1024,
                "base_url": None,
            }
        }


class _IngesterStub:
    def __init__(self, source_name: str, sessions: int) -> None:
        self.source_name = source_name
        self._sessions = sessions

    def discover_sessions(self) -> list[object]:
        return [object() for _ in range(self._sessions)]


@dataclass
class _CostSummary:
    total_tokens: int
    total_cost_usd: float


def _context() -> DashboardRenderContext:
    return DashboardRenderContext(
        console=Console(width=120, record=True),
        theme_manager=_ThemeManagerStub(),
        agent_dir=Path.cwd() / ".agent",
        ralph_max_iterations=10,
        get_storage=lambda: _StorageStub(),
        get_files=lambda: _FilesStub(),
        get_repo_selected_sources=lambda _files: ["cursor", "claude"],
        resolve_repo_root_for_display=lambda: Path("/tmp/repo"),
        filter_ingesters_by_sources=lambda ingesters, _selected: ingesters,
        get_default_ingesters=lambda **_kwargs: [
            _IngesterStub("cursor", 2),
            _IngesterStub("claude", 1),
        ],
        render_iteration_timeline=lambda _store, max_entries: ["line"] * max_entries,
        summarize_costs=lambda _reports: _CostSummary(total_tokens=321, total_cost_usd=1.23),
        format_usd=lambda amount: f"${amount:.2f}",
        is_interactive_terminal=lambda: True,
        help_lines_provider=lambda: ["/help", "/quit"],
    )


def test_build_dashboard_view_panel_counts() -> None:
    expected_counts = {
        "overview": 3,
        "knowledge": 3,
        "sources": 3,
        "llm": 3,
        "settings": 3,
        "timeline": 3,
        "ralph": 3,
        "console": 2,
        "all": 5,
    }

    for view, expected in expected_counts.items():
        group = build_tui_dashboard(
            _context(),
            all_cursor_workspaces=False,
            include_banner_header=True,
            view=view,
            show_slash_console=True,
        )
        assert isinstance(group, Group)
        assert len(group.renderables) == expected


def test_build_dashboard_panels_for_all_view() -> None:
    panels = build_dashboard_panels(
        _context(),
        all_cursor_workspaces=False,
        include_banner_header=True,
        view="all",
        show_slash_console=False,
    )
    assert panels.header is not None
    assert panels.knowledge.title == "Knowledge Base"
    assert panels.sources.title == "Session Sources"
    assert panels.settings.title == "Settings"
    assert panels.timeline.title == "Iteration Timeline"
    assert panels.llm.title == "LLM Configuration"


def test_build_dashboard_without_slash_console() -> None:
    group = build_tui_dashboard(
        _context(),
        include_banner_header=True,
        view="overview",
        show_slash_console=False,
    )
    assert isinstance(group, Group)
    assert len(group.renderables) == 2
