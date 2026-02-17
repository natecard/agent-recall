from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from rich.theme import Theme
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Log, OptionList, Static

from agent_recall.cli.tui.commands.palette_actions import _build_command_suggestions
from agent_recall.cli.tui.logic.activity_mixin import ActivityMixin
from agent_recall.cli.tui.logic.commands_mixin import CommandsMixin
from agent_recall.cli.tui.logic.ralph_mixin import RalphMixin
from agent_recall.cli.tui.logic.theme_mixin import ThemeMixin
from agent_recall.cli.tui.logic.worker_mixin import WorkerMixin
from agent_recall.cli.tui.types import (
    DiscoverModelsFn,
    ExecuteCommandFn,
    ListPrdItemsForPickerFn,
    ListSessionsForPickerFn,
    ThemeDefaultsFn,
    ThemeResolveFn,
    ThemeRuntimeFn,
)
from agent_recall.cli.tui.ui.bindings import TUI_BINDINGS
from agent_recall.cli.tui.ui.styles import APP_CSS
from agent_recall.cli.tui.views import DashboardPanels, build_dashboard_panels
from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext


class AgentRecallTextualApp(
    ActivityMixin,
    ThemeMixin,
    CommandsMixin,
    RalphMixin,
    WorkerMixin,
    App[None],
):
    CSS = APP_CSS
    BINDINGS = list(TUI_BINDINGS)

    def __init__(
        self,
        *,
        render_dashboard: Callable[..., Any],
        dashboard_context: DashboardRenderContext,
        execute_command: ExecuteCommandFn,
        list_sessions_for_picker: ListSessionsForPickerFn,
        list_prd_items_for_picker: ListPrdItemsForPickerFn | None = None,
        run_setup_payload: Callable[[dict[str, object]], tuple[bool, list[str]]],
        run_model_config: Callable[
            [str | None, str | None, str | None, float | None, int | None, bool],
            list[str],
        ],
        theme_defaults_provider: ThemeDefaultsFn,
        theme_runtime_provider: ThemeRuntimeFn | None = None,
        theme_resolve_provider: ThemeResolveFn | None = None,
        model_defaults_provider: Callable[[], dict[str, Any]],
        setup_defaults_provider: Callable[[], dict[str, Any]],
        discover_models: DiscoverModelsFn,
        providers: list[str],
        cli_commands: list[str] | None = None,
        rich_theme: Theme | None = None,
        initial_view: str = "overview",
        refresh_seconds: float = 2.0,
        all_cursor_workspaces: bool = False,
        onboarding_required: bool = False,
        terminal_panel_visible: bool = False,
        terminal_supported: bool = False,
    ):
        super().__init__()
        self._render_dashboard = render_dashboard
        self._dashboard_context = dashboard_context
        self._execute_command = execute_command
        self._list_sessions_for_picker = list_sessions_for_picker
        self._list_prd_items_for_picker = list_prd_items_for_picker
        self._run_setup_payload = run_setup_payload
        self._run_model_config = run_model_config
        self._theme_defaults_provider = theme_defaults_provider
        self._theme_runtime_provider = theme_runtime_provider
        self._theme_resolve_provider = theme_resolve_provider
        self._model_defaults_provider = model_defaults_provider
        self._setup_defaults_provider = setup_defaults_provider
        self._discover_models = discover_models
        self._providers = providers
        self._cli_commands = [
            command.strip() for command in (cli_commands or []) if command.strip()
        ]
        self._command_suggestions = _build_command_suggestions(self._cli_commands)
        self._rich_theme = rich_theme
        self._active_theme_name: str | None = None
        self.current_view = initial_view
        self.refresh_seconds = refresh_seconds
        self.all_cursor_workspaces = all_cursor_workspaces
        self.onboarding_required = onboarding_required
        self.terminal_panel_visible = terminal_panel_visible
        self.terminal_supported = terminal_supported
        self.status = "Ready. Press Ctrl+P for commands."
        self.activity: deque[str] = deque(maxlen=2000)
        self._theme_preview_active = False
        self._theme_commit_inflight = False
        self._theme_preview_origin: str | None = None
        self._result_list_open = False
        self._refresh_timer = None
        self._resize_refresh_timer = None
        self._worker_context: dict[int, str] = {}
        self._knowledge_run_workers: set[int] = set()
        self._pending_setup_payload: dict[str, Any] | None = None
        self._last_activity_render: tuple[str, str] | None = None
        self._debug_scroll_sample_count = 0
        self._activity_follow_tail = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Vertical(id="app_shell"):
                yield Vertical(id="dashboard")
                with Vertical(id="activity"):
                    yield Static(id="terminal_panel")
                    yield Log(id="activity_log", highlight=False, auto_scroll=False)
                    yield OptionList(id="activity_result_list")
        yield Footer()

    def on_mount(self) -> None:
        if self._theme_runtime_provider is not None:
            self._sync_runtime_theme()
        elif self._rich_theme is not None:
            self.console.push_theme(self._rich_theme)
            self._active_theme_name = "__initial__"
        self._configure_refresh_timer(self.refresh_seconds)
        self._append_activity("TUI ready. Press Ctrl+P for commands.")
        self._refresh_dashboard_panel()
        self._update_terminal_panel_visibility(initial=True)
        if self.onboarding_required:
            self.status = "Onboarding required"
            self._append_activity("Onboarding required. Opening setup wizard...")
            self.call_after_refresh(self.action_open_setup_modal)

    def on_unmount(self) -> None:
        self._teardown_runtime()

    def action_command_palette(self) -> None:
        self.action_open_command_palette()

    def action_request_quit(self) -> None:
        self.status = "Closing..."
        self._append_activity("Stopping background operations...")
        self._teardown_runtime()
        self.exit()

    def on_resize(self, event: events.Resize) -> None:
        _ = event
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
        # Debounce resize-driven refreshes to avoid event-loop saturation.
        self._resize_refresh_timer = self.set_timer(0.12, self._flush_resize_refresh)

    def _flush_resize_refresh(self) -> None:
        self._resize_refresh_timer = None
        self._refresh_dashboard_panel()

    def _configure_refresh_timer(self, refresh_seconds: float) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(refresh_seconds, self._refresh_dashboard_panel)

    def _teardown_runtime(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
            self._resize_refresh_timer = None
        self.workers.cancel_all()
        self._worker_context.clear()
        self._knowledge_run_workers.clear()

    def _refresh_dashboard_panel(self) -> None:
        self._sync_runtime_theme()
        panels = build_dashboard_panels(
            self._dashboard_context,
            all_cursor_workspaces=self.all_cursor_workspaces,
            include_banner_header=True,
            view=self.current_view,
            refresh_seconds=self.refresh_seconds,
            show_slash_console=False,
        )
        dashboard = self.query_one("#dashboard", Vertical)
        dashboard.remove_class("view-all")
        dashboard.remove_children()
        if panels.header is not None:
            header = Static(id="dashboard_header")
            header.update(panels.header)
            dashboard.mount(header)
        if self.current_view == "all":
            self._mount_all_view(dashboard, panels)
        elif self.current_view == "knowledge":
            dashboard.mount(Static(panels.knowledge, id="dashboard_knowledge"))
        elif self.current_view == "timeline":
            dashboard.mount(Static(panels.timeline, id="dashboard_timeline"))
        elif self.current_view == "llm":
            dashboard.mount(Static(panels.llm, id="dashboard_llm"))
        elif self.current_view == "sources":
            dashboard.mount(Static(panels.sources, id="dashboard_sources"))
        elif self.current_view == "settings":
            dashboard.mount(Static(panels.settings, id="dashboard_settings"))
        elif self.current_view == "console":
            pass
        else:
            overview_row = Horizontal(id="dashboard_overview_row")
            overview_row.mount(
                Static(panels.knowledge, id="dashboard_knowledge"),
                Static(panels.sources_compact, id="dashboard_sources"),
            )
            dashboard.mount(overview_row)
        self._refresh_activity_panel()

    def _mount_all_view(self, dashboard: Vertical, panels: DashboardPanels) -> None:
        dashboard.add_class("view-all")
        grid = Vertical(id="dashboard_all_grid")
        sidebar = Vertical(id="dashboard_all_sidebar")
        main = Vertical(id="dashboard_all_main")

        sidebar.mount(
            Static(panels.knowledge, id="dashboard_knowledge"),
            Static(panels.sources, id="dashboard_sources"),
            Static(panels.llm, id="dashboard_llm"),
            Static(panels.settings, id="dashboard_settings"),
        )
        main.mount(
            Static(panels.timeline, id="dashboard_timeline"),
        )

        grid.mount(sidebar, main)
        dashboard.mount(grid)
