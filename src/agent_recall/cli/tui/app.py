from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

from rich.theme import Theme
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, Log, OptionList, Static
from textual.widgets.option_list import Option

from agent_recall.cli.tui.commands.help_text import (
    filter_command_suggestions,
    get_all_cli_commands,
)
from agent_recall.cli.tui.commands.palette_actions import _build_command_suggestions
from agent_recall.cli.tui.commands.palette_router import handle_palette_action
from agent_recall.cli.tui.logic.activity_mixin import ActivityMixin
from agent_recall.cli.tui.logic.commands_mixin import CommandsMixin
from agent_recall.cli.tui.logic.ralph_mixin import RalphMixin
from agent_recall.cli.tui.logic.theme_mixin import ThemeMixin
from agent_recall.cli.tui.logic.worker_mixin import WorkerMixin
from agent_recall.cli.tui.types import (
    DiscoverCodingModelsFn,
    DiscoverModelsFn,
    ExecuteCommandFn,
    ListPrdItemsForPickerFn,
    ListSessionsForPickerFn,
    ThemeDefaultsFn,
    ThemeResolveFn,
    ThemeRuntimeFn,
)
from agent_recall.cli.tui.ui.bindings import TUI_BINDINGS
from agent_recall.cli.tui.ui.modals.iteration_detail import IterationDetailModal
from agent_recall.cli.tui.ui.modals.sessions_view import SessionsViewModal
from agent_recall.cli.tui.ui.styles import APP_CSS
from agent_recall.cli.tui.views import DashboardPanels, build_dashboard_panels, build_sources_data
from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext
from agent_recall.cli.tui.widgets import InteractiveSourcesWidget, InteractiveTimelineWidget
from agent_recall.ralph.iteration_store import IterationReport, IterationReportStore
from agent_recall.storage.files import FileStorage


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
        get_sources_and_sessions_for_tui: Callable[
            [int, bool], tuple[list[dict[str, object]], list[dict[str, object]]]
        ],
        get_session_detail_for_tui: Callable[[str, str, bool], Any | None] | None = None,
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
        discover_coding_models: DiscoverCodingModelsFn,
        providers: list[str],
        cli_commands: list[str] | None = None,
        rich_theme: Theme | None = None,
        initial_view: str = "overview",
        all_cursor_workspaces: bool = False,
        ralph_agent_transport: str = "pipe",
        onboarding_required: bool = False,
        terminal_panel_visible: bool = False,
        terminal_supported: bool = False,
        no_delta_setup: bool = False,
    ):
        super().__init__()
        self._render_dashboard = render_dashboard
        self._dashboard_context = dashboard_context
        self._execute_command = execute_command
        self._list_sessions_for_picker = list_sessions_for_picker
        self._get_sources_and_sessions_for_tui = get_sources_and_sessions_for_tui
        self._get_session_detail_for_tui = get_session_detail_for_tui
        self._list_prd_items_for_picker = list_prd_items_for_picker
        self._run_setup_payload = run_setup_payload
        self._run_model_config = run_model_config
        self._theme_defaults_provider = theme_defaults_provider
        self._theme_runtime_provider = theme_runtime_provider
        self._theme_resolve_provider = theme_resolve_provider
        self._model_defaults_provider = model_defaults_provider
        self._setup_defaults_provider = setup_defaults_provider
        self._discover_models = discover_models
        self._discover_coding_models = discover_coding_models
        self._providers = providers
        self._cli_commands = [
            command.strip() for command in (cli_commands or []) if command.strip()
        ]
        self._command_suggestions = _build_command_suggestions(self._cli_commands)
        self._rich_theme = rich_theme
        self._active_theme_name: str | None = None
        self.current_view = initial_view
        self.all_cursor_workspaces = all_cursor_workspaces
        normalized_transport = str(ralph_agent_transport).strip().lower()
        self.ralph_agent_transport = (
            normalized_transport if normalized_transport in {"pipe", "auto", "pty"} else "pipe"
        )
        self.onboarding_required = onboarding_required
        self.terminal_panel_visible = terminal_panel_visible
        self.terminal_supported = terminal_supported
        self.no_delta_setup = no_delta_setup
        self.status = "Ready. Press Ctrl+P for commands."
        self.activity: deque[str] = deque(maxlen=2000)
        self._theme_preview_active = False
        self._theme_commit_inflight = False
        self._theme_preview_origin: str | None = None
        self._result_list_open = False
        self._resize_refresh_timer = None
        self._worker_context: dict[int, str] = {}
        self._knowledge_run_workers: set[int] = set()
        self._pending_setup_payload: dict[str, Any] | None = None
        self._last_activity_render: tuple[str, str] | None = None
        self._debug_scroll_sample_count = 0
        self._activity_follow_tail = True
        self._dashboard_refresh_generation = 0
        self._dashboard_layout_view: str | None = None
        self._cli_commands_cache: list[str] = get_all_cli_commands()
        self._suggestions_visible = False
        self._highlighted_suggestion_index: int | None = None
        self._interactive_sources_widget: InteractiveSourcesWidget | None = None
        self._interactive_timeline_widget: InteractiveTimelineWidget | None = None
        self.tui_widget_visibility: dict[str, bool] = {}
        self.tui_banner_size: str = "normal"
        self._last_layout_signature: tuple[str, tuple[tuple[str, bool], ...]] | None = None
        self._last_banner_classes: str | None = None
        self._last_header_signature: tuple[bool, bool] | None = None
        self._layout_module: object | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="app_shell"):
                yield Vertical(id="dashboard")
                with Vertical(id="cli_input_container"):
                    yield OptionList(id="cli_suggestions")
                    yield Input(id="cli_input", placeholder="/ command  ·  Ctrl+P for palette")
                with Vertical(id="activity"):
                    yield Static(id="terminal_panel")
                    yield Log(id="activity_log", highlight=False, auto_scroll=False)
                    yield OptionList(id="activity_result_list")
        yield Footer()

    def on_mount(self) -> None:
        self._load_tui_layout_settings()
        if not self.no_delta_setup:
            from agent_recall.cli.tui.delta import is_delta_available, is_delta_setup_declined
            from agent_recall.cli.tui.ui.screens.first_launch import FirstLaunchScreen

            if not is_delta_available() and not is_delta_setup_declined():
                self.push_screen(FirstLaunchScreen())
        if self._theme_runtime_provider is not None:
            self._sync_runtime_theme()
        elif self._rich_theme is not None:
            self.console.push_theme(self._rich_theme)
            self._active_theme_name = "__initial__"
        self._append_activity("TUI ready. Press Ctrl+P for commands.")
        self._refresh_dashboard_panel()
        self._update_terminal_panel_visibility(initial=True)
        self._sync_activity_layout()
        if self.onboarding_required:
            self.status = "Onboarding required"
            self._append_activity("Onboarding required. Opening setup wizard...")
            self.call_after_refresh(self.action_open_setup_modal)
        self.call_after_refresh(lambda: self._check_responsive(self.size.width))

    def on_unmount(self) -> None:
        self._teardown_runtime()

    def action_command_palette(self) -> None:
        self._hide_suggestions()
        self.action_open_command_palette()

    def action_request_quit(self) -> None:
        self.status = "Closing..."
        self._append_activity("Stopping background operations...")
        self._teardown_runtime()
        self.exit()

    def on_resize(self, event: events.Resize) -> None:
        self._check_responsive(event.size.width)
        _ = event
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
        # Debounce resize-driven refreshes to avoid event-loop saturation.
        self._resize_refresh_timer = self.set_timer(0.12, self._flush_resize_refresh)

    def _flush_resize_refresh(self) -> None:
        self._resize_refresh_timer = None
        self._refresh_dashboard_panel()

    def _check_responsive(self, width: int) -> None:
        try:
            dashboard = self.query_one("#dashboard")
            if width < 120:
                dashboard.add_class("narrow")
            else:
                dashboard.remove_class("narrow")
        except Exception:
            pass

    def _sync_activity_layout(self) -> None:
        """Adjust activity panel sizing based on current view and terminal visibility."""
        try:
            activity_panel = self.query_one("#activity", Vertical)
            dashboard = self.query_one("#dashboard", Vertical)
        except Exception:
            return

        console_focused = self.current_view == "console"
        ralph_focused = self.current_view == "ralph"
        terminal_visible = bool(self.terminal_supported and self.terminal_panel_visible)

        if console_focused:
            activity_panel.add_class("console-focused")
            dashboard.add_class("console-focused")
        else:
            activity_panel.remove_class("console-focused")
            dashboard.remove_class("console-focused")

        if terminal_visible and not console_focused:
            activity_panel.add_class("terminal-expanded")
        else:
            activity_panel.remove_class("terminal-expanded")

        if ralph_focused and not console_focused:
            activity_panel.add_class("ralph-focused")
        else:
            activity_panel.remove_class("ralph-focused")

    def _teardown_runtime(self) -> None:
        if self._resize_refresh_timer is not None:
            self._resize_refresh_timer.stop()
            self._resize_refresh_timer = None
        self.workers.cancel_all()
        self._worker_context.clear()
        self._knowledge_run_workers.clear()

    def _refresh_dashboard_panel(self) -> None:
        self._dashboard_refresh_generation += 1
        refresh_generation = self._dashboard_refresh_generation
        self._sync_runtime_theme()
        layout_signature = (
            self.tui_banner_size,
            tuple(sorted(self.tui_widget_visibility.items())),
        )
        panels = build_dashboard_panels(
            self._dashboard_context,
            all_cursor_workspaces=self.all_cursor_workspaces,
            include_banner_header=self.tui_banner_size != "hidden",
            banner_size=self.tui_banner_size,
            view=self.current_view,
            ralph_agent_transport=self.ralph_agent_transport,
            show_slash_console=False,
            widget_visibility=self.tui_widget_visibility,
        )
        dashboard = self.query_one("#dashboard", Vertical)
        if panels.source_names:
            self._refresh_source_actions(panels.source_names)

        if (
            self._dashboard_layout_view == self.current_view
            and self._last_layout_signature == layout_signature
            and self._update_dashboard_widgets(panels)
        ):
            self._sync_activity_layout()
            self._refresh_activity_panel()
            return

        await_remove = dashboard.remove_children()

        async def _mount_after_prune() -> None:
            await await_remove
            if refresh_generation != self._dashboard_refresh_generation:
                return
            self._apply_dashboard_view_class(dashboard)
            self._mount_dashboard_widgets(dashboard, panels)
            self._dashboard_layout_view = self.current_view
            self._last_layout_signature = layout_signature
            self._sync_activity_layout()
            self._refresh_activity_panel()

        self.call_next(_mount_after_prune)

    def _apply_dashboard_view_class(self, dashboard: Vertical) -> None:
        if self.current_view == "all":
            dashboard.add_class("view-all")
        else:
            dashboard.remove_class("view-all")

    def _build_view_detail_panel(self, view: str) -> Any:
        detail_panels = build_dashboard_panels(
            self._dashboard_context,
            all_cursor_workspaces=self.all_cursor_workspaces,
            include_banner_header=True,
            banner_size=self.tui_banner_size,
            view=view,
            ralph_agent_transport=self.ralph_agent_transport,
            show_slash_console=False,
            widget_visibility=self.tui_widget_visibility,
        )
        if view == "knowledge":
            return detail_panels.knowledge
        return detail_panels.timeline

    def _build_interactive_sources_widget(self) -> InteractiveSourcesWidget:
        """Build the interactive sources widget with sync buttons."""
        sources_data, last_synced = build_sources_data(
            self._dashboard_context,
            all_cursor_workspaces=self.all_cursor_workspaces,
        )
        return InteractiveSourcesWidget(
            sources=sources_data,
            on_sync=self._handle_source_sync_click,
            last_synced=last_synced,
            id="dashboard_sources_interactive",
        )

    def _build_interactive_timeline_widget(self) -> InteractiveTimelineWidget:
        report_store = IterationReportStore(self._dashboard_context.agent_dir / "ralph")
        return InteractiveTimelineWidget(
            report_store,
            on_select=self._open_iteration_detail,
            id="dashboard_timeline_interactive",
        )

    def _open_iteration_detail(self, report: IterationReport) -> None:
        report_store = IterationReportStore(self._dashboard_context.agent_dir / "ralph")
        diff_text = report_store.load_diff_for_iteration(report.iteration) or ""
        item_label = " ".join(part for part in [report.item_id, report.item_title] if part)
        summary_text = report.summary or "No summary available."
        outcome_text = report.outcome.value if report.outcome else "Unknown"
        self.push_screen(
            IterationDetailModal(
                title=f"Iteration {report.iteration} Detail",
                summary_text=summary_text,
                outcome_text=outcome_text,
                item_text=item_label or "Unknown item",
                diff_text=diff_text,
            )
        )

    def _handle_source_sync_click(self, source_name: str) -> None:
        """Handle sync button click from interactive sources widget."""
        self._run_source_sync(source_name)

    def on_sessions_view_modal_source_sync_requested(
        self, event: SessionsViewModal.SourceSyncRequested
    ) -> None:
        """Handle source sync request from SessionsViewModal."""
        self._run_source_sync(event.source_name)

    def _mount_dashboard_widgets(self, dashboard: Vertical, panels: DashboardPanels) -> None:
        if panels.header is not None:
            classes = f"banner-{self.tui_banner_size}"
            dashboard.mount(Static(panels.header, id="dashboard_header", classes=classes))
            self._last_banner_classes = classes
            self._last_header_signature = (
                self._dashboard_context.ralph_enabled,
                self._dashboard_context.ralph_running,
            )
        elif self.tui_banner_size == "compact":
            dashboard.mount(Static("AGENT RECALL", id="dashboard_header", classes="banner-compact"))
            self._last_banner_classes = "banner-compact"
        if self.current_view == "all":
            self._mount_all_view(dashboard, panels)
        elif self.current_view == "knowledge":
            if self._is_widget_visible("knowledge"):
                dashboard.mount(
                    Static(
                        self._build_view_detail_panel("knowledge"),
                        id="dashboard_knowledge",
                    )
                )
        elif self.current_view == "timeline":
            if self._is_widget_visible("timeline"):
                self._interactive_timeline_widget = self._build_interactive_timeline_widget()
                dashboard.mount(self._interactive_timeline_widget)
        elif self.current_view == "ralph":
            if self._is_widget_visible("ralph"):
                dashboard.mount(Static(panels.ralph, id="dashboard_ralph"))
        elif self.current_view == "settings":
            if self._is_widget_visible("settings"):
                dashboard.mount(Static(panels.settings, id="dashboard_settings"))
        elif self.current_view == "console":
            pass
        elif self.current_view == "overview":
            overview_row = self._build_overview_row(panels)
            if overview_row is not None:
                dashboard.mount(overview_row)
        else:
            overview_row = self._build_overview_row(panels)
            if overview_row is not None:
                dashboard.mount(overview_row)

    def _update_static_widget(self, selector: str, renderable: Any) -> bool:
        try:
            widget = self.query_one(selector, Static)
        except Exception:
            return False
        widget.update(renderable)
        return True

    def _update_dashboard_widgets(self, panels: DashboardPanels) -> bool:
        if panels.header is None:
            if self.tui_banner_size in {"compact", "large"}:
                return False
        else:
            try:
                header_widget = self.query_one("#dashboard_header", Static)
                header_signature = (
                    self._dashboard_context.ralph_enabled,
                    self._dashboard_context.ralph_running,
                )
                if self._last_header_signature != header_signature:
                    header_widget.update(panels.header)
                    self._last_header_signature = header_signature
                new_classes = f"banner-{self.tui_banner_size}"
                if self._last_banner_classes != new_classes:
                    header_widget.classes = new_classes
                    self._last_banner_classes = new_classes
            except Exception:
                return False
        if self.current_view == "all":
            return (
                self._update_static_widget("#dashboard_knowledge", panels.knowledge)
                and self._update_static_widget("#dashboard_sources", panels.sources)
                and self._update_static_widget("#dashboard_settings", panels.settings)
                and self._update_static_widget("#dashboard_timeline", panels.timeline)
            )
        if self.current_view == "knowledge":
            if not self._is_widget_visible("knowledge"):
                return False
            return self._update_static_widget(
                "#dashboard_knowledge",
                self._build_view_detail_panel("knowledge"),
            )
        if self.current_view == "timeline":
            if not self._is_widget_visible("timeline"):
                return False
            try:
                widget = self.query_one(
                    "#dashboard_timeline_interactive", InteractiveTimelineWidget
                )
                widget.refresh_timeline()
                return True
            except Exception:
                return False
        if self.current_view == "ralph":
            if not self._is_widget_visible("ralph"):
                return False
            return self._update_static_widget("#dashboard_ralph", panels.ralph)
        if self.current_view == "settings":
            if not self._is_widget_visible("settings"):
                return False
            return self._update_static_widget("#dashboard_settings", panels.settings)
        if self.current_view == "console":
            return True
        if self.current_view == "overview":
            if not self._is_widget_visible("knowledge") and not self._is_widget_visible("sources"):
                return False
            return (
                not self._is_widget_visible("knowledge")
                or self._update_static_widget("#dashboard_knowledge", panels.knowledge)
            ) and (
                not self._is_widget_visible("sources")
                or self._update_static_widget("#dashboard_sources", panels.sources_compact)
            )
        return False

    def _refresh_source_actions(self, source_names: list[str]) -> None:
        if self.current_view != "all":
            return
        if not source_names:
            return
        action_map: dict[str, str] = {}
        for source_name in source_names:
            if not source_name:
                continue
            action_id = f"sync-source:{source_name}"
            action_label = f"Sync source: {source_name}"
            action_map[action_id] = action_label
        if not action_map:
            return
        picker = self.query_one("#activity_result_list", OptionList)
        if self._result_list_open:
            existing_ids = [option.id for option in picker.options if option.id]
            if existing_ids and all(
                id_value.startswith("sync-source:") for id_value in existing_ids
            ):
                if set(existing_ids) == set(action_map.keys()):
                    return
            else:
                return

        options = [
            Option(action_label, id=action_id) for action_id, action_label in action_map.items()
        ]
        self._set_activity_result_options(options)
        self.status = "Select source to sync"
        self._refresh_activity_panel()

    def _run_source_sync(self, source_name: str) -> None:
        if not source_name:
            return
        if self._result_list_open:
            self._close_inline_result_list(announce=False)
        self._append_activity(f"> sync --no-compact --source {source_name}")
        self.status = f"Syncing source: {source_name}"
        command_parts = ["sync", "--no-compact", "--source", source_name]
        if self.all_cursor_workspaces:
            command_parts.append("--all-cursor-workspaces")
        command = " ".join(command_parts)
        viewport_width = max(int(self.size.width or 0), 80)
        viewport_height = max(int(self.size.height or 0), 24)
        worker = self.run_worker(
            lambda: self._execute_command(command, viewport_width, viewport_height),
            thread=True,
            group="tui-ops",
            exclusive=True,
            exit_on_error=False,
        )
        self._worker_context[id(worker)] = f"sync-source:{source_name}"

    def _mount_all_view(self, dashboard: Vertical, panels: DashboardPanels) -> None:
        sidebar_children: list[Static] = []
        if self._is_widget_visible("knowledge"):
            sidebar_children.append(Static(panels.knowledge, id="dashboard_knowledge"))
        if self._is_widget_visible("sources"):
            sidebar_children.append(Static(panels.sources, id="dashboard_sources"))
        if self._is_widget_visible("settings"):
            sidebar_children.append(Static(panels.settings, id="dashboard_settings"))
        main_children: list[Static] = []
        if self._is_widget_visible("timeline"):
            main_children.append(Static(panels.timeline, id="dashboard_timeline"))
        if not sidebar_children and not main_children:
            return
        if sidebar_children and not main_children:
            dashboard.mount(Vertical(*sidebar_children, id="dashboard_all_fullwidth"))
            return
        if not sidebar_children and main_children:
            dashboard.mount(Vertical(*main_children, id="dashboard_all_fullwidth"))
            return
        sidebar = Vertical(*sidebar_children, id="dashboard_all_sidebar")
        main = Vertical(*main_children, id="dashboard_all_main")
        grid = Vertical(
            sidebar,
            main,
            id="dashboard_all_grid",
        )
        dashboard.mount(grid)

    def _build_overview_row(self, panels: DashboardPanels) -> Horizontal | None:
        show_knowledge = self._is_widget_visible("knowledge")
        show_sources = self._is_widget_visible("sources")

        if show_knowledge and show_sources:
            return Horizontal(
                Static(panels.knowledge, id="dashboard_knowledge"),
                Static(panels.sources_compact, id="dashboard_sources"),
                id="dashboard_overview_row",
            )
        if show_knowledge:
            return Horizontal(
                Static(panels.knowledge, id="dashboard_knowledge"),
                id="dashboard_overview_row",
            )
        if show_sources:
            return Horizontal(
                Static(panels.sources_compact, id="dashboard_sources"),
                id="dashboard_overview_row",
            )
        return None

    def _is_widget_visible(self, key: str) -> bool:
        return bool(self.tui_widget_visibility.get(key, True))

    def _load_tui_layout_settings(self) -> None:
        try:
            agent_dir = self._dashboard_context.agent_dir
            files = FileStorage(agent_dir)
            config = files.read_config()
            tui_config = config.get("tui", {}) if isinstance(config, dict) else {}
            if isinstance(tui_config, dict):
                widgets = tui_config.get("widget_visibility")
                if isinstance(widgets, dict):
                    self.tui_widget_visibility = {
                        key: bool(value) for key, value in widgets.items() if isinstance(key, str)
                    }
                banner = tui_config.get("banner_size")
                if isinstance(banner, str):
                    self.tui_banner_size = banner.strip().lower()
        except Exception:
            self.tui_widget_visibility = {}
            self.tui_banner_size = "normal"

        if not self.tui_widget_visibility:
            self.tui_widget_visibility = {
                "knowledge": True,
                "sources": True,
                "timeline": True,
                "ralph": True,
                "llm": True,
                "settings": True,
            }
        if self.tui_banner_size not in {"hidden", "compact", "normal", "large"}:
            self.tui_banner_size = "normal"

        self._last_layout_signature = None

    def _apply_tui_layout_settings(self) -> None:
        try:
            agent_dir = self._dashboard_context.agent_dir
            files = FileStorage(agent_dir)
            config = files.read_config()
            if "tui" not in config or not isinstance(config.get("tui"), dict):
                config["tui"] = {}
            tui_config = config["tui"]
            if isinstance(tui_config, dict):
                tui_config["widget_visibility"] = dict(self.tui_widget_visibility)
                tui_config["banner_size"] = self.tui_banner_size
            config["tui"] = tui_config
            files.write_config(config)
        except Exception as exc:  # noqa: BLE001
            self._append_activity(f"Failed to save layout settings: {exc}")

        self._last_layout_signature = None
        self.status = "Layout updated"
        self._append_activity("Layout updated.")
        self._refresh_dashboard_panel()

    def _build_slash_command_map(self) -> dict[str, str]:
        return {
            "quit": "quit",
            "q": "quit",
            "exit": "quit",
            "refresh": "status",
            "r": "status",
            "sync": "sync",
            "s": "sync",
            "run": "knowledge-run",
            "k": "knowledge-run",
            "compact": "knowledge-run",
            "help": "help",
            "h": "help",
            "?": "help",
            "status": "status",
            "sources": "sources",
            "sessions": "sessions",
            "setup": "setup",
            "model": "model",
            "settings": "settings",
            "preferences": "settings",
            "theme": "theme",
            "layout": "layout",
            "view:overview": "view:overview",
            "view:llm": "view:llm",
            "view:knowledge": "view:knowledge",
            "view:settings": "view:settings",
            "view:timeline": "view:timeline",
            "view:ralph": "view:ralph",
            "view:console": "view:console",
            "view:all": "view:all",
            "overview": "view:overview",
            "ralph": "ralph-status",
            "ralph-enable": "ralph-enable",
            "ralph-disable": "ralph-disable",
            "ralph-status": "ralph-status",
            "ralph-select": "ralph-select",
            "ralph-run": "ralph-run",
            "ralph-config": "ralph-config",
            "ralph-view-diff": "ralph-view-diff",
            "diff": "ralph-view-diff",
            "ralph-notifications": "ralph-notifications",
            "ralph-notify": "ralph-notifications",
            "notify": "ralph-notifications",
            "ralph-terminal": "ralph-terminal",
            "terminal": "ralph-terminal",
            "run:select": "run:select",
            "run-select": "run:select",
            "select": "run:select",
        }

    def _handle_slash_command(self, command: str) -> None:
        if command == "help":
            self._append_activity("[bold]Available slash commands:[/bold]")
            self._append_activity("")
            self._append_activity("[bold cyan]Core[/bold cyan]")
            self._append_activity("  /setup              - Configure sources and model defaults")
            self._append_activity(
                "  /run, /k            - Run knowledge update (ingest + synthesize)"
            )
            self._append_activity("  /run:select         - Run selected conversations")
            self._append_activity("  /sync, /s           - Sync conversations (ingest only)")
            self._append_activity("  /refresh, /r        - Refresh dashboard")
            self._append_activity("")
            self._append_activity("[bold cyan]Views[/bold cyan]")
            self._append_activity("  /view:overview      - Overview dashboard")
            self._append_activity("  /view:llm           - LLM configuration")
            self._append_activity("  /view:knowledge     - Knowledge base")
            self._append_activity("  /view:timeline      - Iteration timeline")
            self._append_activity("  /view:ralph         - Ralph loop status")
            self._append_activity("  /view:console       - Console output")
            self._append_activity("  /view:all           - All panels")
            self._append_activity("")
            self._append_activity("[bold cyan]Ralph[/bold cyan]")
            self._append_activity("  /ralph-enable       - Start Ralph loop")
            self._append_activity("  /ralph-disable      - Stop Ralph loop")
            self._append_activity("  /ralph-status       - Show Ralph status")
            self._append_activity("  /ralph-select       - Select PRD items")
            self._append_activity("  /ralph-run          - Run Ralph loop")
            self._append_activity("  /ralph-config       - Ralph configuration")
            self._append_activity("  /ralph-view-diff, /diff - Browse iteration history (diffs)")
            self._append_activity("  /ralph-notifications    - Toggle notifications")
            self._append_activity("  /ralph-terminal         - Toggle terminal panel")
            self._append_activity("")
            self._append_activity("[bold cyan]Settings[/bold cyan]")
            self._append_activity("  /model              - Model preferences")
            self._append_activity("  /settings           - Workspace preferences")
            self._append_activity("  /layout             - Customise dashboard layout")
            self._append_activity("  /theme              - Switch theme")
            self._append_activity("")
            self._append_activity("[bold cyan]System[/bold cyan]")
            self._append_activity("  /quit, /q           - Exit the TUI")
            self._append_activity("  /sources            - Check source health")
            self._append_activity("  /sessions           - Browse conversations")
            self._append_activity("")
            self._append_activity("Press Ctrl+P for full command palette.")
            self.status = "Type a slash command and press Enter"
            return

        command_map = self._build_slash_command_map()
        action_id = command_map.get(command)

        if action_id:
            self._append_activity(f"> /{command}")
            handle_palette_action(self, action_id)
        else:
            self._append_activity(f"Unknown command: /{command}")
            self._append_activity("Type /help for available commands.")
            self.status = f"Unknown command: /{command}"

    def action_focus_cli_input(self) -> None:
        input_widget = self.query_one("#cli_input", Input)
        input_widget.focus()
        input_widget.value = "/"
        input_widget.cursor_position = 1

    def _hide_suggestions(self) -> None:
        """Hide the suggestions dropdown."""
        suggestions_widget = self.query_one("#cli_suggestions", OptionList)
        suggestions_widget.display = False
        self._suggestions_visible = False
        self._highlighted_suggestion_index = None
        suggestions_widget.clear_options()

    def _show_suggestions(self, suggestions: list[str]) -> None:
        """Show the suggestions dropdown with the given suggestions."""
        if not suggestions:
            self._hide_suggestions()
            return

        suggestions_widget = self.query_one("#cli_suggestions", OptionList)
        suggestions_widget.clear_options()
        for index, suggestion in enumerate(suggestions):
            suggestions_widget.add_option(Option(suggestion, id=f"suggestion:{index}"))
        suggestions_widget.display = True
        self._suggestions_visible = True
        self._highlighted_suggestion_index = None

    def _update_suggestion_highlight(self, direction: int) -> None:
        """Update the highlighted suggestion index."""
        if not self._suggestions_visible:
            return

        suggestions_widget = self.query_one("#cli_suggestions", OptionList)
        option_count = suggestions_widget.option_count

        if option_count == 0:
            return

        if self._highlighted_suggestion_index is None:
            self._highlighted_suggestion_index = 0 if direction > 0 else option_count - 1
        else:
            self._highlighted_suggestion_index += direction
            self._highlighted_suggestion_index = max(
                0, min(self._highlighted_suggestion_index, option_count - 1)
            )

        suggestions_widget.highlighted = self._highlighted_suggestion_index

    def _apply_suggestion(self) -> None:
        """Apply the currently highlighted suggestion to the input."""
        if not self._suggestions_visible or self._highlighted_suggestion_index is None:
            return

        suggestions_widget = self.query_one("#cli_suggestions", OptionList)
        highlighted = suggestions_widget.get_option_at_index(self._highlighted_suggestion_index)
        if highlighted:
            input_widget = self.query_one("#cli_input", Input)
            input_widget.value = str(highlighted.prompt)
            input_widget.cursor_position = len(input_widget.value)
            self._hide_suggestions()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes to show command suggestions."""
        value = event.value

        # Don't show suggestions for non-slash input or empty input
        if not value or not value.startswith("/"):
            self._hide_suggestions()
            return

        # Filter commands based on input
        suggestions = filter_command_suggestions(value, self._cli_commands_cache, max_results=8)

        if suggestions:
            self._show_suggestions(suggestions)
        else:
            self._hide_suggestions()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard navigation for suggestions."""
        if not self._suggestions_visible:
            # Let parent mixins handle the key
            super().on_key(event)
            return

        # Only intercept keys when suggestions are visible and input is focused
        try:
            focused_widget = self.focused
            if focused_widget is None or focused_widget.id != "cli_input":
                # Let parent mixins handle the key
                super().on_key(event)
                return
        except Exception:
            # Let parent mixins handle the key
            super().on_key(event)
            return

        if event.key == "up":
            self._update_suggestion_highlight(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._update_suggestion_highlight(1)
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            if self._highlighted_suggestion_index is not None:
                self._apply_suggestion()
                event.prevent_default()
                event.stop()
        elif event.key == "escape":
            self._hide_suggestions()
            event.prevent_default()
            event.stop()
        else:
            # Let parent mixins handle other keys
            super().on_key(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission - apply suggestion if one is highlighted."""
        if self._suggestions_visible and self._highlighted_suggestion_index is not None:
            self._apply_suggestion()
            event.prevent_default()
            event.stop()
            return

        self._hide_suggestions()
        input_widget = self.query_one("#cli_input", Input)
        value = event.value.strip()
        input_widget.value = ""

        if not value:
            return

        if value.startswith("/"):
            command = value[1:].strip().lower()
            if command:
                self._handle_slash_command(command)
        else:
            self._append_activity(f"> {value}")
            self._append_activity("Commands must start with /. Type /help for available commands.")
            self.status = "Commands must start with /"

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle suggestion selection via mouse/Enter."""
        if event.option_list.id == "cli_suggestions":
            input_widget = self.query_one("#cli_input", Input)
            input_widget.value = str(event.option.prompt)
            input_widget.cursor_position = len(input_widget.value)
            self._hide_suggestions()
            input_widget.focus()
        else:
            # Let parent mixins handle other option lists
            super().on_option_list_option_selected(event)
