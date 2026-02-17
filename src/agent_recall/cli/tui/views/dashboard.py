from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime

from rich import box
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from agent_recall.cli.banner import BannerRenderer
from agent_recall.cli.tui.views.dashboard_context import DashboardRenderContext
from agent_recall.cli.tui.widgets import (
    KnowledgeWidget,
    LLMConfigWidget,
    SettingsWidget,
    SourcesWidget,
    TimelineWidget,
)
from agent_recall.core.onboarding import API_KEY_ENV_BY_PROVIDER
from agent_recall.ralph.iteration_store import IterationReportStore
from agent_recall.storage.files import KnowledgeTier
from agent_recall.storage.models import LLMConfig


@dataclass(frozen=True)
class DashboardPanels:
    header: Panel | None
    knowledge: Panel
    llm: Panel
    sources: Panel
    sources_compact: Panel
    settings: Panel
    timeline: Panel
    slash_console: Panel | None


def build_dashboard_panels(
    context: DashboardRenderContext,
    all_cursor_workspaces: bool = False,
    include_banner_header: bool = True,
    slash_status: str | None = None,
    slash_output: list[str] | None = None,
    view: str = "overview",
    refresh_seconds: float = 2.0,
    show_slash_console: bool = True,
) -> DashboardPanels:
    storage = context.get_storage()
    files = context.get_files()

    stats = storage.get_stats()
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)

    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    provider_key = llm_config.provider.strip().lower()
    api_key_env = API_KEY_ENV_BY_PROVIDER.get(provider_key)
    api_key_set = bool(os.environ.get(api_key_env, "").strip()) if api_key_env else True
    api_key_set_display = "Yes" if api_key_set else "No"
    selected_sources = context.get_repo_selected_sources(files)
    repo_root = context.resolve_repo_root_for_display()
    repo_name = repo_root.name
    configured_agents = ", ".join(selected_sources) if selected_sources else "all"

    report_store = IterationReportStore(context.agent_dir / "ralph")
    timeline_lines = context.render_iteration_timeline(
        report_store,
        max_entries=8 if view == "all" else 12,
    )
    cost_summary = context.summarize_costs(report_store.load_all())

    if view == "all":
        wrap_width = max(24, min(44, (max(context.console.size.width, 100) // 2) - 24))
    else:
        wrap_width = 52
    configured_agents_wrapped = textwrap.fill(
        configured_agents,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    source_table = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    source_table.add_column("Source", style="table_header")
    source_table.add_column("Status")
    source_table.add_column("Sessions", justify="right")

    source_compact_lines: list[str] = []
    active_source_names: list[str] = []

    ingesters = context.filter_ingesters_by_sources(
        context.get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )

    for ingester in ingesters:
        try:
            sessions = ingester.discover_sessions()
            session_count = len(sessions)
            available = session_count > 0
            status_text = "✓ Available" if available else "- No sessions"
            status_style = "success" if available else "dim"

            if available:
                active_source_names.append(ingester.source_name)

            source_table.add_row(
                ingester.source_name,
                f"[{status_style}]{status_text}[/{status_style}]",
                str(session_count),
            )
            source_compact_lines.append(
                f"[table_header]{ingester.source_name}[/table_header]  "
                f"[{status_style}]{status_text}[/{status_style}]  "
                f"({session_count} session{'s' if session_count != 1 else ''})"
            )
        except Exception:  # noqa: BLE001
            source_table.add_row(
                ingester.source_name,
                "[error]✗ Error[/error]",
                "-",
            )
            source_compact_lines.append(
                f"[table_header]{ingester.source_name}[/table_header]  [error]✗ Error[/error]"
            )

    if not source_compact_lines:
        source_compact_lines.append("[dim]No configured session sources.[/dim]")

    last_processed_at = storage.get_last_processed_at()
    if last_processed_at is None:
        last_synced_display = "Never"
    else:
        normalized_last_processed = (
            last_processed_at.replace(tzinfo=UTC)
            if last_processed_at.tzinfo is None
            else last_processed_at.astimezone(UTC)
        )
        last_synced_display = normalized_last_processed.strftime("%Y-%m-%d %H:%M UTC")

    active_agents = ", ".join(active_source_names) if active_source_names else "none"
    active_agents_wrapped = textwrap.fill(
        active_agents,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    banner_renderer = BannerRenderer(context.console, context.theme_manager)
    header_text = banner_renderer.get_tui_header_text()

    knowledge_widget = KnowledgeWidget(
        repo_name=repo_name,
        stats=stats,
        guardrails_len=len(guardrails),
        style_len=len(style),
        recent_len=len(recent),
        total_tokens=cost_summary.total_tokens,
        total_cost_usd=cost_summary.total_cost_usd,
        format_usd=context.format_usd,
    )
    llm_widget = LLMConfigWidget(
        llm_config=llm_config,
        api_key_set_display=api_key_set_display,
        view=view,
    )
    sources_widget = SourcesWidget(
        source_table=source_table,
        compact_lines=source_compact_lines,
        last_synced_display=last_synced_display,
        compact=False,
    )
    sources_compact_widget = SourcesWidget(
        source_table=source_table,
        compact_lines=source_compact_lines,
        last_synced_display=last_synced_display,
        compact=True,
    )
    settings_widget = SettingsWidget(
        view=view,
        refresh_seconds=refresh_seconds,
        theme_name=context.theme_manager.get_theme_name(),
        interactive_shell=context.is_interactive_terminal(),
        repo_name=repo_name,
        active_agents_wrapped=active_agents_wrapped,
        configured_agents_wrapped=configured_agents_wrapped,
    )
    timeline_widget = TimelineWidget(timeline_lines=timeline_lines)

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_panel = None
    if include_banner_header:
        header_panel = Panel(
            header_text,
            title=f"[dim]Updated {now_text}[/dim]",
            subtitle="[dim]Press Ctrl+Q to exit[/dim]",
            border_style="banner.border",
        )

    knowledge_panel = knowledge_widget.render()
    llm_panel = llm_widget.render()
    sources_panel = sources_widget.render()
    sources_compact_panel = sources_compact_widget.render()
    settings_panel = settings_widget.render()
    timeline_panel = timeline_widget.render()

    slash_panel = None
    if show_slash_console:
        slash_lines = slash_output or context.help_lines_provider()
        if slash_status:
            slash_lines = [f"[accent]{escape(slash_status)}[/accent]", *slash_lines]
        line_budget = 14 if view in {"console", "all", "settings"} else 6
        slash_lines = slash_lines[-line_budget:]
        slash_panel = Panel(
            "\n".join(slash_lines),
            title="Slash Console",
            subtitle="[dim]Type /help and press Enter. Use /quit to exit.[/dim]",
            border_style="accent",
        )

    return DashboardPanels(
        header=header_panel,
        knowledge=knowledge_panel,
        llm=llm_panel,
        sources=sources_panel,
        sources_compact=sources_compact_panel,
        settings=settings_panel,
        timeline=timeline_panel,
        slash_console=slash_panel,
    )


def build_tui_dashboard(
    context: DashboardRenderContext,
    all_cursor_workspaces: bool = False,
    include_banner_header: bool = True,
    slash_status: str | None = None,
    slash_output: list[str] | None = None,
    view: str = "overview",
    refresh_seconds: float = 2.0,
    show_slash_console: bool = True,
) -> Group:
    panels = build_dashboard_panels(
        context,
        all_cursor_workspaces=all_cursor_workspaces,
        include_banner_header=include_banner_header,
        slash_status=slash_status,
        slash_output=slash_output,
        view=view,
        refresh_seconds=refresh_seconds,
        show_slash_console=show_slash_console,
    )

    def _two_panel_row(left: Panel, right: Panel) -> Table:
        row = Table.grid(expand=True, padding=(0, 2))
        row.add_column(ratio=1)
        row.add_column(ratio=1)
        row.add_row(left, right)
        return row

    renderables = []
    if panels.header is not None:
        renderables.append(panels.header)

    if view == "knowledge":
        renderables.append(panels.knowledge)
    elif view == "timeline":
        renderables.append(panels.timeline)
    elif view == "llm":
        renderables.append(panels.llm)
    elif view == "sources":
        renderables.append(panels.sources)
    elif view == "settings":
        renderables.append(panels.settings)
    elif view == "console":
        pass
    elif view == "all":
        renderables.extend(
            [
                _two_panel_row(panels.knowledge, panels.llm),
                _two_panel_row(panels.sources, panels.settings),
                panels.timeline,
            ]
        )
    else:
        renderables.append(_two_panel_row(panels.knowledge, panels.sources_compact))

    if panels.slash_console is not None:
        renderables.append(panels.slash_console)

    return Group(*renderables)
