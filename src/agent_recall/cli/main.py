from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agent_recall.cli.banner import print_banner
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.compact import CompactionEngine
from agent_recall.core.context import ContextAssembler
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.retrieve import Retriever
from agent_recall.core.session import SessionManager
from agent_recall.core.sync import AutoSync
from agent_recall.ingest import get_default_ingesters, get_ingester
from agent_recall.llm import (
    Message,
    create_llm_provider,
    get_available_providers,
    validate_provider_config,
)
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LLMConfig, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage

app = typer.Typer(help="Agent Memory System - Persistent knowledge for AI coding agents")

# Initialize theme manager and console
_theme_manager = ThemeManager(DEFAULT_THEME)
console = Console(theme=_theme_manager.get_theme())

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"
T = TypeVar("T")

INITIAL_GUARDRAILS = """# Guardrails

Rules and warnings for this codebase. Entries added automatically from agent sessions.
"""

INITIAL_STYLE = """# Style Guide

Coding patterns and preferences for this codebase. Entries added automatically from agent sessions.
"""

INITIAL_RECENT = """# Recent Sessions

Summaries of recent agent sessions.
"""

INITIAL_CONFIG = """# Agent Memory Configuration

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514

compaction:
  max_recent_tokens: 1500
  max_sessions_before_compact: 5
  promote_pattern_after_occurrences: 3
  archive_sessions_older_than_days: 30

retrieval:
  backend: fts5
  top_k: 5

theme:
  name: dark+
"""


def _get_theme_manager() -> ThemeManager:
    """Get or initialize theme manager from config."""
    global _theme_manager
    if AGENT_DIR.exists():
        files = FileStorage(AGENT_DIR)
        config_dict = files.read_config()
        theme_name = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        if theme_name != _theme_manager.get_theme_name():
            _theme_manager.set_theme(theme_name)
            # Update console with new theme
            global console
            console = Console(theme=_theme_manager.get_theme())
    return _theme_manager


def ensure_initialized() -> None:
    if AGENT_DIR.exists():
        return
    _get_theme_manager()  # Ensure theme is loaded
    console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
    raise typer.Exit(1)


def get_storage() -> SQLiteStorage:
    ensure_initialized()
    return SQLiteStorage(DB_PATH)


def get_files() -> FileStorage:
    ensure_initialized()
    return FileStorage(AGENT_DIR)


def get_llm():
    files = get_files()
    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    return create_llm_provider(llm_config)


def run_with_spinner(description: str, action: Callable[[], T]) -> T:
    """Run a blocking action with a transient spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        return action()


def _as_clickable_uri(location: str) -> str:
    try:
        path = Path(location)
        if path.is_absolute():
            return path.as_uri()
    except (OSError, ValueError):
        return location
    return location


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    no_splash: bool = typer.Option(False, "--no-splash", help="Skip splash screen"),
):
    """Initialize agent memory in the current repository."""
    _get_theme_manager()

    # Show splash banner unless disabled
    if not no_splash:
        console.clear()
        # Use 'medium' as default for init to ensure it shows properly
        print_banner(console, _theme_manager, animated=True, delay=0.012)
        console.print()  # Add spacing after banner

    if AGENT_DIR.exists() and not force:
        console.print("[warning].agent/ already exists. Use --force to reinitialize.[/warning]")
        raise typer.Exit(1)

    if AGENT_DIR.exists() and force:
        shutil.rmtree(AGENT_DIR)

    AGENT_DIR.mkdir(exist_ok=True)
    (AGENT_DIR / "logs").mkdir(exist_ok=True)
    (AGENT_DIR / "archive").mkdir(exist_ok=True)

    (AGENT_DIR / "GUARDRAILS.md").write_text(INITIAL_GUARDRAILS)
    (AGENT_DIR / "STYLE.md").write_text(INITIAL_STYLE)
    (AGENT_DIR / "RECENT.md").write_text(INITIAL_RECENT)
    (AGENT_DIR / "config.yaml").write_text(INITIAL_CONFIG)

    SQLiteStorage(DB_PATH)

    console.print(
        Panel.fit(
            "[success]✓ Initialized .agent/ directory[/success]\n\n"
            "Files created:\n"
            "  • config.yaml - Configuration\n"
            "  • GUARDRAILS.md - Hard rules and warnings\n"
            "  • STYLE.md - Patterns and preferences\n"
            "  • RECENT.md - Session summaries\n"
            "  • state.db - Session and log storage",
            title="agent-recall",
        )
    )


@app.command()
def splash(
    animated: bool = typer.Option(
        True,
        "--animated/--no-animated",
        "-a/-A",
        help="Enable/disable line-by-line animation",
    ),
    delay: float = typer.Option(
        0.015,
        "--delay",
        "-d",
        min=0.0,
        max=0.5,
        help="Animation delay between lines (seconds)",
    ),
    size: str = typer.Option(
        None,
        "--size",
        "-s",
        help="Force banner size: full, medium, compact, minimal (default: auto-detect)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show debug information about terminal size",
    ),
):
    """Display the Agent Recall splash banner."""
    _get_theme_manager()

    if debug:
        console.print(f"[dim]Terminal width: {console.width}[/dim]")
        console.print(f"[dim]Terminal height: {console.height}[/dim]")
        console.print(f"[dim]Force size: {size or 'auto'}[/dim]")
        console.print()

    console.clear()
    print_banner(console, _theme_manager, animated=animated, delay=delay)


@app.command()
def start(task: str = typer.Argument(..., help="Description of what this session is working on")):
    """Start a new session and output context for the agent."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    session_mgr = SessionManager(storage)
    context_asm = ContextAssembler(storage, files)

    session = session_mgr.start(task)
    context = context_asm.assemble(task=task)

    console.print(f"[dim]Session started: {session.id}[/dim]\n")
    console.print(context)


@app.command()
def log(
    content: str = typer.Argument(..., help="The observation or learning to log"),
    label: str = typer.Option(..., "--label", "-l", help="Semantic label"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Log an observation or learning from the current session."""
    storage = get_storage()
    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)

    _get_theme_manager()  # Ensure theme is loaded
    try:
        semantic_label = SemanticLabel(label)
    except ValueError:
        valid = ", ".join(item.value for item in SemanticLabel)
        console.print(f"[error]Invalid label '{label}'. Valid: {valid}[/error]")
        raise typer.Exit(1) from None

    active = session_mgr.get_active()
    session_id = active.id if active else None

    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

    log_writer.log(
        content=content,
        label=semantic_label,
        session_id=session_id,
        tags=tag_list,
    )

    session_note = f" (session: {session_id})" if session_id else " (no active session)"
    console.print(f"[success]✓ Logged [{label}]{session_note}[/success]")


@app.command()
def end(summary: str = typer.Argument(..., help="Summary of what was accomplished")):
    """End the current session."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    session_mgr = SessionManager(storage)

    active = session_mgr.get_active()
    if not active:
        console.print("[warning]No active session to end.[/warning]")
        raise typer.Exit(1)

    session = session_mgr.end(active.id, summary)
    console.print(f"[success]✓ Session ended: {session.id}[/success]")
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Entries logged: {session.entry_count}[/dim]")


@app.command()
def context(
    task: str | None = typer.Option(None, "--task", "-t", help="Task for relevant retrieval"),
    output_format: str = typer.Option("md", "--format", "-f", help="Output format: md or json"),
):
    """Output current context (guardrails, style, recent, relevant)."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    context_asm = ContextAssembler(storage, files)
    output = context_asm.assemble(task=task)

    if output_format == "md":
        console.print(output)
        return

    if output_format == "json":
        payload = {"task": task, "context": output}
        console.print(json.dumps(payload, indent=2))
        return

    console.print("[error]Invalid format. Use 'md' or 'json'.[/error]")
    raise typer.Exit(1)


@app.command()
def compact(force: bool = typer.Option(False, "--force", "-f", help="Force compaction")):
    """Run compaction to update knowledge tiers from logs."""
    storage = get_storage()
    files = get_files()

    _get_theme_manager()  # Ensure theme is loaded
    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]LLM configuration error: {exc}[/error]")
        raise typer.Exit(1) from None

    engine = CompactionEngine(storage, files, llm)
    results = run_with_spinner(
        "Running compaction...",
        lambda: asyncio.run(engine.compact(force=force)),
    )

    console.print(
        Panel.fit(
            f"[success]✓ Compaction complete[/success]\n\n"
            f"Guardrails updated: {results['guardrails_updated']}\n"
            f"Style updated: {results['style_updated']}\n"
            f"Recent updated: {results['recent_updated']}\n"
            f"Chunks indexed: {results['chunks_indexed']}",
            title="Results",
        )
    )


@app.command()
def sync(
    compact: bool = typer.Option(
        True,
        "--compact/--no-compact",
        help="Run compaction after sync",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Only sync from specific source: cursor, claude-code",
    ),
    since_days: int | None = typer.Option(
        None,
        "--since-days",
        "-d",
        help="Only sync sessions from the last N days",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force compaction even if no new learnings",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
    cursor_db_path: Path | None = typer.Option(
        None,
        "--cursor-db-path",
        help="Path to a specific Cursor state.vscdb for testing/override",
    ),
    cursor_storage_dir: Path | None = typer.Option(
        None,
        "--cursor-storage-dir",
        help="Override Cursor workspaceStorage root path",
    ),
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces (not only current repo match)",
    ),
):
    """Automatically discover and process native agent sessions."""
    storage = get_storage()
    files = get_files()

    _get_theme_manager()  # Ensure theme is loaded
    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]LLM configuration error: {exc}[/error]")
        console.print("[dim]Check .agent/config.yaml or required API key env vars.[/dim]")
        raise typer.Exit(1) from None

    since = datetime.now(UTC) - timedelta(days=since_days) if since_days else None
    sources = [source] if source else None

    if source:
        try:
            ingesters = [
                get_ingester(
                    source,
                    cursor_db_path=cursor_db_path,
                    workspace_storage_dir=cursor_storage_dir,
                    cursor_all_workspaces=all_cursor_workspaces,
                )
            ]
        except ValueError as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(1) from None
    else:
        ingesters = get_default_ingesters(
            cursor_db_path=cursor_db_path,
            workspace_storage_dir=cursor_storage_dir,
            cursor_all_workspaces=all_cursor_workspaces,
        )

    auto_sync = AutoSync(storage, files, llm, ingesters)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Syncing sessions...", total=None)

        try:
            if compact:
                results = asyncio.run(
                    auto_sync.sync_and_compact(
                        since=since,
                        sources=sources,
                        force_compact=force,
                    )
                )
            else:
                results = asyncio.run(auto_sync.sync(since=since, sources=sources))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[error]Sync failed: {exc}[/error]")
            raise typer.Exit(1) from None

    lines = [""]
    lines.append(f"Sessions discovered: {results['sessions_discovered']}")
    lines.append(f"Sessions processed:  {results['sessions_processed']}")
    if int(results["sessions_skipped"]) > 0:
        lines.append(f"Sessions skipped:    {results['sessions_skipped']} (already processed)")
    lines.append(f"Learnings extracted: {results['learnings_extracted']}")

    if verbose and results.get("by_source"):
        lines.append("")
        lines.append("[bold]By source:[/bold]")
        by_source = results["by_source"]
        for source_name, source_stats in by_source.items():
            lines.append(
                f"  {source_name}: {source_stats['processed']}/{source_stats['discovered']} "
                f"processed, {source_stats['learnings']} learnings"
            )
            if int(source_stats.get("empty", 0)) > 0:
                lines.append(
                    f"    [dim]{source_stats['empty']} discovered session(s) had no parseable "
                    "messages and were skipped[/dim]"
                )

    if "compaction" in results:
        comp = results["compaction"]
        lines.append("")
        lines.append("[bold]Compaction:[/bold]")
        lines.append(f"  Guardrails updated: {'✓' if comp.get('guardrails_updated') else '-'}")
        lines.append(f"  Style updated:      {'✓' if comp.get('style_updated') else '-'}")
        lines.append(f"  Chunks indexed:     {comp.get('chunks_indexed', 0)}")

    if results.get("errors"):
        errors = results["errors"]
        lines.append("")
        lines.append(f"[warning]Warnings: {len(errors)}[/warning]")
        if verbose:
            for error in errors[:10]:
                lines.append(f"  [dim]- {error}[/dim]")
            if len(errors) > 10:
                lines.append(f"  [dim]... and {len(errors) - 10} more[/dim]")

    lines.append("")

    if int(results["learnings_extracted"]) > 0:
        title = "[success]✓ Sync Complete[/success]"
    elif int(results["sessions_processed"]) > 0:
        title = "[warning]Sync Complete (no learnings extracted)[/warning]"
    else:
        title = "[dim]Sync Complete (no new sessions)[/dim]"

    console.print(Panel.fit("\n".join(lines), title=title))


@app.command()
def sources(
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Show Cursor sessions from all workspaces, not only current repo match",
    ),
):
    """Show available native session sources and discovery status."""
    _get_theme_manager()  # Ensure theme is loaded
    ensure_initialized()

    ingesters = get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces)

    table = Table(title="Session Sources")
    table.add_column("Source", style="table_header")
    table.add_column("Status")
    table.add_column("Sessions", justify="right")
    table.add_column("Location", overflow="fold")

    rows: list[tuple[str, str, str, str]] = []

    def _collect_rows() -> None:
        for ingester in ingesters:
            try:
                sessions = ingester.discover_sessions()
                available = len(sessions) > 0
                status_text = "✓ Available" if available else "- No sessions"
                status_style = "success" if available else "dim"

                if available:
                    location = str(sessions[0].resolve())
                elif ingester.source_name == "cursor" and hasattr(ingester, "storage_dir"):
                    location = str(Path(getattr(ingester, "storage_dir")).resolve())
                elif ingester.source_name == "claude-code" and hasattr(ingester, "claude_dir"):
                    location = str((Path(getattr(ingester, "claude_dir")) / "projects").resolve())
                else:
                    location = "Unknown"

                rows.append(
                    (
                        ingester.source_name,
                        f"[{status_style}]{status_text}[/{status_style}]",
                        str(len(sessions)),
                        _as_clickable_uri(location),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    (
                        ingester.source_name,
                        "[error]✗ Error[/error]",
                        "-",
                        f"[error]{str(exc)[:40]}[/error]",
                    )
                )

    run_with_spinner("Discovering session sources...", _collect_rows)

    for row in rows:
        table.add_row(*row)

    console.print(table)


@app.command()
def status():
    """Show agent-recall status and statistics."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    stats = storage.get_stats()
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)

    lines = [
        "[bold]Knowledge Base:[/bold]",
        f"  Processed sessions: {stats.get('processed_sessions', 0)}",
        f"  Log entries:        {stats.get('log_entries', 0)}",
        f"  Indexed chunks:     {stats.get('chunks', 0)}",
        "",
        "[bold]Tier Files:[/bold]",
        f"  GUARDRAILS.md: {len(guardrails):,} chars",
        f"  STYLE.md:      {len(style):,} chars",
        f"  RECENT.md:     {len(recent):,} chars",
        "",
        "[bold]Session Sources:[/bold]",
    ]

    source_lines: list[str] = []

    def _collect_source_lines() -> None:
        for ingester in get_default_ingesters():
            try:
                available = len(ingester.discover_sessions())
                icon = "✓" if available else "-"
                source_lines.append(
                    f"  {icon} {ingester.source_name}: {available} sessions available"
                )
            except Exception as exc:  # noqa: BLE001
                source_lines.append(
                    f"  ✗ {ingester.source_name}: [error]Error - {str(exc)[:30]}[/error]"
                )

    run_with_spinner("Collecting source status...", _collect_source_lines)
    lines.extend(source_lines)

    console.print(Panel.fit("\n".join(lines), title="Agent Recall Status"))


def _build_tui_dashboard(
    all_cursor_workspaces: bool = False,
    include_banner_header: bool = True,
) -> Group:
    """Build the live TUI dashboard renderable."""
    from agent_recall.cli.banner import BannerRenderer

    storage = get_storage()
    files = get_files()

    stats = storage.get_stats()
    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)

    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))

    # Create banner header
    banner_renderer = BannerRenderer(console, _theme_manager)
    header_text = banner_renderer.get_tui_header_text()

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="table_header")
    summary.add_column()
    summary.add_row("Processed sessions", str(stats.get("processed_sessions", 0)))
    summary.add_row("Log entries", str(stats.get("log_entries", 0)))
    summary.add_row("Indexed chunks", str(stats.get("chunks", 0)))
    summary.add_row("GUARDRAILS.md", f"{len(guardrails):,} chars")
    summary.add_row("STYLE.md", f"{len(style):,} chars")
    summary.add_row("RECENT.md", f"{len(recent):,} chars")

    llm_summary = Table.grid(padding=(0, 2))
    llm_summary.add_column(style="table_header")
    llm_summary.add_column()
    llm_summary.add_row("Provider", llm_config.provider)
    llm_summary.add_row("Model", llm_config.model)
    llm_summary.add_row("Base URL", llm_config.base_url or "default")
    llm_summary.add_row("Temperature", str(llm_config.temperature))
    llm_summary.add_row("Max tokens", str(llm_config.max_tokens))

    source_table = Table(expand=True)
    source_table.add_column("Source", style="table_header")
    source_table.add_column("Status")
    source_table.add_column("Sessions", justify="right")
    source_table.add_column("Location", overflow="fold")

    for ingester in get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces):
        try:
            sessions = ingester.discover_sessions()
            available = len(sessions) > 0
            status_text = "✓ Available" if available else "- No sessions"
            status_style = "success" if available else "dim"

            if available:
                location = str(sessions[0].resolve())
            elif ingester.source_name == "cursor" and hasattr(ingester, "storage_dir"):
                location = str(Path(getattr(ingester, "storage_dir")).resolve())
            elif ingester.source_name == "claude-code" and hasattr(ingester, "claude_dir"):
                location = str((Path(getattr(ingester, "claude_dir")) / "projects").resolve())
            else:
                location = "Unknown"

            source_table.add_row(
                ingester.source_name,
                f"[{status_style}]{status_text}[/{status_style}]",
                str(len(sessions)),
                _as_clickable_uri(location),
            )
        except Exception as exc:  # noqa: BLE001
            source_table.add_row(
                ingester.source_name,
                "[error]✗ Error[/error]",
                "-",
                f"[error]{str(exc)[:40]}[/error]",
            )

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    panels = []

    if include_banner_header:
        panels.append(
            Panel(
                header_text,
                title=f"[dim]Updated {now_text}[/dim]",
                subtitle="[dim]Press Ctrl+C to exit[/dim]",
                border_style="banner.border",
            )
        )

    panels.extend(
        [
            Panel(summary, title="Knowledge Base", border_style="accent"),
            Panel(llm_summary, title="LLM Configuration", border_style="accent"),
            Panel(source_table, title="Session Sources", border_style="accent"),
        ]
    )

    return Group(*panels)


@app.command()
def tui(
    refresh_seconds: float = typer.Option(
        2.0,
        "--refresh-seconds",
        "-r",
        min=0.2,
        help="Refresh interval for live dashboard updates.",
    ),
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces.",
    ),
    iterations: int | None = typer.Option(
        None,
        "--iterations",
        hidden=True,
        help="Number of refresh loops (for tests).",
    ),
    no_splash: bool = typer.Option(
        False,
        "--no-splash",
        help="Skip the initial splash animation.",
    ),
    splash_delay: float = typer.Option(
        0.012,
        "--splash-delay",
        min=0.0,
        max=0.1,
        help="Animation delay for splash screen.",
    ),
):
    """Start a live terminal UI dashboard for agent-recall."""
    _get_theme_manager()
    ensure_initialized()

    if iterations is not None and iterations < 1:
        console.print("[error]--iterations must be >= 1[/error]")
        raise typer.Exit(1)

    # Show splash animation first (unless disabled)
    if not no_splash:
        console.clear()
        print_banner(console, _theme_manager, animated=True, delay=splash_delay)
        time.sleep(0.8)  # Pause to admire the banner

    refresh_rate = max(1, int(round(1 / refresh_seconds)))
    completed = 0

    try:
        with Live(
            _build_tui_dashboard(all_cursor_workspaces=all_cursor_workspaces),
            console=console,
            screen=True,
            refresh_per_second=refresh_rate,
        ) as live:
            while True:
                live.update(_build_tui_dashboard(all_cursor_workspaces=all_cursor_workspaces))
                completed += 1
                if iterations is not None and completed >= iterations:
                    break
                time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        console.print("\n[dim]TUI closed.[/dim]")


@app.command("reset-sync")
def reset_sync(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Clear only one source: cursor or claude-code",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Clear only one exact processed source session ID",
    ),
):
    """Reset processed-session markers so sessions can be re-synced for testing."""
    _get_theme_manager()
    storage = get_storage()

    if source:
        normalized = source.strip().lower().replace("_", "-")
        if normalized not in {"cursor", "claude-code"}:
            console.print("[error]Invalid source. Use: cursor or claude-code[/error]")
            raise typer.Exit(1)
        source = normalized

    removed = storage.clear_processed_sessions(source=source, source_session_id=session_id)

    if session_id:
        scope = f"session_id={session_id}"
    elif source:
        scope = f"source={source}"
    else:
        scope = "all sources"

    console.print(
        Panel.fit(
            f"[success]✓ Reset sync markers[/success]\n\n"
            f"Scope: {scope}\n"
            f"Cleared processed session markers: {removed}\n\n"
            "[dim]Note: log entries/chunks are unchanged.[/dim]",
            title="Reset Complete",
        )
    )


@app.command()
def config_llm(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=(
            "LLM provider: anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible"
        ),
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        "-u",
        help="API base URL (for local/custom providers)",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate configuration after setting",
    ),
):
    """Configure the LLM provider for extraction and compaction."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    config_dict = files.read_config()
    llm_config = dict(config_dict.get("llm", {}))

    if provider:
        llm_config["provider"] = provider
    if model:
        llm_config["model"] = model
    if base_url:
        llm_config["base_url"] = base_url

    try:
        parsed = LLMConfig(**llm_config)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Invalid LLM config: {exc}[/error]")
        raise typer.Exit(1) from None

    if validate and (provider or model or base_url):
        valid, message = validate_provider_config(parsed)
        if not valid:
            console.print(f"[warning]Warning: {message}[/warning]")
        else:
            try:
                llm = create_llm_provider(parsed)
                success, validation_message = run_with_spinner(
                    "Validating provider connection...",
                    llm.validate,
                )
                if success:
                    console.print(f"[success]✓ {validation_message}[/success]")
                else:
                    console.print(f"[warning]Warning: {validation_message}[/warning]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[warning]Warning: Could not validate provider: {exc}[/warning]")

    config_dict["llm"] = llm_config
    files.write_config(config_dict)

    base_url_display = llm_config.get("base_url") or "default"
    console.print(
        Panel.fit(
            f"Provider: {llm_config.get('provider', 'not set')}\n"
            f"Model: {llm_config.get('model', 'not set')}\n"
            f"Base URL: {base_url_display}",
            title="LLM Configuration Updated",
        )
    )


@app.command()
def providers():
    """List available LLM providers and their configuration hints."""
    _get_theme_manager()
    available = set(get_available_providers())

    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="table_header")
    table.add_column("Type")
    table.add_column("API Key Env")
    table.add_column("Default Base URL")

    providers_info = [
        ("anthropic", "Cloud", "ANTHROPIC_API_KEY", "-"),
        ("openai", "Cloud", "OPENAI_API_KEY", "-"),
        ("google", "Cloud", "GOOGLE_API_KEY", "-"),
        ("ollama", "Local", "-", "http://localhost:11434/v1"),
        ("vllm", "Local", "- (optional)", "http://localhost:8000/v1"),
        ("lmstudio", "Local", "-", "http://localhost:1234/v1"),
        ("openai-compatible", "Custom", "- (optional)", "(required)"),
    ]

    for name, provider_type, key_env, default_base_url in providers_info:
        if name in available:
            table.add_row(name, provider_type, key_env, default_base_url)

    console.print(table)
    console.print()
    console.print("[bold]Examples:[/bold]")
    console.print("  export ANTHROPIC_API_KEY=sk-...")
    console.print(
        "  agent-recall config-llm --provider anthropic --model claude-sonnet-4-20250514"
    )
    console.print("  agent-recall config-llm --provider ollama --model llama3.1")
    console.print("  agent-recall config-llm --provider lmstudio --model local-model")
    console.print(
        "  agent-recall config-llm --provider openai-compatible --base-url "
        "http://localhost:8080/v1 --model my-model"
    )


@app.command()
def test_llm():
    """Test the configured LLM provider with a small generation call."""
    _get_theme_manager()
    ensure_initialized()

    try:
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Failed to initialize LLM: {exc}[/error]")
        raise typer.Exit(1) from None

    success, validation_message = run_with_spinner(
        f"Validating {llm.provider_name} ({llm.model_name})...",
        llm.validate,
    )
    if not success:
        console.print(f"[error]✗ {validation_message}[/error]")
        raise typer.Exit(1)
    console.print(f"[success]✓ {validation_message}[/success]")

    async def _run() -> str:
        response = await llm.generate(
            [
                Message(
                    role="user",
                    content="Say 'Hello from agent-recall!' in exactly those words.",
                )
            ],
            max_tokens=50,
        )
        return response.content

    try:
        content = run_with_spinner("Running test generation...", lambda: asyncio.run(_run()))
        console.print("[success]✓ Generation successful[/success]")
        console.print(f"[dim]Response: {content[:120]}[/dim]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]✗ Generation failed: {exc}[/error]")
        raise typer.Exit(1) from None


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Maximum results"),
):
    """Retrieve relevant memory chunks using FTS5 search."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    retriever = Retriever(storage)

    _get_theme_manager()  # Ensure theme is loaded
    chunks = retriever.search(query=query, top_k=top_k)
    if not chunks:
        console.print("[warning]No matching chunks found.[/warning]")
        raise typer.Exit(0)

    for idx, chunk in enumerate(chunks, 1):
        tags = f" [{', '.join(chunk.tags)}]" if chunk.tags else ""
        console.print(f"{idx}. ({chunk.label.value}) {chunk.content}{tags}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    source_session_id: str | None = typer.Option(
        None,
        "--source-session-id",
        help="Native session ID",
    ),
):
    """Ingest a JSONL transcript into log entries."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    ingestor = TranscriptIngestor(storage)

    count = run_with_spinner(
        "Ingesting transcript...",
        lambda: ingestor.ingest_jsonl(path=path, source_session_id=source_session_id),
    )
    console.print(f"[success]✓ Ingested {count} transcript entries[/success]")


theme_app = typer.Typer(help="Manage CLI themes")


@theme_app.command("list")
def theme_list():
    """List all available themes."""
    # Get current theme from config if available
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            files = FileStorage(AGENT_DIR)
            config_dict = files.read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass

    themes = ThemeManager.get_available_themes()

    table = Table(title="Available Themes")
    table.add_column("Name", style="table_header")
    table.add_column("Status")

    for theme_name in themes:
        status = "[success]✓ Current[/success]" if theme_name == current_theme else ""
        table.add_row(theme_name, status)

    console.print(table)


@theme_app.command("set")
def theme_set(name: str = typer.Argument(..., help="Theme name to set")):
    """Set the active theme."""
    global console
    if not ThemeManager.is_valid_theme(name):
        available = ", ".join(ThemeManager.get_available_themes())
        console.print(f"[error]Invalid theme: {name}[/error]")
        console.print(f"[dim]Available themes: {available}[/dim]")
        raise typer.Exit(1)

    if not AGENT_DIR.exists():
        console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
        raise typer.Exit(1)

    try:
        files = get_files()
        config_dict = files.read_config()
        if "theme" not in config_dict:
            config_dict["theme"] = {}
        config_dict["theme"]["name"] = name
        files.write_config(config_dict)

        # Update current theme manager
        _theme_manager.set_theme(name)
        console = Console(theme=_theme_manager.get_theme())

        console.print(f"[success]✓ Theme set to '{name}'[/success]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Failed to set theme: {exc}[/error]")
        raise typer.Exit(1) from None


@theme_app.command("show")
def theme_show():
    """Show the current theme."""
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            files = FileStorage(AGENT_DIR)
            config_dict = files.read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass

    console.print(f"Current theme: [accent]{current_theme}[/accent]")


app.add_typer(theme_app, name="theme")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
