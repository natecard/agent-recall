from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import shutil
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typer.main import get_command as get_typer_command
from typer.testing import CliRunner

from agent_recall.cli.banner import print_banner
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.compact import CompactionEngine
from agent_recall.core.context import ContextAssembler
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.onboarding import (
    apply_repo_setup,
    default_agent_recall_home,
    discover_provider_models,
    ensure_repo_onboarding,
    get_onboarding_defaults,
    get_repo_preferred_sources,
    inject_stored_api_keys,
    is_interactive_terminal,
    is_repo_onboarding_complete,
)
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
_slash_runner = CliRunner()
config_app = typer.Typer(help="Manage onboarding and model configuration")

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
    inject_stored_api_keys()
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


def _resolve_repo_root_for_display(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _get_repo_selected_sources(files: FileStorage) -> list[str] | None:
    try:
        return get_repo_preferred_sources(files)
    except Exception:  # noqa: BLE001
        return None


def _filter_ingesters_by_sources(ingesters, selected_sources: list[str] | None):
    if not selected_sources:
        return ingesters
    allowed = set(selected_sources)
    filtered = [ingester for ingester in ingesters if ingester.source_name in allowed]
    return filtered if filtered else ingesters


def _tui_help_lines() -> list[str]:
    return [
        "[bold]Slash Commands[/bold]",
        "[dim]/view overview|sources|llm|knowledge|settings|console|all[/dim] - switch TUI view",
        "[dim]/status[/dim] - Show stats and source availability",
        "[dim]/sync --no-compact[/dim] - Sync sessions quickly",
        "[dim]/compact[/dim] - Run compaction now",
        "[dim]/sources[/dim] - Show detected session sources",
        "[dim]/config model --temperature 0.2 --max-tokens 8192[/dim] - Tune model settings",
        "[dim]/config setup --force[/dim] - Re-run setup in this repo",
        "[dim]/settings[/dim] - Open settings view",
        "[dim]/quit[/dim] - Exit the TUI",
    ]


def _collect_cli_commands_for_palette() -> list[str]:
    root = get_typer_command(app)
    seen: set[str] = set()
    commands: list[str] = []

    def _walk(command, prefix: str = "") -> None:
        mapping = getattr(command, "commands", None)
        if not isinstance(mapping, dict):
            return
        for name in sorted(mapping):
            subcommand = mapping[name]
            if getattr(subcommand, "hidden", False):
                continue
            full = f"{prefix} {name}".strip()
            if full not in seen:
                seen.add(full)
                commands.append(full)
            _walk(subcommand, full)

    _walk(root)
    return commands


def _normalize_tui_command(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    return text if text.startswith("/") else f"/{text}"


def _read_tui_command(timeout_seconds: float) -> str | None:
    if not is_interactive_terminal():
        time.sleep(timeout_seconds)
        return None

    if os.name == "nt":
        time.sleep(timeout_seconds)
        return None

    try:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        time.sleep(timeout_seconds)
        return None

    if not ready:
        return None

    line = sys.stdin.readline()
    if line == "":
        return "/quit"
    return line.rstrip("\n")


def _execute_tui_slash_command(raw: str) -> tuple[bool, list[str]]:
    value = _normalize_tui_command(raw)
    if not value:
        return False, []

    if not value.startswith("/"):
        return False, ["[warning]Commands must start with '/'. Try /help.[/warning]"]

    command_text = value[1:].strip()
    if not command_text:
        return False, ["[warning]Empty command. Try /help.[/warning]"]

    try:
        parts = shlex.split(command_text)
    except ValueError as exc:
        return False, [f"[error]Invalid command: {escape(str(exc))}[/error]"]

    command_name = parts[0].lower()
    if command_name in {"q", "quit", "exit"}:
        return True, ["[dim]Leaving TUI...[/dim]"]

    if command_name in {"help", "h", "?"}:
        return False, _tui_help_lines()

    if command_name == "settings":
        return False, [
            "[success]✓ Switched to settings view[/success]",
            "[dim]Use /view settings[/dim]",
        ]

    if command_name == "config" and len(parts) >= 2 and parts[1].lower() == "setup":
        force = "--force" in parts or "-f" in parts
        quick = "--quick" in parts
        _run_onboarding_setup(force=force, quick=quick)
        return False, ["[success]✓ Setup flow completed[/success]"]

    if command_name == "tui":
        return False, ["[warning]/tui is already running.[/warning]"]

    result = _slash_runner.invoke(app, parts)
    command_label = "/" + " ".join(parts)

    lines: list[str] = []
    if result.exit_code == 0:
        lines.append(f"[success]✓ {escape(command_label)}[/success]")
    else:
        lines.append(f"[error]✗ {escape(command_label)} (exit {result.exit_code})[/error]")

    output = result.output.strip()
    if output:
        output_lines = output.splitlines()
        max_lines = 8
        for line in output_lines[:max_lines]:
            lines.append(f"[dim]{escape(line)}[/dim]")
        remaining = len(output_lines) - max_lines
        if remaining > 0:
            lines.append(f"[dim]... and {remaining} more line(s)[/dim]")

    return False, lines


def _handle_tui_view_command(raw: str, current_view: str) -> tuple[bool, str, list[str]]:
    value = _normalize_tui_command(raw)
    if not value.startswith("/"):
        return False, current_view, []

    try:
        parts = shlex.split(value[1:].strip())
    except ValueError as exc:
        return True, current_view, [f"[error]Invalid command: {escape(str(exc))}[/error]"]

    if not parts:
        return False, current_view, []

    command = parts[0].lower()
    if command not in {"view", "menu"}:
        return False, current_view, []

    valid_views = {"overview", "sources", "llm", "knowledge", "settings", "console", "all"}
    if len(parts) == 1:
        return (
            True,
            current_view,
            [
                f"[dim]Current view: {current_view}[/dim]",
                "[dim]Available views: overview, sources, llm, "
                "knowledge, settings, console, all[/dim]",
            ],
        )

    requested = parts[1].strip().lower()
    if requested not in valid_views:
        return (
            True,
            current_view,
            [f"[warning]Unknown view '{escape(requested)}'. Try /view overview[/warning]"],
        )

    return True, requested, [f"[success]✓ Switched to {requested} view[/success]"]


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
    selected_sources = None if source else _get_repo_selected_sources(files)
    sources = [source] if source else selected_sources

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
        ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)

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
    if selected_sources and not source:
        lines.append(f"Using configured sources: {', '.join(selected_sources)}")
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
    files = get_files()

    selected_sources = _get_repo_selected_sources(files)
    ingesters = get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces)
    ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)

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

    if selected_sources:
        console.print(f"[dim]Configured sources: {', '.join(selected_sources)}[/dim]")

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
    selected_sources = _get_repo_selected_sources(files)

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
        "[bold]Onboarding:[/bold]",
        f"  Completed: {'yes' if is_repo_onboarding_complete(files) else 'no'}",
        f"  Agents:    {', '.join(selected_sources) if selected_sources else 'all'}",
        "",
        "[bold]Session Sources:[/bold]",
    ]

    source_lines: list[str] = []

    def _collect_source_lines() -> None:
        ingesters = _filter_ingesters_by_sources(
            get_default_ingesters(),
            selected_sources,
        )
        for ingester in ingesters:
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
    slash_status: str | None = None,
    slash_output: list[str] | None = None,
    view: str = "overview",
    refresh_seconds: float = 2.0,
    show_slash_console: bool = True,
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
    selected_sources = _get_repo_selected_sources(files)
    repo_root = _resolve_repo_root_for_display()
    repo_name = repo_root.name

    # Create banner header
    banner_renderer = BannerRenderer(console, _theme_manager)
    header_text = banner_renderer.get_tui_header_text()

    knowledge_summary = Group(
        (
            "[table_header]Repo[/table_header] "
            f"{repo_name}    "
            "[table_header]Agents[/table_header] "
            f"{', '.join(selected_sources) if selected_sources else 'all'}"
        ),
        (
            "[table_header]Processed[/table_header] "
            f"{stats.get('processed_sessions', 0)}    "
            "[table_header]Logs[/table_header] "
            f"{stats.get('log_entries', 0)}    "
            "[table_header]Chunks[/table_header] "
            f"{stats.get('chunks', 0)}"
        ),
        (
            "[table_header]Files[/table_header] "
            f"GUARDRAILS {len(guardrails):,}c  |  "
            f"STYLE {len(style):,}c  |  "
            f"RECENT {len(recent):,}c"
        ),
    )

    llm_summary = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    llm_summary.add_column("Provider", style="table_header")
    llm_summary.add_column("Model", style="table_header", overflow="fold")
    llm_summary.add_column("Base URL", style="table_header", overflow="fold")
    llm_summary.add_column("Temperature", style="table_header")
    llm_summary.add_column("Max tokens", style="table_header")
    llm_summary.add_row(
        llm_config.provider,
        llm_config.model,
        llm_config.base_url or "default",
        str(llm_config.temperature),
        str(llm_config.max_tokens),
    )

    settings_table = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    settings_table.add_column("Setting", style="table_header")
    settings_table.add_column("Value", overflow="fold")
    settings_table.add_row("Current view", view)
    settings_table.add_row("Refresh seconds", str(refresh_seconds))
    settings_table.add_row("Interactive shell", "yes" if is_interactive_terminal() else "no")
    settings_table.add_row("Theme", _theme_manager.get_theme_name())
    settings_table.add_row("Repository", repo_name)
    settings_table.add_row("Repository path", str(repo_root))
    settings_table.add_row(
        "Configured agents",
        ", ".join(selected_sources) if selected_sources else "all",
    )
    settings_table.add_row("Config path", str((AGENT_DIR / "config.yaml").resolve()))
    settings_table.add_row("Local home", str(default_agent_recall_home()))

    source_table = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    source_table.add_column("Source", style="table_header")
    source_table.add_column("Status")
    source_table.add_column("Sessions", justify="right")
    source_table.add_column("Location", overflow="fold")

    source_compact_lines: list[str] = []

    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )

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

            source_table.add_row(
                ingester.source_name,
                f"[{status_style}]{status_text}[/{status_style}]",
                str(len(sessions)),
                _as_clickable_uri(location),
            )
            source_compact_lines.append(
                f"[table_header]{ingester.source_name}[/table_header]  "
                f"[{status_style}]{status_text}[/{status_style}]  "
                f"({len(sessions)} session{'s' if len(sessions) != 1 else ''})"
            )
        except Exception as exc:  # noqa: BLE001
            source_table.add_row(
                ingester.source_name,
                "[error]✗ Error[/error]",
                "-",
                f"[error]{str(exc)[:40]}[/error]",
            )
            source_compact_lines.append(
                f"[table_header]{ingester.source_name}[/table_header]  [error]✗ Error[/error]"
            )

    if not source_compact_lines:
        source_compact_lines.append("[dim]No configured session sources.[/dim]")

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

    knowledge_panel = Panel(knowledge_summary, title="Knowledge Base", border_style="accent")
    llm_panel = Panel(llm_summary, title="LLM Configuration", border_style="accent")
    sources_panel = Panel(source_table, title="Session Sources", border_style="accent")
    sources_compact_panel = Panel(
        "\n".join(source_compact_lines),
        title="Session Sources",
        border_style="accent",
    )
    settings_panel = Panel(settings_table, title="Settings", border_style="accent")

    if view == "knowledge":
        panels.append(knowledge_panel)
    elif view == "llm":
        panels.append(llm_panel)
    elif view == "sources":
        panels.append(sources_panel)
    elif view == "settings":
        panels.append(settings_panel)
    elif view == "console":
        pass
    elif view == "all":
        panels.extend([knowledge_panel, llm_panel, sources_panel, settings_panel])
    else:
        panels.append(Columns([knowledge_panel, sources_compact_panel], expand=True, equal=True))

    if show_slash_console:
        slash_lines = slash_output or _tui_help_lines()
        if slash_status:
            slash_lines = [f"[accent]{escape(slash_status)}[/accent]", *slash_lines]
        line_budget = 14 if view in {"console", "all", "settings"} else 6
        slash_lines = slash_lines[-line_budget:]

        panels.append(
            Panel(
                "\n".join(slash_lines),
                title="Slash Console",
                subtitle="[dim]Type /help and press Enter. Use /quit to exit.[/dim]",
                border_style="accent",
            )
        )

    return Group(*panels)


def _run_onboarding_setup(force: bool, quick: bool) -> None:
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    interactive = is_interactive_terminal() and not quick
    if not interactive and not quick:
        console.print("[dim]No interactive terminal detected; applying saved defaults.[/dim]")

    changed = ensure_repo_onboarding(
        files,
        console,
        force=force,
        interactive=interactive,
    )
    if not changed:
        console.print("[dim]Onboarding already complete for this repository.[/dim]")


def _run_setup_from_payload(payload: dict[str, object]) -> tuple[bool, list[str]]:
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    selected_agents_raw = payload.get("selected_agents")
    selected_agents = (
        [source for source in selected_agents_raw if isinstance(source, str)]
        if isinstance(selected_agents_raw, list)
        else []
    )

    temp_output = io.StringIO()
    capture_console = Console(
        file=temp_output, theme=_theme_manager.get_theme(), force_terminal=False
    )
    temperature_raw = payload.get("temperature", 0.3)
    if isinstance(temperature_raw, int | float | str):
        try:
            temperature = float(temperature_raw)
        except ValueError:
            temperature = 0.3
    else:
        temperature = 0.3

    max_tokens_raw = payload.get("max_tokens", 4096)
    if isinstance(max_tokens_raw, int | float | str):
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            max_tokens = 4096
    else:
        max_tokens = 4096

    changed = apply_repo_setup(
        files,
        capture_console,
        force=bool(payload.get("force", False)),
        repository_verified=bool(payload.get("repository_verified", False)),
        selected_agents=selected_agents,
        provider=str(payload.get("provider", "")).strip(),
        model=str(payload.get("model", "")).strip(),
        base_url=(
            None
            if str(payload.get("base_url", "")).strip().lower() in {"", "none", "null"}
            else str(payload.get("base_url", "")).strip()
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        validate=bool(payload.get("validate", False)),
        api_key=(
            str(payload.get("api_key", "")).strip()
            if isinstance(payload.get("api_key"), str)
            else None
        ),
    )

    lines = [line.strip() for line in temp_output.getvalue().splitlines() if line.strip()]
    return changed, lines[-12:]


def _run_model_config(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    validate: bool,
    output_console: Console | None = None,
) -> dict[str, object]:
    active_console = output_console or console
    _get_theme_manager()
    ensure_initialized()
    inject_stored_api_keys()
    files = get_files()

    config_dict = files.read_config()
    llm_config = dict(config_dict.get("llm", {}))

    if provider:
        llm_config["provider"] = provider
    if model:
        llm_config["model"] = model
    if base_url:
        llm_config["base_url"] = base_url
    if temperature is not None:
        llm_config["temperature"] = float(temperature)
    if max_tokens is not None:
        llm_config["max_tokens"] = int(max_tokens)

    try:
        parsed = LLMConfig(**llm_config)
    except Exception as exc:  # noqa: BLE001
        active_console.print(f"[error]Invalid LLM config: {exc}[/error]")
        raise typer.Exit(1) from None

    if validate and (
        provider or model or base_url or temperature is not None or max_tokens is not None
    ):
        valid, message = validate_provider_config(parsed)
        if not valid:
            active_console.print(f"[warning]Warning: {message}[/warning]")
        else:
            try:
                llm = create_llm_provider(parsed)
                if output_console is None:
                    success, validation_message = run_with_spinner(
                        "Validating provider connection...",
                        llm.validate,
                    )
                else:
                    success, validation_message = llm.validate()
                if success:
                    active_console.print(f"[success]✓ {validation_message}[/success]")
                else:
                    active_console.print(f"[warning]Warning: {validation_message}[/warning]")
            except Exception as exc:  # noqa: BLE001
                active_console.print(
                    f"[warning]Warning: Could not validate provider: {exc}[/warning]"
                )

    config_dict["llm"] = llm_config
    files.write_config(config_dict)

    base_url_display = llm_config.get("base_url") or "default"
    active_console.print(
        Panel.fit(
            f"Provider: {llm_config.get('provider', 'not set')}\n"
            f"Model: {llm_config.get('model', 'not set')}\n"
            f"Base URL: {base_url_display}\n"
            f"Temperature: {llm_config.get('temperature', 0.3)}\n"
            f"Max tokens: {llm_config.get('max_tokens', 4096)}",
            title="LLM Configuration Updated",
        )
    )
    return llm_config


def _run_model_config_for_tui(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    validate: bool,
) -> list[str]:
    temp_output = io.StringIO()
    capture_console = Console(
        file=temp_output, theme=_theme_manager.get_theme(), force_terminal=False
    )
    _run_model_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        validate=validate,
        output_console=capture_console,
    )
    return [line.strip() for line in temp_output.getvalue().splitlines() if line.strip()][-12:]


@app.command(hidden=True)
def onboard(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Run onboarding even if this repository is already configured.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Apply onboarding defaults without interactive prompts.",
    ),
):
    """Run onboarding for this repository and persist local provider credentials."""
    console.print("[warning]'onboard' is deprecated; use 'config setup' instead.[/warning]")
    _run_onboarding_setup(force=force, quick=quick)


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
    onboarding: bool = typer.Option(
        True,
        "--onboarding/--no-onboarding",
        help="Run onboarding setup before launching the dashboard.",
    ),
    force_onboarding: bool = typer.Option(
        False,
        "--force-onboarding",
        help="Run onboarding even if this repository is already configured.",
    ),
):
    """Start a live terminal UI dashboard for agent-recall."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()
    interactive_shell = is_interactive_terminal()

    if iterations is not None and iterations < 1:
        console.print("[error]--iterations must be >= 1[/error]")
        raise typer.Exit(1)

    if onboarding:
        ensure_repo_onboarding(
            files,
            console,
            force=force_onboarding,
            interactive=interactive_shell,
        )
    elif not is_repo_onboarding_complete(files):
        console.print(
            "[warning]Onboarding is incomplete for this repository. "
            "Run 'agent-recall config setup' to configure providers and sources.[/warning]"
        )

    # Show splash animation first (unless disabled)
    if not no_splash:
        console.clear()
        print_banner(console, _theme_manager, animated=True, delay=splash_delay)
        time.sleep(0.8)  # Pause to admire the banner

    if iterations is not None:
        for index in range(iterations):
            console.print(
                _build_tui_dashboard(
                    all_cursor_workspaces=all_cursor_workspaces,
                    include_banner_header=True,
                    view="overview",
                    refresh_seconds=refresh_seconds,
                    show_slash_console=False,
                )
            )
            if index < iterations - 1:
                time.sleep(refresh_seconds)
        return

    if not interactive_shell:
        console.print(
            "[error]The Textual TUI requires an interactive terminal. "
            "Use a terminal session or run with --iterations for non-interactive checks.[/error]"
        )
        raise typer.Exit(1)

    try:
        from agent_recall.cli.textual_tui import AgentRecallTextualApp
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Textual TUI unavailable: {exc}[/error]")
        raise typer.Exit(1) from None

    try:
        app_instance = AgentRecallTextualApp(
            render_dashboard=_build_tui_dashboard,
            execute_command=lambda raw: _execute_tui_slash_command(_normalize_tui_command(raw)),
            run_setup_payload=_run_setup_from_payload,
            run_model_config=_run_model_config_for_tui,
            model_defaults_provider=lambda: get_files().read_config().get("llm", {}),
            setup_defaults_provider=lambda: get_onboarding_defaults(get_files()),
            discover_models=lambda provider, base_url, api_key_env: discover_provider_models(
                provider,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_seconds=4.0,
            ),
            providers=get_available_providers(),
            cli_commands=_collect_cli_commands_for_palette(),
            rich_theme=_theme_manager.get_theme(),
            initial_view="overview",
            refresh_seconds=refresh_seconds,
            all_cursor_workspaces=all_cursor_workspaces,
        )
        app_instance.run()
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


@app.command("config-llm", hidden=True)
def config_llm(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=("LLM provider: anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible"),
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
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (0.0 to 2.0)",
        min=0.0,
        max=2.0,
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum output tokens (>0)",
        min=1,
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate configuration after setting",
    ),
):
    """Backwards-compatible alias for `config model`."""
    console.print("[warning]'config-llm' is deprecated; use 'config model' instead.[/warning]")
    _run_model_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        validate=validate,
    )


@config_app.command("setup")
def config_setup(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Run setup even if this repository is already configured.",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Apply setup defaults without interactive prompts.",
    ),
):
    """Run repository setup and credential onboarding."""
    _run_onboarding_setup(force=force, quick=quick)


@config_app.command("model")
def config_model(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=("LLM provider: anthropic, openai, google, ollama, vllm, lmstudio, openai-compatible"),
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
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (0.0 to 2.0)",
        min=0.0,
        max=2.0,
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum output tokens (>0)",
        min=1,
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate configuration after setting",
    ),
):
    """Configure model/provider settings."""
    _run_model_config(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        validate=validate,
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
        "  agent-recall config model --provider anthropic --model claude-sonnet-4-20250514"
    )
    console.print("  agent-recall config model --provider ollama --model llama3.1")
    console.print("  agent-recall config model --provider lmstudio --model local-model")
    console.print(
        "  agent-recall config model --provider openai-compatible --base-url "
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
app.add_typer(config_app, name="config")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
