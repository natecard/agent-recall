from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shlex
import shutil
import sys
import textwrap
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme
from typer.main import get_command as get_typer_command
from typer.testing import CliRunner

from agent_recall.cli.banner import print_banner
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.background_sync import BackgroundSyncManager
from agent_recall.core.compact import CompactionEngine
from agent_recall.core.config import load_config
from agent_recall.core.context import ContextAssembler
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.onboarding import (
    API_KEY_ENV_BY_PROVIDER,
    apply_repo_setup,
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
from agent_recall.ingest.sources import (
    VALID_SOURCE_NAMES,
    normalize_source_name,
    resolve_source_location_hint,
)
from agent_recall.llm import (
    Message,
    create_llm_provider,
    ensure_provider_dependency,
    get_available_providers,
    validate_provider_config,
)
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LLMConfig, RetrievalConfig, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage

app = typer.Typer(help="Agent Memory System - Persistent knowledge for AI coding agents")
_slash_runner = CliRunner()
config_app = typer.Typer(help="Manage onboarding and model configuration")
_ansi_escape_pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_box_drawing_chars = set("│┃─━┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿")
_box_drawing_translation = str.maketrans(
    {
        "│": " ",
        "┃": " ",
        "─": " ",
        "━": " ",
        "┄": " ",
        "┅": " ",
        "┆": " ",
        "┇": " ",
        "┈": " ",
        "┉": " ",
        "┊": " ",
        "┋": " ",
        "┌": " ",
        "┍": " ",
        "┎": " ",
        "┏": " ",
        "┐": " ",
        "┑": " ",
        "┒": " ",
        "┓": " ",
        "└": " ",
        "┕": " ",
        "┖": " ",
        "┗": " ",
        "┘": " ",
        "┙": " ",
        "┚": " ",
        "┛": " ",
        "├": " ",
        "┝": " ",
        "┞": " ",
        "┟": " ",
        "┠": " ",
        "┡": " ",
        "┢": " ",
        "┣": " ",
        "┤": " ",
        "┥": " ",
        "┦": " ",
        "┧": " ",
        "┨": " ",
        "┩": " ",
        "┪": " ",
        "┫": " ",
        "┬": " ",
        "┭": " ",
        "┮": " ",
        "┯": " ",
        "┰": " ",
        "┱": " ",
        "┲": " ",
        "┳": " ",
        "┴": " ",
        "┵": " ",
        "┶": " ",
        "┷": " ",
        "┸": " ",
        "┹": " ",
        "┺": " ",
        "┻": " ",
        "┼": " ",
        "┽": " ",
        "┾": " ",
        "┿": " ",
    }
)

# Initialize theme manager and console
_theme_manager = ThemeManager(DEFAULT_THEME)
console = Console(theme=_theme_manager.get_theme())

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"
T = TypeVar("T")
SOURCE_CHOICES_TEXT = ", ".join(VALID_SOURCE_NAMES)

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
  index_decision_entries: true
  index_decision_min_confidence: 0.7
  index_exploration_entries: true
  index_exploration_min_confidence: 0.7
  index_narrative_entries: false
  index_narrative_min_confidence: 0.8
  archive_sessions_older_than_days: 30

retrieval:
  backend: fts5
  top_k: 5
  fusion_k: 60
  rerank_enabled: false
  rerank_candidate_k: 20
  embedding_enabled: false
  embedding_dimensions: 64

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


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    """Get a database connection."""
    ensure_initialized()
    config = load_config(AGENT_DIR)
    return create_storage_backend(config, DB_PATH)


def get_files() -> FileStorage:
    ensure_initialized()
    return FileStorage(AGENT_DIR)


def get_llm():
    inject_stored_api_keys()
    files = get_files()
    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    dependency_ok, dependency_message = ensure_provider_dependency(
        llm_config.provider,
        auto_install=True,
    )
    if not dependency_ok:
        raise RuntimeError(dependency_message or "Provider dependency setup failed.")
    if dependency_message:
        console.print(f"[dim]{dependency_message}[/dim]")
    return create_llm_provider(llm_config)


def _load_retrieval_config(files: FileStorage) -> RetrievalConfig:
    config_dict = files.read_config()
    retrieval_data = config_dict.get("retrieval", {})
    if retrieval_data is None:
        retrieval_data = {}
    if not isinstance(retrieval_data, dict):
        raise ValueError("Invalid retrieval configuration: 'retrieval' must be a mapping.")
    try:
        return RetrievalConfig.model_validate(retrieval_data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid retrieval configuration: {exc}") from exc


def _normalize_retrieval_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"fts5", "hybrid"}:
        return normalized
    raise ValueError("Invalid retrieval backend. Use 'fts5' or 'hybrid'.")


def _build_retriever(
    storage: Storage,
    files: FileStorage,
    *,
    backend: str | None = None,
    fusion_k: int | None = None,
    rerank: bool | None = None,
    rerank_candidate_k: int | None = None,
) -> tuple[Retriever, RetrievalConfig]:
    retrieval_cfg = _load_retrieval_config(files)
    selected_backend = (
        _normalize_retrieval_backend(backend) if backend is not None else retrieval_cfg.backend
    )
    selected_fusion_k = fusion_k if fusion_k is not None else retrieval_cfg.fusion_k
    selected_rerank = retrieval_cfg.rerank_enabled if rerank is None else rerank
    selected_rerank_candidate_k = (
        rerank_candidate_k if rerank_candidate_k is not None else retrieval_cfg.rerank_candidate_k
    )
    retriever = Retriever(
        storage,
        backend=selected_backend,
        fusion_k=selected_fusion_k,
        rerank_enabled=selected_rerank,
        rerank_candidate_k=selected_rerank_candidate_k,
    )
    return retriever, retrieval_cfg


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


def _format_session_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.strftime("%Y-%m-%d %H:%M UTC")


def _compact_session_ref(session_id: str, max_chars: int = 18) -> str:
    cleaned = session_id.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    head = max_chars // 2 - 1
    tail = max_chars - head - 1
    return f"{cleaned[:head]}…{cleaned[-tail:]}"


def _conversation_table_mode(width: int) -> str:
    if width >= 150:
        return "wide"
    if width >= 120:
        return "medium"
    return "compact"


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
        "[dim]/compact[/dim] - Run knowledge synthesis now",
        "[dim]/run[/dim] - Alias for /sync (includes synthesis by default)",
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


def _normalize_tui_output_line(line: str) -> str:
    without_ansi = _ansi_escape_pattern.sub("", line)
    without_box_chars = without_ansi.translate(_box_drawing_translation)
    return " ".join(without_box_chars.split())


def _strip_ansi_control_sequences(line: str) -> str:
    return _ansi_escape_pattern.sub("", line)


def _looks_like_table_output(lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    table_like_rows = 0
    for line in lines:
        if any(char in _box_drawing_chars for char in line):
            table_like_rows += 1
    return table_like_rows >= 2


def _execute_tui_slash_command(
    raw: str,
    *,
    terminal_width: int | None = None,
    terminal_height: int | None = None,
) -> tuple[bool, list[str]]:
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

    original_parts = list(parts)
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

    if command_name in {"tui", "open"}:
        return False, ["[warning]/open is already running.[/warning]"]

    # User-facing alias: "run" communicates intent better than "compact".
    if command_name == "run":
        parts = ["sync", *parts[1:]]
        command_name = "sync"

    invoke_env: dict[str, str] | None = None
    if terminal_width is not None or terminal_height is not None:
        invoke_env = {}
        if terminal_width is not None and terminal_width > 0:
            invoke_env["COLUMNS"] = str(int(terminal_width))
        if terminal_height is not None and terminal_height > 0:
            invoke_env["LINES"] = str(int(terminal_height))

    runner_terminal_width = (
        terminal_width if terminal_width is not None and terminal_width > 0 else 120
    )
    result = _slash_runner.invoke(
        app,
        parts,
        env=invoke_env,
        terminal_width=max(runner_terminal_width, 80),
    )
    command_label_parts = original_parts if original_parts and original_parts[0] == "run" else parts
    command_label = "/" + " ".join(command_label_parts)

    lines: list[str] = []
    if result.exit_code == 0:
        lines.append(f"[success]✓ {escape(command_label)}[/success]")
    else:
        lines.append(f"[error]✗ {escape(command_label)} (exit {result.exit_code})[/error]")

    output = result.output.strip()
    if output:
        raw_output_lines = output.splitlines()
        ansi_stripped_lines = [
            _strip_ansi_control_sequences(raw_line).rstrip()
            for raw_line in raw_output_lines
            if raw_line.strip()
        ]
        if _looks_like_table_output(ansi_stripped_lines):
            output_lines = ansi_stripped_lines
        else:
            meaningful_lines = []
            for raw_line in ansi_stripped_lines:
                normalized = _normalize_tui_output_line(raw_line)
                if not normalized:
                    continue
                if not any(char.isalnum() for char in normalized):
                    continue
                meaningful_lines.append(normalized)
            output_lines = meaningful_lines if meaningful_lines else ansi_stripped_lines
        for line in output_lines:
            lines.append(f"[dim]{escape(line)}[/dim]")

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

    get_storage()

    console.print(
        Panel.fit(
            "[success]✓ Initialized .agent/ directory[/success]\n\n"
            "Files created:\n"
            "  • config.yaml - Configuration\n"
            "  • GUARDRAILS.md - Hard rules and warnings\n"
            "  • STYLE.md - Patterns and preferences\n"
            "  • RECENT.md - Session summaries\n"
            "  • state.db - Session and log storage\n\n"
            "Next: [bold]agent-recall open[/bold] (TUI onboarding + dashboard)",
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
    try:
        retriever, retrieval_cfg = _build_retriever(storage, files)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=retrieval_cfg.top_k,
    )

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
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1,
        help="Override maximum number of retrieved chunks (default from config)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override retrieval backend (fts5 or hybrid)",
    ),
    fusion_k: int | None = typer.Option(
        None,
        "--fusion-k",
        min=1,
        help="Override hybrid fusion constant (default from config)",
    ),
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Override reranking behavior (default from config)",
    ),
    rerank_candidate_k: int | None = typer.Option(
        None,
        "--rerank-candidate-k",
        min=1,
        help="Override rerank candidate pool size (default from config)",
    ),
):
    """Output current context (guardrails, style, recent, relevant)."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    try:
        retriever, retrieval_cfg = _build_retriever(
            storage,
            files,
            backend=backend,
            fusion_k=fusion_k,
            rerank=rerank,
            rerank_candidate_k=rerank_candidate_k,
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=top_k if top_k is not None else retrieval_cfg.top_k,
    )
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


@app.command("refresh-context")
def refresh_context(
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Task for relevant retrieval (defaults to active session task)",
    ),
    output_dir: Path = typer.Option(
        AGENT_DIR / "context",
        "--output-dir",
        help="Directory where refreshed context bundle files are written",
    ),
):
    """Refresh context bundle files for active task and repository state."""
    _get_theme_manager()
    storage = get_storage()
    files = get_files()

    session_mgr = SessionManager(storage)
    active = session_mgr.get_active()
    resolved_task = task or (active.task if active else None)

    try:
        retriever, retrieval_cfg = _build_retriever(storage, files)
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None
    context_asm = ContextAssembler(
        storage,
        files,
        retriever=retriever,
        retrieval_top_k=retrieval_cfg.top_k,
    )
    markdown_path = output_dir / "context.md"
    json_path = output_dir / "context.json"

    retry_attempts = 3
    retry_backoff_seconds = 1.0
    diagnostics: list[str] = []
    output = ""

    for attempt in range(1, retry_attempts + 1):
        try:
            output = context_asm.assemble(task=resolved_task)

            output_dir.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(output)
            payload = {
                "task": resolved_task,
                "active_session_id": str(active.id) if active else None,
                "repo_path": str(Path.cwd().resolve()),
                "refreshed_at": datetime.now(UTC).isoformat(),
                "context": output,
            }
            json_path.write_text(json.dumps(payload, indent=2))
            break
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                f"Attempt {attempt}/{retry_attempts} failed: {type(exc).__name__}: {exc}"
            )
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * attempt)
                continue

            console.print(
                f"[error]Context refresh failed after {retry_attempts} attempt(s).[/error]"
            )
            for detail in diagnostics:
                console.print(f"[dim]- {detail}[/dim]")
            raise typer.Exit(1) from None

    lines = [
        "[success]✓ Context bundle refreshed[/success]",
        f"  Markdown: {markdown_path}",
        f"  JSON:     {json_path}",
    ]
    if diagnostics:
        lines.append(f"  Retries:  {len(diagnostics)}")
    if resolved_task:
        lines.append(f"  Task:     {resolved_task}")
    else:
        lines.append("  Task:     none (set --task or start a session for task retrieval)")

    console.print("\n".join(lines))


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
            f"LLM requests: {results.get('llm_requests', 0)} "
            f"(responses: {results.get('llm_responses', 0)})\n"
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
        help=f"Only sync from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    since_days: int | None = typer.Option(
        None,
        "--since-days",
        "-d",
        help="Only sync sessions from the last N days",
    ),
    session_id: list[str] = typer.Option(
        None,
        "--session-id",
        help="Only sync specific source session ID(s); repeat option to include multiple",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit sync to the most recent N discovered sessions",
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
    normalized_source = normalize_source_name(source) if source else None
    selected_sources = None if source else _get_repo_selected_sources(files)
    sources = [normalized_source] if normalized_source else selected_sources
    selected_session_ids = [item.strip() for item in (session_id or []) if item.strip()]
    session_ids = selected_session_ids or None

    if source:
        ingester_source = normalized_source or source
        try:
            ingesters = [
                get_ingester(
                    ingester_source,
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

    def _sync_progress_logger(event: dict[str, Any]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "extraction_session_started":
            source_name = str(event.get("source", "?"))
            source_session_id = str(event.get("session_id", "?"))
            total_messages = event.get("messages_total")
            message_text = (
                str(int(total_messages)) if isinstance(total_messages, int | float) else "?"
            )
            console.print(
                f"[dim]{source_name}:{source_session_id} extraction started "
                f"({message_text} messages).[/dim]"
            )
            return

        if event_name != "extraction_batch_complete":
            return

        source_name = str(event.get("source", "?"))
        source_session_id = str(event.get("session_id", "?"))
        processed = int(event.get("messages_processed", 0))
        total = int(event.get("messages_total", 0))
        batch_index = int(event.get("batch_index", 0))
        batch_count = int(event.get("batch_count", 0))
        learnings = int(event.get("batch_learnings", 0))
        console.print(
            f"[dim]{source_name}:{source_session_id} sent {processed}/{total} messages "
            f"to LLM (batch {batch_index}/{batch_count}, learnings={learnings}).[/dim]"
        )

    auto_sync = AutoSync(storage, files, llm, ingesters)
    if hasattr(auto_sync, "progress_callback"):
        auto_sync.progress_callback = _sync_progress_logger

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
                        session_ids=session_ids,
                        max_sessions=max_sessions,
                        force_compact=force,
                    )
                )
            else:
                results = asyncio.run(
                    auto_sync.sync(
                        since=since,
                        sources=sources,
                        session_ids=session_ids,
                        max_sessions=max_sessions,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[error]Sync failed: {exc}[/error]")
            raise typer.Exit(1) from None

    lines = [""]
    if selected_sources and not source:
        lines.append(f"Using configured sources: {', '.join(selected_sources)}")
    if session_ids:
        lines.append(f"Requested session IDs: {len(session_ids)}")
    if max_sessions is not None:
        lines.append(f"Max sessions: {max_sessions}")
    lines.append(f"Sessions discovered: {results['sessions_discovered']}")
    lines.append(f"Sessions processed:  {results['sessions_processed']}")
    if int(results["sessions_skipped"]) > 0:
        lines.append(
            f"Sessions skipped:    {results['sessions_skipped']} "
            f"(already processed: {int(results.get('sessions_already_processed', 0))}, "
            f"empty: {int(results.get('empty_sessions', 0))})"
        )
    lines.append(f"Learnings extracted: {results['learnings_extracted']}")
    lines.append(f"LLM extraction requests: {results.get('llm_requests', 0)}")
    if session_ids and int(results["sessions_processed"]) == 0:
        already_processed = int(results.get("sessions_already_processed", 0))
        if already_processed == int(results["sessions_discovered"]) and already_processed > 0:
            lines.append(
                "[warning]Selected sessions were already processed; ingestion did not "
                "rerun.[/warning]"
            )
            lines.append(
                "[dim]Use `agent-recall reset-sync --session-id <id>` to reprocess a session.[/dim]"
            )

    if verbose and results.get("by_source"):
        lines.append("")
        lines.append("[bold]By source:[/bold]")
        by_source = results["by_source"]
        for source_name, source_stats in by_source.items():
            lines.append(
                f"  {source_name}: {source_stats['processed']}/{source_stats['discovered']} "
                f"processed, {source_stats['learnings']} learnings"
            )
            lines.append(f"    [dim]LLM extraction requests: {source_stats['llm_batches']}[/dim]")
            if int(source_stats.get("already_processed", 0)) > 0:
                lines.append(
                    f"    [dim]{source_stats['already_processed']} session(s) skipped as already "
                    "processed[/dim]"
                )
            if int(source_stats.get("empty", 0)) > 0:
                lines.append(
                    f"    [dim]{source_stats['empty']} discovered session(s) had no parseable "
                    "messages and were skipped[/dim]"
                )
    session_diagnostics = results.get("session_diagnostics") or []
    if verbose and session_diagnostics:
        lines.append("")
        lines.append("[bold]Session diagnostics:[/bold]")
        for item in session_diagnostics[:30]:
            source_name = str(item.get("source", "?"))
            source_session_id = str(item.get("session_id", "?"))
            status = str(item.get("status", "unknown"))
            message_count = item.get("message_count")
            learnings = int(item.get("learnings_extracted", 0))
            message_text = (
                f"messages={message_count}" if isinstance(message_count, int) else "messages=?"
            )
            lines.append(
                f"  {source_name}:{source_session_id}  status={status}  "
                f"{message_text}  learnings={learnings}"
            )
            warning = item.get("warning")
            if warning:
                lines.append(f"    [warning]{warning}[/warning]")
        if len(session_diagnostics) > 30:
            lines.append(f"  [dim]... and {len(session_diagnostics) - 30} more session(s)[/dim]")

    if "compaction" in results:
        comp = results["compaction"]
        lines.append("")
        lines.append("[bold]Knowledge synthesis:[/bold]")
        lines.append(
            f"  LLM requests:       {comp.get('llm_requests', 0)} "
            f"(responses: {comp.get('llm_responses', 0)})"
        )
        lines.append(f"  Guardrails updated: {'✓' if comp.get('guardrails_updated') else '-'}")
        lines.append(f"  Style updated:      {'✓' if comp.get('style_updated') else '-'}")
        lines.append(f"  Recent updated:     {'✓' if comp.get('recent_updated') else '-'}")
        lines.append(f"  Chunks indexed:     {comp.get('chunks_indexed', 0)}")
        changed_files = []
        if comp.get("guardrails_updated"):
            changed_files.append(".agent/GUARDRAILS.md")
        if comp.get("style_updated"):
            changed_files.append(".agent/STYLE.md")
        if comp.get("recent_updated"):
            changed_files.append(".agent/RECENT.md")
        lines.append(
            "  Updated files:      " + (", ".join(changed_files) if changed_files else "none")
        )
    elif compact:
        lines.append("")
        lines.append("[bold]Knowledge synthesis:[/bold]")
        lines.append("  Status: skipped (no new learnings extracted)")
        lines.append("[dim]Use `--force` to run synthesis without new ingestion.[/dim]")

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


@app.command("sync-background")
def sync_background(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Only sync from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit sync to the most recent N discovered sessions",
    ),
    no_compact: bool = typer.Option(
        False,
        "--no-compact",
        help="Skip compaction after sync",
    ),
):
    """Run sync in background with safe locking (prevents duplicate syncs)."""
    storage = get_storage()
    files = get_files()

    _get_theme_manager()

    # Check if already running
    bg_manager = BackgroundSyncManager(storage, files, auto_sync=None)
    can_start, reason = bg_manager.can_start_sync()

    if not can_start:
        console.print(f"[warning]Cannot start background sync: {reason}[/warning]")
        raise typer.Exit(1)

    try:
        llm = get_llm()
    except Exception as exc:
        console.print(f"[error]LLM configuration error: {exc}[/error]")
        raise typer.Exit(1) from None

    selected_sources = _get_repo_selected_sources(files)
    sources = None
    if source:
        sources = [normalize_source_name(source)]
    elif selected_sources:
        sources = selected_sources

    ingesters = get_default_ingesters()
    ingesters = _filter_ingesters_by_sources(ingesters, selected_sources)
    auto_sync = AutoSync(storage, files, llm, ingesters)

    # Create manager with actual auto_sync
    bg_manager = BackgroundSyncManager(storage, files, auto_sync)

    # Run sync
    result = asyncio.run(
        bg_manager.run_sync(
            sources=sources,
            max_sessions=max_sessions,
            compact=not no_compact,
        )
    )

    if result.was_already_running:
        console.print("[warning]Sync already running in another process[/warning]")
        raise typer.Exit(1)

    if result.success:
        console.print(
            f"[success]✓ Background sync complete[/success]\n"
            f"  Sessions processed: {result.sessions_processed}\n"
            f"  Learnings extracted: {result.learnings_extracted}"
        )
    else:
        console.print("[error]Sync failed[/error]")
        if result.error_message:
            console.print(f"[error]{result.error_message}[/error]")
        if result.diagnostics:
            console.print("[dim]Failure diagnostics:[/dim]")
            for detail in result.diagnostics:
                console.print(f"[dim]- {detail}[/dim]")
        raise typer.Exit(1)


@app.command("sessions")
def sessions(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Only list sessions from specific source: {SOURCE_CHOICES_TEXT}",
    ),
    since_days: int | None = typer.Option(
        None,
        "--since-days",
        "-d",
        help="Only list sessions from the last N days",
    ),
    session_id: list[str] = typer.Option(
        None,
        "--session-id",
        help="Only include specific source session ID(s); repeat option to include multiple",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit list to the most recent N discovered sessions",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
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
    """List discoverable agent sessions with source IDs and inferred titles."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()

    since = datetime.now(UTC) - timedelta(days=since_days) if since_days else None
    normalized_source = normalize_source_name(source) if source else None
    selected_sources = None if source else _get_repo_selected_sources(files)
    sources = [normalized_source] if normalized_source else selected_sources
    selected_session_ids = [item.strip() for item in (session_id or []) if item.strip()]
    session_ids = selected_session_ids or None

    if source:
        ingester_source = normalized_source or source
        try:
            ingesters = [
                get_ingester(
                    ingester_source,
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

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = run_with_spinner(
        "Discovering sessions...",
        lambda: auto_sync.list_sessions(
            since=since,
            sources=sources,
            session_ids=session_ids,
            max_sessions=max_sessions,
        ),
    )

    output_format = format.strip().lower()
    if output_format == "json":
        payload = {
            "sessions_discovered": results["sessions_discovered"],
            "by_source": results["by_source"],
            "errors": results["errors"],
            "sessions": [
                {
                    **{
                        key: value
                        for key, value in session_row.items()
                        if key not in {"started_at", "ended_at", "session_path", "project_path"}
                    },
                    "started_at": (
                        session_row["started_at"].isoformat()
                        if isinstance(session_row.get("started_at"), datetime)
                        else None
                    ),
                    "ended_at": (
                        session_row["ended_at"].isoformat()
                        if isinstance(session_row.get("ended_at"), datetime)
                        else None
                    ),
                    "session_path": str(session_row["session_path"]),
                    "project_path": (
                        str(session_row["project_path"])
                        if isinstance(session_row.get("project_path"), Path)
                        else None
                    ),
                }
                for session_row in results["sessions"]
            ],
        }
        console.print(json.dumps(payload, indent=2))
        return

    if output_format != "table":
        console.print("[error]Invalid format. Use 'table' or 'json'.[/error]")
        raise typer.Exit(1)

    if selected_sources and not source:
        console.print(f"[dim]Configured sources: {', '.join(selected_sources)}[/dim]")
    if session_ids:
        console.print(f"[dim]Requested session IDs: {len(session_ids)}[/dim]")
    if max_sessions is not None:
        console.print(f"[dim]Max sessions: {max_sessions}[/dim]")

    table = Table(
        title="Discovered Sessions",
        box=box.SQUARE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Source", style="table_header")
    table.add_column("Session ID", overflow="fold")
    table.add_column("Title", overflow="fold")
    table.add_column("Started", overflow="fold")
    table.add_column("Messages", justify="right")
    table.add_column("Processed", justify="center")

    for session_row in results["sessions"]:
        title_text = str(session_row.get("title") or "-")
        table.add_row(
            str(session_row["source"]),
            str(session_row["session_id"]),
            title_text,
            _format_session_time(session_row.get("started_at")),
            str(session_row.get("message_count", 0)),
            "[success]✓[/success]" if session_row.get("processed") else "[warning]X[/warning]",
        )

    console.print(table)

    errors = results.get("errors") or []
    if errors:
        warning_lines = ["", f"[warning]Warnings: {len(errors)}[/warning]"]
        for error in errors[:10]:
            warning_lines.append(f"[dim]- {error}[/dim]")
        if len(errors) > 10:
            warning_lines.append(f"[dim]... and {len(errors) - 10} more[/dim]")
        console.print("\n".join(warning_lines))


@app.command()
def sources(
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Show Cursor sessions from all workspaces, not only current repo match",
    ),
    max_sessions: int | None = typer.Option(
        None,
        "--max-sessions",
        "-n",
        min=1,
        help="Limit detailed session list to the most recent N sessions",
    ),
):
    """Show source status and discovered sessions/conversations."""
    _get_theme_manager()  # Ensure theme is loaded
    ensure_initialized()
    storage = get_storage()
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
                else:
                    hint = resolve_source_location_hint(ingester)
                    location = str(hint) if hint else "Unknown"

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

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    session_results = run_with_spinner(
        "Loading session details...",
        lambda: auto_sync.list_sessions(max_sessions=max_sessions),
    )

    sessions = session_results.get("sessions", [])
    if not sessions:
        console.print("[dim]No session conversations discovered.[/dim]")
        return

    mode = _conversation_table_mode(console.width)
    sessions_table = Table(
        title="Discovered Conversations",
        box=box.SQUARE,
        show_lines=True,
        expand=True,
    )
    sessions_table.add_column("#", justify="right", style="table_header", no_wrap=True, width=3)
    sessions_table.add_column("Conversation", overflow="fold")

    if mode == "wide":
        sessions_table.add_column("Started", no_wrap=True, width=20)
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)
        sessions_table.add_column("Ref", overflow="fold", no_wrap=True, width=13)
    elif mode == "medium":
        sessions_table.add_column("Started", no_wrap=True, width=20)
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)
    else:
        sessions_table.add_column("Messages", justify="right", no_wrap=True, width=8)
        sessions_table.add_column("Processed", justify="center", no_wrap=True, width=9)

    for index, session_row in enumerate(sessions, start=1):
        title = str(session_row.get("title") or "").strip()
        conversation = title if title else "Untitled conversation"
        processed = (
            "[success]✓[/success]" if session_row.get("processed") else "[warning]X[/warning]"
        )

        if mode == "wide":
            sessions_table.add_row(
                str(index),
                conversation,
                _format_session_time(session_row.get("started_at")),
                str(session_row.get("message_count", 0)),
                processed,
                _compact_session_ref(str(session_row["session_id"])),
            )
        elif mode == "medium":
            sessions_table.add_row(
                str(index),
                conversation,
                _format_session_time(session_row.get("started_at")),
                str(session_row.get("message_count", 0)),
                processed,
            )
        else:
            sessions_table.add_row(
                str(index),
                conversation,
                str(session_row.get("message_count", 0)),
                processed,
            )

    console.print(sessions_table)

    errors = session_results.get("errors") or []
    if errors:
        console.print(f"[warning]Warnings: {len(errors)}[/warning]")
        for error in errors[:10]:
            console.print(f"[dim]- {error}[/dim]")
        if len(errors) > 10:
            console.print(f"[dim]... and {len(errors) - 10} more[/dim]")


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

    # Add background sync status
    lines.append("")
    lines.append("[bold]Background Sync:[/bold]")
    bg_manager = BackgroundSyncManager(storage, files, None)
    bg_status = bg_manager.get_status()
    if bg_status.is_running:
        lines.append(f"  Status: [warning]Running (PID: {bg_status.pid})[/warning]")
    else:
        lines.append("  Status: [dim]Idle[/dim]")
    if bg_status.started_at:
        lines.append(f"  Last started:  {bg_status.started_at.strftime('%Y-%m-%d %H:%M UTC')}")
    if bg_status.completed_at:
        lines.append(f"  Last completed: {bg_status.completed_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"  Last sessions:  {bg_status.sessions_processed}")
        lines.append(f"  Last learnings: {bg_status.learnings_extracted}")
    if bg_status.error_message:
        lines.append(f"  [error]Last error: {bg_status.error_message}[/error]")

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
    provider_key = llm_config.provider.strip().lower()
    api_key_env = API_KEY_ENV_BY_PROVIDER.get(provider_key)
    api_key_set = bool(os.environ.get(api_key_env, "").strip()) if api_key_env else True
    api_key_set_display = "Yes" if api_key_set else "No"
    selected_sources = _get_repo_selected_sources(files)
    repo_root = _resolve_repo_root_for_display()
    repo_name = repo_root.name
    configured_agents = ", ".join(selected_sources) if selected_sources else "all"

    # Keep long source lists readable in half-width panels (all view).
    if view == "all":
        wrap_width = max(24, min(44, (max(console.size.width, 100) // 2) - 24))
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

    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
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

    # Create banner header
    banner_renderer = BannerRenderer(console, _theme_manager)
    header_text = banner_renderer.get_tui_header_text()

    knowledge_summary = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    knowledge_summary.add_column("Item", style="table_header", width=12, no_wrap=True)
    knowledge_summary.add_column("Value", overflow="fold")
    knowledge_summary.add_row("Repository", repo_name)
    knowledge_summary.add_row("Processed", str(stats.get("processed_sessions", 0)))
    knowledge_summary.add_row("Logs", str(stats.get("log_entries", 0)))
    knowledge_summary.add_row("Chunks", str(stats.get("chunks", 0)))
    knowledge_summary.add_row("GUARDRAILS", f"{len(guardrails):,} chars")
    knowledge_summary.add_row("STYLE", f"{len(style):,} chars")
    knowledge_summary.add_row("RECENT", f"{len(recent):,} chars")

    llm_base_url_display = llm_config.base_url or "default"
    llm_summary = Table(
        expand=True,
        box=box.SIMPLE,
        pad_edge=False,
        collapse_padding=True,
    )
    if view == "all":
        llm_summary.add_column("Setting", style="table_header", width=12, no_wrap=True)
        llm_summary.add_column("Value", overflow="fold")
        llm_summary.add_row("Provider", llm_config.provider)
        llm_summary.add_row("Model", llm_config.model)
        llm_summary.add_row("Temperature", str(llm_config.temperature))
        llm_summary.add_row("Max tokens", str(llm_config.max_tokens))
        llm_summary.add_row("API Key Set", api_key_set_display)
    else:
        llm_summary.add_column("Provider", style="table_header")
        llm_summary.add_column("Model", style="table_header", overflow="fold")
        llm_summary.add_column("Base URL", style="table_header", overflow="fold")
        llm_summary.add_column("Temperature", style="table_header")
        llm_summary.add_column("Max tokens", style="table_header")
        llm_summary.add_column("API Key Set", style="table_header")
        llm_summary.add_row(
            llm_config.provider,
            llm_config.model,
            llm_base_url_display,
            str(llm_config.temperature),
            str(llm_config.max_tokens),
            api_key_set_display,
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
    settings_table.add_row("Theme", _theme_manager.get_theme_name())
    if view != "all":
        settings_table.add_row("Interactive shell", "yes" if is_interactive_terminal() else "no")
        settings_table.add_row("Repository", repo_name)
    settings_table.add_row(
        "Active agents" if view == "all" else "Configured agents",
        active_agents_wrapped if view == "all" else configured_agents_wrapped,
    )

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    panels = []

    if include_banner_header:
        panels.append(
            Panel(
                header_text,
                title=f"[dim]Updated {now_text}[/dim]",
                subtitle="[dim]Press Ctrl+Q to exit[/dim]",
                border_style="banner.border",
            )
        )

    knowledge_panel = Panel(knowledge_summary, title="Knowledge Base", border_style="accent")
    llm_panel = Panel(llm_summary, title="LLM Configuration", border_style="accent")
    sources_panel = Panel(
        Group(source_table, f"[dim]Last Synced:[/dim] {last_synced_display}"),
        title="Session Sources",
        border_style="accent",
    )
    source_compact_lines.append(f"[dim]Last Synced:[/dim] {last_synced_display}")
    sources_compact_panel = Panel(
        "\n".join(source_compact_lines),
        title="Session Sources",
        border_style="accent",
    )
    settings_panel = Panel(settings_table, title="Settings", border_style="accent")

    def _two_panel_row(left: Panel, right: Panel) -> Table:
        row = Table.grid(expand=True, padding=(0, 2))
        row.add_column(ratio=1)
        row.add_column(ratio=1)
        row.add_row(left, right)
        return row

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
        panels.extend(
            [
                _two_panel_row(knowledge_panel, llm_panel),
                _two_panel_row(sources_panel, settings_panel),
            ]
        )
    else:
        panels.append(_two_panel_row(knowledge_panel, sources_compact_panel))

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

    dependency_ok, dependency_message = ensure_provider_dependency(
        parsed.provider,
        auto_install=True,
    )
    if not dependency_ok:
        active_console.print(
            f"[error]Provider dependency setup failed: {dependency_message}[/error]"
        )
        raise typer.Exit(1)
    if dependency_message:
        active_console.print(f"[dim]{dependency_message}[/dim]")

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


def _list_sessions_for_tui_picker(
    max_sessions: int = 200,
    *,
    all_cursor_workspaces: bool = False,
) -> list[dict[str, object]]:
    ensure_initialized()
    storage = get_storage()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )
    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = auto_sync.list_sessions(max_sessions=max_sessions)
    sessions = results.get("sessions", [])

    prepared: list[dict[str, object]] = []
    for row in sessions:
        title = str(row.get("title") or "").strip() or "Untitled conversation"
        prepared.append(
            {
                "source": str(row.get("source") or ""),
                "session_id": str(row.get("session_id") or ""),
                "title": title,
                "started": _format_session_time(row.get("started_at")),
                "message_count": int(row.get("message_count", 0)),
                "processed": bool(row.get("processed")),
            }
        )
    return prepared


def _get_theme_defaults_for_tui() -> tuple[list[str], str]:
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            config_dict = get_files().read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass
    return ThemeManager.get_available_themes(), current_theme


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
    inject_stored_api_keys()
    files = get_files()
    interactive_shell = is_interactive_terminal()

    if iterations is not None and iterations < 1:
        console.print("[error]--iterations must be >= 1[/error]")
        raise typer.Exit(1)

    onboarding_complete = is_repo_onboarding_complete(files)
    onboarding_required = False
    if onboarding:
        onboarding_required = force_onboarding or not onboarding_complete
    elif not onboarding_complete:
        console.print(
            "[warning]Onboarding is incomplete for this repository. "
            "Run 'agent-recall open' to complete setup in the TUI.[/warning]"
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
            execute_command=lambda raw, width, height: _execute_tui_slash_command(
                _normalize_tui_command(raw),
                terminal_width=width,
                terminal_height=max(height * 2, 200),
            ),
            list_sessions_for_picker=(
                lambda max_items, include_all_cursor: _list_sessions_for_tui_picker(
                    max_items,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            run_setup_payload=_run_setup_from_payload,
            run_model_config=_run_model_config_for_tui,
            theme_defaults_provider=_get_theme_defaults_for_tui,
            theme_runtime_provider=lambda: (
                _theme_manager.get_theme_name(),
                _theme_manager.get_theme(),
            ),
            theme_resolve_provider=lambda theme_name: (
                Theme(
                    ThemeManager.get_theme_colors(theme_name),
                    inherit=True,
                )
                if ThemeManager.is_valid_theme(theme_name)
                else None
            ),
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
            onboarding_required=onboarding_required,
        )
        app_instance.run()
    except KeyboardInterrupt:
        console.print("\n[dim]TUI closed.[/dim]")


@app.command("open")
def open_dashboard(
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
    force_onboarding: bool = typer.Option(
        False,
        "--force-onboarding",
        help="Force setup flow even if this repository is already configured.",
    ),
):
    """Open the TUI dashboard (recommended entrypoint)."""
    tui(
        refresh_seconds=refresh_seconds,
        all_cursor_workspaces=all_cursor_workspaces,
        iterations=iterations,
        no_splash=no_splash,
        splash_delay=splash_delay,
        onboarding=True,
        force_onboarding=force_onboarding,
    )


@app.command("reset-sync")
def reset_sync(
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help=f"Clear only one source: {SOURCE_CHOICES_TEXT}",
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
        normalized = normalize_source_name(source)
        if normalized not in VALID_SOURCE_NAMES:
            console.print(f"[error]Invalid source. Use one of: {SOURCE_CHOICES_TEXT}[/error]")
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
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        min=1,
        help="Maximum results (default from config)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override retrieval backend (fts5 or hybrid)",
    ),
    fusion_k: int | None = typer.Option(
        None,
        "--fusion-k",
        min=1,
        help="Override hybrid fusion constant (default from config)",
    ),
    rerank: bool | None = typer.Option(
        None,
        "--rerank/--no-rerank",
        help="Override reranking behavior (default from config)",
    ),
    rerank_candidate_k: int | None = typer.Option(
        None,
        "--rerank-candidate-k",
        min=1,
        help="Override rerank candidate pool size (default from config)",
    ),
):
    """Retrieve relevant memory chunks."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    files = get_files()
    try:
        retriever, retrieval_cfg = _build_retriever(
            storage,
            files,
            backend=backend,
            fusion_k=fusion_k,
            rerank=rerank,
            rerank_candidate_k=rerank_candidate_k,
        )
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1) from None

    _get_theme_manager()  # Ensure theme is loaded
    effective_top_k = top_k if top_k is not None else retrieval_cfg.top_k
    chunks = retriever.search(query=query, top_k=effective_top_k)
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
