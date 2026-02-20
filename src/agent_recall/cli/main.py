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
from uuid import UUID

import httpx
import typer
from packaging.version import Version
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme
from typer.main import get_command as get_typer_command
from typer.testing import CliRunner

from agent_recall import __version__
from agent_recall.cli import ralph as ralph_cli
from agent_recall.cli.banner import print_banner
from agent_recall.cli.command_contract import command_parity_report, get_command_contract
from agent_recall.cli.ralph import (
    load_prd_items,
    ralph_app,
    read_ralph_config,
    render_ralph_status,
)
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.cli.tui import get_palette_actions
from agent_recall.cli.tui.commands.help_text import build_tui_help_lines
from agent_recall.cli.tui.views import DashboardRenderContext, build_tui_dashboard
from agent_recall.core.adapters import get_default_adapters, write_adapter_payloads
from agent_recall.core.background_sync import BackgroundSyncManager
from agent_recall.core.compact import CompactionEngine
from agent_recall.core.config import load_config
from agent_recall.core.context import ContextAssembler
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.onboarding import (
    apply_repo_setup,
    discover_coding_cli_models,
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
from agent_recall.core.tier_compaction import (
    TierCompactionConfig,
    TierCompactionHook,
)
from agent_recall.core.tier_writer import (
    TierValidationError,
    TierWriter,
    WriteMode,
    WritePolicy,
    get_tier_statistics,
    lint_tier_file,
)
from agent_recall.ingest import get_default_ingesters, get_ingester
from agent_recall.ingest.sources import (
    VALID_SOURCE_NAMES,
    normalize_source_name,
    resolve_source_location_hint,
)
from agent_recall.llm import (
    LLMProvider,
    Message,
    create_llm_provider,
    ensure_provider_dependency,
    get_available_providers,
    validate_provider_config,
)
from agent_recall.llm.coding_cli import CodingCLIProvider
from agent_recall.ralph.costs import format_usd, summarize_costs
from agent_recall.ralph.iteration_store import IterationOutcome, IterationReportStore
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import CurationStatus, LLMConfig, RetrievalConfig, SemanticLabel
from agent_recall.storage.remote import resolve_shared_db_path

app = typer.Typer(help="Agent Memory System - Persistent knowledge for AI coding agents")
_slash_runner = CliRunner()


def _print_version_and_maybe_update() -> None:
    """Print installed version and update hint if available."""
    current = __version__
    console.print(f"agent-recall {current}")

    latest = _fetch_latest_version()
    if latest is None:
        return

    try:
        if Version(latest) > Version(current):
            console.print(
                f"[dim]A newer version ({latest}) is available. "
                "Upgrade with: uv tool upgrade agent-recall[/dim]"
            )
    except Exception:  # noqa: BLE001
        pass


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    if version_flag:
        _print_version_and_maybe_update()
        raise typer.Exit()


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

INITIAL_RULES = """# Rules

User-authored operating rules for coding agents in this repository.
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

tier_compaction:
  auto_run: true
  max_entries_per_tier: 50
  strict_deduplication: false
  summary_threshold_entries: 40
  summary_max_entries: 20

retrieval:
  backend: fts5
  top_k: 5
  fusion_k: 60
  rerank_enabled: false
  rerank_candidate_k: 20
  embedding_enabled: false
  embedding_dimensions: 64

storage:
  backend: local
  shared:
    base_url: null
    api_key_env: AGENT_RECALL_SHARED_API_KEY
    require_api_key: false
    timeout_seconds: 10.0
    retry_attempts: 2

theme:
  name: dark+

ralph:
  enabled: false
  max_iterations: 10
  sleep_seconds: 2
  compact_mode: always
  forecast:
    window: 5
    use_llm: false
    llm_on_consecutive_failures: 2
    llm_model: null
  synthesis:
    auto_after_loop: true
    max_guardrails: 30
    max_style: 30

adapters:
  enabled: false
  output_dir: .agent/context
  token_budget: null
  per_adapter_token_budget: {}
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
    config = load_config(AGENT_DIR)
    shared_tiers_dir: Path | None = None
    if config.storage.backend == "shared":
        try:
            resolved = resolve_shared_db_path(config.storage.shared.base_url)
            if isinstance(resolved, Path):
                shared_tiers_dir = resolved.parent
        except (NotImplementedError, ValueError):
            shared_tiers_dir = None
    return FileStorage(AGENT_DIR, shared_tiers_dir=shared_tiers_dir)


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
    return build_tui_help_lines()


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

    # Include CLI-only contract commands not discovered by Typer walk
    # (e.g. "ralph set-prds" where action is an argument, not a subcommand)
    for contract in get_command_contract():
        if "cli" not in contract.surfaces:
            continue
        for cmd in [contract.command, *contract.aliases]:
            if cmd and cmd not in seen:
                seen.add(cmd)
                commands.append(cmd)

    return commands


def get_default_prd_path() -> Path:
    return ralph_cli.get_default_prd_path()


def get_default_script_path() -> Path:
    return ralph_cli.get_default_script_path()


@app.command("command-inventory")
def command_inventory() -> None:
    """Print command inventory for CLI, TUI, and palette."""
    cli_commands = set(_collect_cli_commands_for_palette())
    tui_slash_commands: set[str] = set()
    for contract in get_command_contract():
        if "tui" not in contract.surfaces:
            continue
        tui_slash_commands.add(contract.command)
        tui_slash_commands.update(contract.aliases)
    palette_actions = {action.action_id for action in get_palette_actions()}
    parity = command_parity_report(
        cli_commands=cli_commands,
        tui_commands=tui_slash_commands,
    )

    table = Table(title="Command Inventory", box=box.SIMPLE)
    table.add_column("Surface")
    table.add_column("Count", justify="right")
    table.add_row("CLI", str(len(cli_commands)))
    table.add_row("TUI Slash", str(len(tui_slash_commands)))
    table.add_row("Palette", str(len(palette_actions)))
    console.print(table)

    contract_table = Table(title="Contract Commands", box=box.SIMPLE)
    contract_table.add_column("Command")
    contract_table.add_column("Aliases")
    contract_table.add_column("Surfaces")
    for contract in get_command_contract():
        aliases = ", ".join(contract.aliases) if contract.aliases else "-"
        surfaces = ", ".join(contract.surfaces)
        contract_table.add_row(contract.command, aliases, surfaces)
    console.print(contract_table)

    if (
        parity["missing_in_tui"]
        or parity["missing_in_cli"]
        or parity["extra_in_tui"]
        or parity["extra_in_cli"]
    ):
        parity_table = Table(title="Parity Gaps", box=box.SIMPLE)
        parity_table.add_column("Category")
        parity_table.add_column("Commands")
        parity_table.add_row(
            "Missing in TUI",
            ", ".join(sorted(parity["missing_in_tui"])) or "-",
        )
        parity_table.add_row(
            "Missing in CLI",
            ", ".join(sorted(parity["missing_in_cli"])) or "-",
        )
        parity_table.add_row(
            "Extra in TUI",
            ", ".join(sorted(parity["extra_in_tui"])) or "-",
        )
        parity_table.add_row(
            "Extra in CLI",
            ", ".join(sorted(parity["extra_in_cli"])) or "-",
        )
        console.print(parity_table)

    palette_table = Table(title="Palette Actions", box=box.SIMPLE)
    palette_table.add_column("Action")
    for action_id in sorted(palette_actions):
        palette_table.add_row(action_id)
    console.print(palette_table)


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


def _read_adapter_config(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    adapter_config = config_dict.get("adapters", {}) if isinstance(config_dict, dict) else {}
    return adapter_config if isinstance(adapter_config, dict) else {}


def _write_adapter_config(files: FileStorage, updates: dict[str, object]) -> dict[str, object]:
    config_dict = files.read_config()
    if "adapters" not in config_dict or not isinstance(config_dict.get("adapters"), dict):
        config_dict["adapters"] = {}
    adapter_config = config_dict["adapters"]
    if isinstance(adapter_config, dict):
        adapter_config.update(updates)
    config_dict["adapters"] = adapter_config
    files.write_config(config_dict)
    return config_dict


def _format_adapter_budgets(adapter_config: dict[str, object]) -> str:
    budgets = adapter_config.get("per_adapter_token_budget")
    if not isinstance(budgets, dict):
        return "none"
    normalized: list[str] = []
    for name, budget in budgets.items():
        if not isinstance(name, str):
            continue
        if not isinstance(budget, int):
            continue
        normalized.append(f"{name}={budget}")
    return ", ".join(normalized) if normalized else "none"


def _format_adapter_token_budget(adapter_config: dict[str, object]) -> str:
    value = adapter_config.get("token_budget")
    return str(value) if value is not None else "none"


def _format_named_token_budgets(values: object) -> str:
    if not isinstance(values, dict):
        return "none"
    normalized: list[str] = []
    for name, budget in values.items():
        if not isinstance(name, str):
            continue
        if not isinstance(budget, int):
            continue
        normalized.append(f"{name}={budget}")
    return ", ".join(normalized) if normalized else "none"


def _parse_named_token_budgets(raw_value: str, *, label: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    if not raw_value.strip():
        return parsed
    for raw in raw_value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            console.print(f"[error]Invalid {label} budget '{item}'. Use name=tokens.[/error]")
            raise typer.Exit(1)
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            console.print(f"[error]{label.title()} name cannot be empty.[/error]")
            raise typer.Exit(1)
        try:
            tokens = int(value)
        except ValueError:
            console.print(f"[error]Invalid token budget '{value}'. Use an integer.[/error]")
            raise typer.Exit(1)
        if tokens <= 0:
            console.print(f"[error]Token budget must be > 0 (got {tokens}).[/error]")
            raise typer.Exit(1)
        parsed[name] = tokens
    return parsed


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
    get_storage.cache_clear()
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

    if (
        command_name == "ralph"
        and len(parts) >= 2
        and parts[1].lower()
        in {
            "enable",
            "disable",
            "status",
            "select",
            "set-prds",
            "get-selected-prds",
        }
    ):
        command = ["ralph", *parts[1:]]
        result = _slash_runner.invoke(app, command)
        lines = []
        if result.exit_code == 0:
            lines.append(f"[success]✓ /{escape(' '.join(parts))}[/success]")
        else:
            lines.append(f"[error]✗ /{escape(' '.join(parts))} (exit {result.exit_code})[/error]")
        output = result.output.strip()
        if output:
            for line in output.splitlines():
                if line.strip():
                    lines.append(f"[dim]{escape(line.strip())}[/dim]")
        return False, lines

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

    valid_views = {
        "overview",
        "sources",
        "llm",
        "knowledge",
        "settings",
        "timeline",
        "ralph",
        "console",
        "all",
    }
    if len(parts) == 1:
        return (
            True,
            current_view,
            [
                f"[dim]Current view: {current_view}[/dim]",
                "[dim]Available views: overview, "
                "knowledge, settings, timeline, ralph, console, all[/dim]",
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

    (AGENT_DIR / "RULES.md").write_text(INITIAL_RULES)
    (AGENT_DIR / "GUARDRAILS.md").write_text(INITIAL_GUARDRAILS)
    (AGENT_DIR / "STYLE.md").write_text(INITIAL_STYLE)
    (AGENT_DIR / "RECENT.md").write_text(INITIAL_RECENT)
    (AGENT_DIR / "config.yaml").write_text(INITIAL_CONFIG)

    # Ensure storage points at the newly initialized repository path.
    get_storage.cache_clear()
    get_storage()

    console.print(
        Panel.fit(
            "[success]✓ Initialized .agent/ directory[/success]\n\n"
            "Files created:\n"
            "  • config.yaml - Configuration\n"
            "  • RULES.md - User-authored agent rules\n"
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
    adapter_payloads: bool | None = typer.Option(
        None,
        "--adapter-payloads/--no-adapter-payloads",
        help="Write adapter-ready context payloads for supported agents",
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
    config = load_config(AGENT_DIR)
    adapter_cfg = config.adapters
    llm_cfg = config.llm
    if adapter_payloads is None:
        adapter_payloads = bool(adapter_cfg.enabled)
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
            refreshed_at = datetime.now(UTC)
            repo_path = Path.cwd().resolve()
            payload = {
                "task": resolved_task,
                "active_session_id": str(active.id) if active else None,
                "repo_path": str(repo_path),
                "refreshed_at": refreshed_at.isoformat(),
                "context": output,
            }
            json_path.write_text(json.dumps(payload, indent=2))
            if adapter_payloads:
                write_adapter_payloads(
                    context=output,
                    task=resolved_task,
                    active_session_id=str(active.id) if active else None,
                    repo_path=repo_path,
                    refreshed_at=refreshed_at,
                    output_dir=Path(adapter_cfg.output_dir),
                    adapters=get_default_adapters(),
                    token_budget=adapter_cfg.token_budget,
                    per_adapter_budgets=adapter_cfg.per_adapter_token_budget,
                    per_provider_budgets=adapter_cfg.per_provider_token_budget,
                    per_model_budgets=adapter_cfg.per_model_token_budget,
                    provider=llm_cfg.provider,
                    model=llm_cfg.model,
                )
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
    if adapter_payloads:
        adapter_names = ", ".join(adapter.name for adapter in get_default_adapters())
        adapter_dir = Path(adapter_cfg.output_dir)
        lines.append(f"  Adapters: {adapter_names}")
        lines.append(f"  Adapter dir: {adapter_dir}")
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

    _get_theme_manager()

    config_dict = files.read_config()
    compaction_cfg = config_dict.get("compaction", {}) if isinstance(config_dict, dict) else {}
    backend = compaction_cfg.get("backend", "llm") if isinstance(compaction_cfg, dict) else "llm"

    llm: LLMProvider
    if backend == "coding_cli":
        ralph_cfg = config_dict.get("ralph", {}) if isinstance(config_dict, dict) else {}
        coding_cli = ralph_cfg.get("coding_cli") if isinstance(ralph_cfg, dict) else None
        cli_model = ralph_cfg.get("cli_model") if isinstance(ralph_cfg, dict) else None

        if not coding_cli:
            console.print(
                "[error]compaction.backend is 'coding_cli' "
                "but ralph.coding_cli is not configured[/error]"
            )
            raise typer.Exit(1)

        try:
            llm = CodingCLIProvider(coding_cli=str(coding_cli), model=cli_model)
            console.print(f"[dim]Using coding CLI backend: {coding_cli}[/dim]")
        except Exception as exc:
            console.print(f"[error]Coding CLI provider error: {exc}[/error]")
            raise typer.Exit(1) from None
    else:
        try:
            llm = get_llm()
        except Exception as exc:
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
            f"Backend: {backend}\n"
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
    adapter_config = _read_adapter_config(files)

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
        "[bold]Context Adapters:[/bold]",
        f"  Enabled: {'yes' if bool(adapter_config.get('enabled')) else 'no'}",
        f"  Output dir: {adapter_config.get('output_dir') or '.agent/context'}",
        f"  Token budget: {_format_adapter_token_budget(adapter_config)}",
        f"  Per-adapter budgets: {_format_adapter_budgets(adapter_config)}",
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

    config_dict = files.read_config()
    lines.extend(render_ralph_status(config_dict))

    console.print(Panel.fit("\n".join(lines), title="Agent Recall Status"))


def _fetch_latest_version() -> str | None:
    """Fetch the latest version from PyPI. Returns None on any error."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("https://pypi.org/pypi/agent-recall/json")
            resp.raise_for_status()
            data = resp.json()
            return data.get("info", {}).get("version")
    except Exception:  # noqa: BLE001
        return None


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "-"
    if duration_seconds < 60:
        return f"{duration_seconds:.0f}s"
    minutes = duration_seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _render_iteration_timeline(store: IterationReportStore, *, max_entries: int) -> list[str]:
    if max_entries <= 0:
        return ["[dim]No entries to display.[/dim]"]
    reports = store.load_recent(count=max_entries)
    if not reports:
        return ["[dim]No iteration history yet.[/dim]"]

    lines: list[str] = []
    for report in reports:
        outcome = report.outcome
        symbol = "✓" if outcome == IterationOutcome.COMPLETED else "✗"
        outcome_text = outcome.value if outcome is not None else "UNKNOWN"
        duration = _format_duration(report.duration_seconds)
        summary = report.summary or report.failure_reason or report.validation_hint or ""
        summary = _truncate(summary, 64)
        item_label = report.item_id or "unknown"
        headline = f"{report.iteration:03d} {symbol} {item_label} ({duration}) {outcome_text}"
        lines.append(headline)
        if summary:
            lines.append(f"  {summary}")
    return lines


def _load_ralph_state_payload() -> dict:
    """Load ralph_state.json and return parsed payload or empty dict."""
    try:
        state_path = AGENT_DIR / "ralph" / "ralph_state.json"
        if not state_path.exists():
            return {}
        import json

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _dashboard_render_context() -> DashboardRenderContext:
    ralph_max_iterations: int | None = None
    try:
        ralph_config = read_ralph_config(get_files())
        max_iter_value = ralph_config.get("max_iterations")
        if isinstance(max_iter_value, int | float) and max_iter_value >= 1:
            ralph_max_iterations = int(max_iter_value)
    except Exception:  # noqa: BLE001
        ralph_max_iterations = None

    # Load Ralph state for context-aware dashboard
    ralph_state = _load_ralph_state_payload()
    ralph_enabled = ralph_state.get("enabled", False) is True
    ralph_running = ralph_state.get("status", "").lower() == "running"

    return DashboardRenderContext(
        console=console,
        theme_manager=_theme_manager,
        agent_dir=AGENT_DIR,
        ralph_max_iterations=ralph_max_iterations,
        get_storage=get_storage,
        get_files=get_files,
        get_repo_selected_sources=_get_repo_selected_sources,
        resolve_repo_root_for_display=_resolve_repo_root_for_display,
        filter_ingesters_by_sources=_filter_ingesters_by_sources,
        get_default_ingesters=get_default_ingesters,
        render_iteration_timeline=_render_iteration_timeline,
        summarize_costs=summarize_costs,
        format_usd=format_usd,
        is_interactive_terminal=is_interactive_terminal,
        help_lines_provider=_tui_help_lines,
        ralph_enabled=ralph_enabled,
        ralph_running=ralph_running,
    )


def _build_tui_dashboard(
    all_cursor_workspaces: bool = False,
    include_banner_header: bool = True,
    slash_status: str | None = None,
    slash_output: list[str] | None = None,
    view: str = "overview",
    ralph_agent_transport: str = "pipe",
    show_slash_console: bool = True,
) -> Group:
    """Build the live TUI dashboard renderable."""
    try:
        tui_config = _read_tui_config(get_files())
    except Exception:  # noqa: BLE001
        tui_config = {}
    banner_size = "normal"
    if isinstance(tui_config, dict):
        raw_banner = tui_config.get("banner_size")
        if isinstance(raw_banner, str):
            banner_size = raw_banner.strip().lower()
    return build_tui_dashboard(
        _dashboard_render_context(),
        all_cursor_workspaces=all_cursor_workspaces,
        include_banner_header=include_banner_header,
        banner_size=banner_size,
        slash_status=slash_status,
        slash_output=slash_output,
        view=view,
        ralph_agent_transport=ralph_agent_transport,
        show_slash_console=show_slash_console,
        widget_visibility=tui_config.get("widget_visibility")
        if isinstance(tui_config, dict)
        else None,
    )


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

    configure_coding_agent = bool(payload.get("configure_coding_agent", False))
    coding_cli_value = payload.get("coding_cli")
    coding_cli = (
        str(coding_cli_value).strip()
        if isinstance(coding_cli_value, str) and str(coding_cli_value).strip()
        else None
    )
    cli_model_value = payload.get("cli_model")
    cli_model = (
        str(cli_model_value).strip()
        if isinstance(cli_model_value, str) and str(cli_model_value).strip()
        else None
    )
    ralph_enabled_raw = payload.get("ralph_enabled")
    ralph_enabled = bool(ralph_enabled_raw) if isinstance(ralph_enabled_raw, bool) else None
    if configure_coding_agent and ralph_enabled is None:
        ralph_enabled = bool(coding_cli)

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
        configure_coding_agent=configure_coding_agent,
        coding_cli=coding_cli,
        cli_model=cli_model,
        ralph_enabled=ralph_enabled,
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


def _list_prd_items_for_tui_picker() -> dict[str, Any]:
    """Load PRD items and Ralph config for the TUI PRD selection modal."""
    prd_path = get_default_prd_path()
    items = load_prd_items(prd_path)
    files = get_files()
    ralph_config = read_ralph_config(files)
    selected_value = ralph_config.get("selected_prd_ids")
    selected_ids: list[str] = []
    if isinstance(selected_value, list):
        selected_ids = [str(x) for x in selected_value if x]
    max_iterations = int(ralph_config.get("max_iterations") or 10)
    prepared: list[dict[str, Any]] = []
    for it in items:
        item_id = str(it.get("id") or "")
        if not item_id:
            continue
        prepared.append(
            {
                "id": item_id,
                "title": str(it.get("title") or "Untitled"),
                "priority": int(it.get("priority") or 0),
                "passes": bool(it.get("passes")),
            }
        )
    return {
        "items": prepared,
        "selected_ids": selected_ids,
        "max_iterations": max_iterations,
    }


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


def _get_sources_and_sessions_for_tui(
    max_sessions: int = 200,
    *,
    all_cursor_workspaces: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ensure_initialized()
    storage = get_storage()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )

    # Gather sources status
    sources_data: list[dict[str, object]] = []
    for ingester in ingesters:
        try:
            sessions = ingester.discover_sessions()
            available = len(sessions) > 0
            if available:
                location = str(sessions[0].resolve())
            else:
                hint = resolve_source_location_hint(ingester)
                location = str(hint) if hint else "Unknown"
            sources_data.append(
                {
                    "name": ingester.source_name,
                    "available": available,
                    "location": location,
                    "count": len(sessions),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            sources_data.append(
                {
                    "name": ingester.source_name,
                    "available": False,
                    "location": None,
                    "count": 0,
                    "error": str(exc),
                }
            )

    auto_sync = AutoSync(storage, files, llm=None, ingesters=ingesters)
    results = auto_sync.list_sessions(max_sessions=max_sessions)
    sessions = results.get("sessions", [])

    sessions_data: list[dict[str, object]] = []
    for row in sessions:
        title = str(row.get("title") or "").strip() or "Untitled conversation"
        sessions_data.append(
            {
                "source": str(row.get("source") or ""),
                "session_id": str(row.get("session_id") or ""),
                "title": title,
                "started": _format_session_time(row.get("started_at")),
                "message_count": int(row.get("message_count", 0)),
                "processed": bool(row.get("processed")),
            }
        )
    return sources_data, sessions_data


def _get_session_detail_for_tui(
    source: str,
    session_id: str,
    all_cursor_workspaces: bool = False,
) -> Any | None:
    ensure_initialized()
    files = get_files()
    selected_sources = _get_repo_selected_sources(files)
    ingesters = _filter_ingesters_by_sources(
        get_default_ingesters(cursor_all_workspaces=all_cursor_workspaces),
        selected_sources,
    )
    for ingester in ingesters:
        if str(ingester.source_name) == str(source):
            try:
                paths = ingester.discover_sessions()
                for path in paths:
                    try:
                        extracted = ingester.get_session_id(path)
                        if str(extracted) == str(session_id):
                            return ingester.parse_session(path)
                    except Exception:
                        pass
            except Exception:
                pass
    return None


def _get_theme_defaults_for_tui() -> tuple[list[str], str]:
    current_theme = DEFAULT_THEME
    if AGENT_DIR.exists():
        try:
            config_dict = get_files().read_config()
            current_theme = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        except Exception:  # noqa: BLE001
            pass
    return ThemeManager.get_available_themes(), current_theme


def _read_tui_config(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    tui_config = config_dict.get("tui", {}) if isinstance(config_dict, dict) else {}
    return tui_config if isinstance(tui_config, dict) else {}


def _write_tui_config(files: FileStorage, updates: dict[str, object]) -> dict[str, object]:
    config_dict = files.read_config()
    if "tui" not in config_dict or not isinstance(config_dict.get("tui"), dict):
        config_dict["tui"] = {}
    tui_config = config_dict["tui"]
    if isinstance(tui_config, dict):
        tui_config.update(updates)
    config_dict["tui"] = tui_config
    files.write_config(config_dict)
    return config_dict


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
    all_cursor_workspaces: bool = typer.Option(
        False,
        "--all-cursor-workspaces",
        help="Include Cursor sessions from all workspaces.",
    ),
    show_terminal: bool = typer.Option(
        False,
        "--show-terminal",
        help="Show the embedded terminal panel when supported.",
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
    no_delta_setup: bool = typer.Option(
        False,
        "--no-delta-setup",
        help="Skip first-launch delta diff renderer setup prompt.",
    ),
    force_delta_setup: bool = typer.Option(
        False,
        "--force-delta-setup",
        help="Reset delta setup: clear cached binary and show the download prompt again.",
    ),
):
    """Start a live terminal UI dashboard for agent-recall."""
    _get_theme_manager()
    ensure_initialized()
    if force_delta_setup:
        from agent_recall.cli.tui.delta import reset_delta_setup

        reset_delta_setup()
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
                    show_slash_console=False,
                )
            )
            if index < iterations - 1:
                time.sleep(0.5)
        return

    if not interactive_shell:
        console.print(
            "[error]The Textual TUI requires an interactive terminal. "
            "Use a terminal session or run with --iterations for non-interactive checks.[/error]"
        )
        raise typer.Exit(1)

    try:
        from agent_recall.cli.tui import AgentRecallTextualApp
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Textual TUI unavailable: {exc}[/error]")
        raise typer.Exit(1) from None

    try:
        tui_config = _read_tui_config(files)
        if show_terminal:
            _write_tui_config(files, {"terminal_panel_visible": True})
            tui_config = _read_tui_config(files)

        app_instance = AgentRecallTextualApp(
            render_dashboard=_build_tui_dashboard,
            dashboard_context=_dashboard_render_context(),
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
            get_sources_and_sessions_for_tui=(
                lambda max_items, include_all_cursor: _get_sources_and_sessions_for_tui(
                    max_items,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            get_session_detail_for_tui=(
                lambda source, session_id, include_all_cursor: _get_session_detail_for_tui(
                    source,
                    session_id,
                    all_cursor_workspaces=include_all_cursor,
                )
            ),
            list_prd_items_for_picker=_list_prd_items_for_tui_picker,
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
            discover_coding_models=lambda cli: discover_coding_cli_models(cli, timeout_seconds=8.0),
            providers=get_available_providers(),
            cli_commands=_collect_cli_commands_for_palette(),
            rich_theme=_theme_manager.get_theme(),
            initial_view=str(tui_config.get("default_view", "overview")),
            all_cursor_workspaces=bool(
                tui_config.get("all_cursor_workspaces", all_cursor_workspaces)
            ),
            ralph_agent_transport=str(tui_config.get("ralph_agent_transport", "pipe")),
            onboarding_required=onboarding_required,
            terminal_panel_visible=bool(tui_config.get("terminal_panel_visible", False)),
            terminal_supported=show_terminal,
            no_delta_setup=no_delta_setup,
        )
        app_instance.run()
    except KeyboardInterrupt:
        console.print("\n[dim]TUI closed.[/dim]")


@app.command("open")
def open_dashboard(
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
    show_terminal: bool = typer.Option(
        False,
        "--show-terminal",
        help="Show the embedded terminal panel when supported.",
    ),
    force_onboarding: bool = typer.Option(
        False,
        "--force-onboarding",
        help="Force setup flow even if this repository is already configured.",
    ),
):
    """Open the TUI dashboard (recommended entrypoint)."""
    tui(
        all_cursor_workspaces=all_cursor_workspaces,
        iterations=iterations,
        no_splash=no_splash,
        splash_delay=splash_delay,
        show_terminal=show_terminal,
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


@config_app.command("adapters")
def config_adapters(
    enabled: bool | None = typer.Option(
        None,
        "--enabled/--disabled",
        help="Enable or disable automatic context adapter payloads",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory where adapter payloads are written",
    ),
    token_budget: int | None = typer.Option(
        None,
        "--token-budget",
        min=1,
        help="Global token budget for adapter payload context",
    ),
    per_adapter_token_budget: str | None = typer.Option(
        None,
        "--per-adapter-token-budget",
        help="Per-adapter token budgets as name=tokens pairs",
    ),
    per_provider_token_budget: str | None = typer.Option(
        None,
        "--per-provider-token-budget",
        help="Per-provider token budgets as provider=tokens pairs",
    ),
    per_model_token_budget: str | None = typer.Option(
        None,
        "--per-model-token-budget",
        help="Per-model token budgets as model=tokens pairs",
    ),
):
    """Configure automatic context adapter payloads."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    updates: dict[str, object] = {}
    if enabled is not None:
        updates["enabled"] = enabled
    if output_dir is not None:
        updates["output_dir"] = str(output_dir)
    if token_budget is not None:
        updates["token_budget"] = token_budget
    if per_adapter_token_budget is not None:
        updates["per_adapter_token_budget"] = _parse_named_token_budgets(
            per_adapter_token_budget,
            label="per-adapter",
        )
    if per_provider_token_budget is not None:
        updates["per_provider_token_budget"] = _parse_named_token_budgets(
            per_provider_token_budget,
            label="per-provider",
        )
    if per_model_token_budget is not None:
        updates["per_model_token_budget"] = _parse_named_token_budgets(
            per_model_token_budget,
            label="per-model",
        )

    if not updates:
        adapter_config = _read_adapter_config(files)
        lines = [
            "[bold]Context Adapters:[/bold]",
            f"  Enabled: {'yes' if bool(adapter_config.get('enabled')) else 'no'}",
            f"  Output dir: {adapter_config.get('output_dir') or '.agent/context'}",
            f"  Token budget: {_format_adapter_token_budget(adapter_config)}",
            f"  Per-adapter budgets: {_format_adapter_budgets(adapter_config)}",
            "  Per-provider budgets: "
            f"{_format_named_token_budgets(adapter_config.get('per_provider_token_budget'))}",
            "  Per-model budgets: "
            f"{_format_named_token_budgets(adapter_config.get('per_model_token_budget'))}",
        ]
        console.print(Panel.fit("\n".join(lines), title="Context Adapters"))
        return

    _write_adapter_config(files, updates)
    adapter_config = _read_adapter_config(files)
    lines = [
        "[success]✓ Adapter settings updated[/success]",
        f"  Enabled: {'yes' if bool(adapter_config.get('enabled')) else 'no'}",
        f"  Output dir: {adapter_config.get('output_dir') or '.agent/context'}",
        f"  Token budget: {_format_adapter_token_budget(adapter_config)}",
        f"  Per-adapter budgets: {_format_adapter_budgets(adapter_config)}",
        "  Per-provider budgets: "
        f"{_format_named_token_budgets(adapter_config.get('per_provider_token_budget'))}",
        "  Per-model budgets: "
        f"{_format_named_token_budgets(adapter_config.get('per_model_token_budget'))}",
    ]
    console.print(Panel.fit("\n".join(lines), title="Context Adapters"))


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
curation_app = typer.Typer(help="Review and approve extracted learnings")


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
app.add_typer(curation_app, name="curation")
app.add_typer(ralph_app, name="ralph")


@curation_app.command("list")
def curation_list(
    status: str = typer.Option("pending", "--status", "-s", help="pending, approved, rejected"),
    limit: int = typer.Option(50, "--limit", "-l", min=1, help="Maximum entries"),
):
    """List log entries by curation status."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        status_enum = CurationStatus(status.strip().lower())
    except ValueError:
        valid = ", ".join(item.value for item in CurationStatus)
        console.print(f"[error]Invalid status '{status}'. Valid: {valid}[/error]")
        raise typer.Exit(1) from None

    entries = storage.list_entries_by_curation_status(status_enum, limit=limit)
    if not entries:
        console.print(f"[warning]No entries found for status '{status_enum.value}'.[/warning]")
        raise typer.Exit(0)

    table = Table(title=f"Curation Queue ({status_enum.value})", box=box.SIMPLE)
    table.add_column("ID", style="dim")
    table.add_column("Label")
    table.add_column("Confidence", justify="right")
    table.add_column("Status")
    table.add_column("Content")
    for entry in entries:
        preview = textwrap.shorten(entry.content, width=80, placeholder="...")
        table.add_row(
            str(entry.id),
            entry.label.value,
            f"{entry.confidence:.2f}",
            entry.curation_status.value,
            preview,
        )
    console.print(table)


@curation_app.command("approve")
def curation_approve(
    entry_id: str = typer.Argument(..., help="Entry UUID to approve"),
):
    """Approve a pending log entry."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        entry_uuid = UUID(entry_id)
    except ValueError:
        console.print(f"[error]Invalid entry id '{entry_id}'. Expected UUID.[/error]")
        raise typer.Exit(1) from None

    updated = storage.update_entry_curation_status(entry_uuid, CurationStatus.APPROVED)
    if updated is None:
        console.print(f"[warning]Entry not found: {entry_id}[/warning]")
        raise typer.Exit(1)
    console.print(f"[success]✓ Approved entry {entry_id}[/success]")


@curation_app.command("reject")
def curation_reject(
    entry_id: str = typer.Argument(..., help="Entry UUID to reject"),
):
    """Reject a pending log entry."""
    _get_theme_manager()  # Ensure theme is loaded
    storage = get_storage()
    try:
        entry_uuid = UUID(entry_id)
    except ValueError:
        console.print(f"[error]Invalid entry id '{entry_id}'. Expected UUID.[/error]")
        raise typer.Exit(1) from None

    updated = storage.update_entry_curation_status(entry_uuid, CurationStatus.REJECTED)
    if updated is None:
        console.print(f"[warning]Entry not found: {entry_id}[/warning]")
        raise typer.Exit(1)
    console.print(f"[success]✓ Rejected entry {entry_id}[/success]")


@app.command("write-guardrails")
def write_guardrails(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    reason: str = typer.Option("progressed", "--reason", "-r", help="Reason for entry"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
    mode: str = typer.Option("append", "--mode", "-m", help="Write mode: append, replace-section"),
    skip_duplicate: bool = typer.Option(
        True, "--skip-duplicate/--no-skip-duplicate", help="Skip if duplicate"
    ),
):
    """Write a structured entry to GUARDRAILS.md."""
    ensure_initialized()
    files = get_files()

    policy = WritePolicy(
        mode=WriteMode(mode),
        deduplicate=skip_duplicate,
    )
    writer = TierWriter(files, policy)

    try:
        written = writer.write_guardrails_entry(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            reason=reason,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote guardrails entry for iteration {iteration}[/success]")
        else:
            console.print(f"[info]Skipped duplicate entry for iteration {iteration}[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@app.command("write-guardrails-failure")
def write_guardrails_failure(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    validation_errors: list[str] = typer.Option([], "--error", help="Validation error messages"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Primary validation signal"),
):
    """Write a hard failure entry to GUARDRAILS.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_guardrails_hard_failure(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            validation_errors=validation_errors,
            validation_hint=validation_hint,
        )
        if written:
            console.print(
                f"[success]Wrote hard failure guardrails entry for iteration {iteration}[/success]"
            )
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@app.command("write-style")
def write_style(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
):
    """Write a structured entry to STYLE.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_style_entry(
            iteration=iteration,
            item_id=item_id,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote style entry for iteration {iteration}[/success]")
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@app.command("write-recent")
def write_recent(
    iteration: int = typer.Option(..., "--iteration", "-i", help="Iteration number"),
    item_id: str = typer.Option(..., "--item-id", help="PRD item ID"),
    item_title: str = typer.Option(..., "--item-title", help="PRD item title"),
    work_mode: str = typer.Option("feature", "--mode", "-m", help="Work mode"),
    agent_exit: int = typer.Option(0, "--agent-exit", help="Agent exit code"),
    validate_status: str = typer.Option("passed", "--validate-status", help="Validation status"),
    outcome: str = typer.Option("progressed", "--outcome", "-o", help="Outcome"),
    validation_hint: str = typer.Option("", "--validation-hint", help="Validation signal hint"),
):
    """Write a structured entry to RECENT.md."""
    ensure_initialized()
    files = get_files()

    writer = TierWriter(files)

    try:
        written = writer.write_recent_entry(
            iteration=iteration,
            item_id=item_id,
            item_title=item_title,
            work_mode=work_mode,
            agent_exit=agent_exit,
            validate_status=validate_status,
            outcome=outcome,
            validation_hint=validation_hint,
        )
        if written:
            console.print(f"[success]Wrote recent entry for iteration {iteration}[/success]")
        else:
            console.print("[info]Skipped duplicate entry[/info]")
    except TierValidationError as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


@app.command("lint-tiers")
def lint_tiers(
    tier: str = typer.Option(
        "all", "--tier", "-t", help="Tier to lint: guardrails, style, recent, all"
    ),
    strict: bool = typer.Option(False, "--strict", "-s", help="Treat warnings as errors"),
):
    """Lint tier files for formatting issues and validation errors."""
    ensure_initialized()
    files = get_files()

    tiers_to_lint: list[KnowledgeTier] = []
    if tier == "all":
        tiers_to_lint = [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]
    else:
        tier_map = {
            "guardrails": KnowledgeTier.GUARDRAILS,
            "style": KnowledgeTier.STYLE,
            "recent": KnowledgeTier.RECENT,
        }
        if tier not in tier_map:
            console.print(f"[error]Unknown tier: {tier}[/error]")
            raise typer.Exit(1)
        tiers_to_lint = [tier_map[tier]]

    all_passed = True
    for kt in tiers_to_lint:
        content = files.read_tier(kt)
        errors, warnings = lint_tier_file(kt, content, strict=strict)

        console.print(f"\n[accent]{kt.value}.md[/accent]")

        if errors:
            console.print(f"  [error]Errors ({len(errors)}):[/error]")
            for error in errors:
                console.print(f"    • {error}")
            all_passed = False
        else:
            console.print("  [success]No errors[/success]")

        if warnings:
            console.print(f"  [warning]Warnings ({len(warnings)}):[/warning]")
            for warning in warnings:
                console.print(f"    • {warning}")
            if strict:
                all_passed = False

    if all_passed:
        console.print("\n[success]All tier files passed linting[/success]")
        raise typer.Exit(0)
    else:
        console.print("\n[error]Some tier files have issues[/error]")
        raise typer.Exit(1)


@app.command("tier-stats")
def tier_stats():
    """Show statistics for all tier files."""
    ensure_initialized()
    files = get_files()

    table = Table(title="Tier File Statistics")
    table.add_column("Tier", style="accent")
    table.add_column("Entries", justify="right")
    table.add_column("Size (chars)", justify="right")
    table.add_column("Lines", justify="right")
    table.add_column("Date Range")

    for tier in [KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE, KnowledgeTier.RECENT]:
        content = files.read_tier(tier)
        stats = get_tier_statistics(content)

        date_range = "N/A"
        if stats["date_range"]["earliest"] and stats["date_range"]["latest"]:
            earliest = stats["date_range"]["earliest"][:10]  # Just the date part
            latest = stats["date_range"]["latest"][:10]
            if earliest == latest:
                date_range = earliest
            else:
                date_range = f"{earliest} to {latest}"

        table.add_row(
            tier.value,
            str(stats["entry_count"]),
            f"{stats['content_size']:,}",
            str(stats["line_count"]),
            date_range,
        )

    console.print(table)


@app.command("compact-tiers")
def compact_tiers(
    max_entries: int | None = typer.Option(
        None,
        "--max-entries",
        "-n",
        min=10,
        max=200,
        help="Maximum entries per tier (overrides config)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict deduplication",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Show what would be changed without writing",
    ),
):
    """Compact tier files: normalize, deduplicate, and apply size budgets."""
    ensure_initialized()
    files = get_files()

    # Load config
    config_dict = files.read_config()
    config = TierCompactionConfig.from_config(config_dict)

    # Apply CLI overrides
    if max_entries is not None:
        config.max_entries_per_tier = max_entries
    if strict:
        config.strict_deduplication = True

    hook = TierCompactionHook(files, config)

    if dry_run:
        console.print("[dim]Dry run mode - no changes will be written[/dim]")

    summary = hook.compact_all()

    # Build result display
    lines = ["[success]Tier compaction complete[/success]"]
    lines.append("")

    for result in summary.results:
        tier_name = result.tier.value.upper()
        lines.append(f"[accent]{tier_name}:[/accent]")
        lines.append(f"  Entries: {result.entries_before} → {result.entries_after}")
        lines.append(f"  Size: {result.bytes_before:,} → {result.bytes_after:,} bytes")
        if result.duplicates_removed:
            lines.append(f"  Duplicates removed: {result.duplicates_removed}")
        if result.entries_summarized:
            lines.append(f"  Entries summarized: {result.entries_summarized}")
        lines.append("")

    lines.append("[bold]Total:[/bold]")
    lines.append(f"  Entries: {summary.total_entries_before} → {summary.total_entries_after}")
    lines.append(f"  Size: {summary.total_bytes_before:,} → {summary.total_bytes_after:,} bytes")
    if summary.total_duplicates_removed:
        lines.append(f"  Duplicates removed: {summary.total_duplicates_removed}")
    if summary.total_entries_summarized:
        lines.append(f"  Entries summarized: {summary.total_entries_summarized}")

    console.print(Panel.fit("\n".join(lines), title="compact-tiers"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
