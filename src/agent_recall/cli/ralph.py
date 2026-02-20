from __future__ import annotations

import asyncio
import json
import re
import shlex
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_recall.cli.stream_pipeline import (
    run_streaming_command,
    stream_debug_dir,
    stream_debug_enabled,
)
from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.config import load_config
from agent_recall.core.onboarding import inject_stored_api_keys
from agent_recall.llm import create_llm_provider, ensure_provider_dependency
from agent_recall.llm.base import LLMProvider
from agent_recall.ralph.context_refresh import ContextRefreshHook
from agent_recall.ralph.costs import (
    budget_exceeded,
    format_usd,
    summarize_costs,
)
from agent_recall.ralph.extraction import (
    extract_from_artifacts,
    extract_outcome,
    extract_token_usage,
)
from agent_recall.ralph.forecast import ForecastConfig, ForecastGenerator
from agent_recall.ralph.hooks import (
    build_hook_command,
    generate_notification_script,
    generate_post_tool_script,
    generate_pre_tool_script,
    get_hook_paths,
    install_hooks,
    uninstall_hooks,
)
from agent_recall.ralph.iteration_store import IterationOutcome, IterationReportStore
from agent_recall.ralph.loop import RalphLoop, RalphStateManager, RalphStatus
from agent_recall.ralph.notifications import (
    build_notification_content,
    dispatch_claude_notification,
    dispatch_notification,
)
from agent_recall.ralph.opencode_plugin import (
    get_opencode_plugin_paths,
    install_opencode_plugin,
    uninstall_opencode_plugin,
)
from agent_recall.ralph.prd_archive import PRDArchive
from agent_recall.ralph.synthesis import ClimateSynthesizer, SynthesisConfig
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LLMConfig, RalphNotificationEvent
from agent_recall.storage.remote import resolve_shared_db_path
from agent_recall.storage.sqlite import SQLiteStorage

ralph_app = typer.Typer(help="Manage Ralph loop configuration")
hooks_app = typer.Typer(help="Manage Ralph Claude Code hooks")
plugin_app = typer.Typer(help="Manage Ralph OpenCode plugins")

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"
DEFAULT_WATCH_POLL_SECONDS = 0.5

CODING_CLIS: dict[str, dict[str, str]] = {
    "claude-code": {"binary": "claude", "model_flag": "--model"},
    "codex": {"binary": "codex", "model_flag": "--model"},
    "opencode": {"binary": "opencode", "model_flag": "--model"},
}

_theme_manager = ThemeManager(DEFAULT_THEME)
console = Console(theme=_theme_manager.get_theme())


def _get_theme_manager() -> ThemeManager:
    global console
    if AGENT_DIR.exists():
        files = FileStorage(AGENT_DIR)
        config_dict = files.read_config()
        theme_name = config_dict.get("theme", {}).get("name", DEFAULT_THEME)
        if theme_name != _theme_manager.get_theme_name():
            _theme_manager.set_theme(theme_name)
            console = Console(theme=_theme_manager.get_theme())
    return _theme_manager


def ensure_initialized() -> None:
    if AGENT_DIR.exists():
        return
    _get_theme_manager()
    console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
    raise typer.Exit(1)


def get_agent_dir() -> Path:
    return Path.cwd() / ".agent"


def get_ralph_components() -> tuple[Path, SQLiteStorage, FileStorage]:
    agent_dir = get_agent_dir()
    if not agent_dir.exists():
        _get_theme_manager()
        console.print("[error]Not initialized. Run 'agent-recall init' first.[/error]")
        raise typer.Exit(1)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)
    return agent_dir, storage, files


def get_default_prd_path() -> Path:
    config_path = Path(".agent/config.json")
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            custom_path = config.get("ralph", {}).get("prd_path")
            if custom_path:
                return Path(custom_path)
        except (OSError, json.JSONDecodeError):
            pass

    candidates = [
        Path(".agent/ralph/prd.json"),
        Path("agent_recall/ralph/prd.json"),
        Path("prd.json"),
    ]

    # First pass: look for a file that exists AND has items
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if data.get("items"):
                    return candidate
            except (OSError, json.JSONDecodeError):
                pass

    # Second pass: if none have items, return the first that exists
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def get_default_script_path() -> Path:
    candidates = [
        Path("agent_recall/scripts/ralph-agent-recall-loop.sh"),
        Path("scripts/ralph-agent-recall-loop.sh"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def load_prd_items(prd_path: Path) -> list[dict[str, Any]]:
    """Load PRD items from JSON file. Returns empty list on error."""
    try:
        data = json.loads(prd_path.read_text(encoding="utf-8"))
        items = data.get("items")
        return list(items) if isinstance(items, list) else []
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def validate_prd_ids(
    prd_path: Path,
    ids: list[str],
    max_iterations: int,
) -> tuple[list[str], list[str]]:
    """
    Validate PRD IDs. Returns (valid_ids, invalid_ids).
    Raises no error; caller checks invalid_ids and count vs max_iterations.
    """
    items = load_prd_items(prd_path)
    valid_ids_set = {str(it.get("id", "")) for it in items if it.get("id")}
    valid: list[str] = []
    invalid: list[str] = []
    for i in ids:
        s = str(i).strip()
        if not s:
            continue
        if s in valid_ids_set:
            valid.append(s)
        else:
            invalid.append(s)
    return (valid, invalid)


def read_ralph_config(files: FileStorage) -> dict[str, Any]:
    config_dict = files.read_config()
    ralph_config = config_dict.get("ralph", {}) if isinstance(config_dict, dict) else {}
    return ralph_config if isinstance(ralph_config, dict) else {}


def _read_notification_config(
    ralph_config: dict[str, Any],
) -> tuple[bool, list[RalphNotificationEvent]]:
    notifications = ralph_config.get("notifications")
    if not isinstance(notifications, dict):
        return False, []
    enabled = bool(notifications.get("enabled"))
    events_raw = notifications.get("events")
    if not isinstance(events_raw, list):
        return enabled, []
    parsed: list[RalphNotificationEvent] = []
    for value in events_raw:
        if isinstance(value, RalphNotificationEvent):
            parsed.append(value)
            continue
        if not isinstance(value, str):
            continue
        try:
            parsed.append(RalphNotificationEvent(value))
        except ValueError:
            continue
    return enabled, parsed


def build_agent_cmd_from_ralph_config(ralph_config: dict[str, Any]) -> str | None:
    """Build shell-safe --agent-cmd from ralph coding_cli/cli_model settings."""
    coding_cli_value = ralph_config.get("coding_cli")
    coding_cli = (
        str(coding_cli_value).strip()
        if isinstance(coding_cli_value, str) and coding_cli_value.strip()
        else ""
    )
    if not coding_cli:
        return None
    cli_info = CODING_CLIS.get(coding_cli)
    if not cli_info:
        return None

    binary = str(cli_info.get("binary") or "").strip()
    model_flag = str(cli_info.get("model_flag") or "").strip()
    if not binary:
        return None

    cli_model_value = ralph_config.get("cli_model")
    cli_model = (
        str(cli_model_value).strip()
        if isinstance(cli_model_value, str) and cli_model_value.strip()
        else ""
    )

    # OpenCode behaves better when the prompt is passed as an explicit arg from a file
    # path placeholder, rather than piping stdin through a generic --print mode.
    if coding_cli == "opencode":
        model_segment = ""
        if cli_model:
            model_segment = f"-m {shlex.quote(cli_model)} "
        return f'{shlex.quote(binary)} run {model_segment}"$(cat {{prompt_file}})"'

    if coding_cli == "codex":
        # Codex CLI no longer supports --print; use non-interactive exec and read
        # the generated prompt from stdin (the shell loop already redirects stdin).
        parts = [
            binary,
            "--ask-for-approval",
            "never",
            "exec",
            "--sandbox",
            "danger-full-access",
        ]
        if cli_model and model_flag:
            parts.extend([model_flag, cli_model])
        parts.append("-")
        return " ".join(shlex.quote(part) for part in parts)

    parts = [binary, "--print"]
    if cli_model and model_flag:
        parts.extend([model_flag, cli_model])
    return " ".join(shlex.quote(part) for part in parts)


def write_ralph_config(files: FileStorage, updates: dict[str, object]) -> dict[str, object]:
    config_dict = files.read_config()
    if "ralph" not in config_dict or not isinstance(config_dict.get("ralph"), dict):
        config_dict["ralph"] = {}
    ralph_config = config_dict["ralph"]
    if isinstance(ralph_config, dict):
        ralph_config.update(updates)
    config_dict["ralph"] = ralph_config
    files.write_config(config_dict)
    return config_dict


@lru_cache(maxsize=1)
def get_storage() -> Storage:
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


def _ensure_agent_dir_exists() -> None:
    if not AGENT_DIR.exists():
        _get_theme_manager()
        console.print("[error].agent directory not found. Run 'agent-recall init'.[/error]")
        raise typer.Exit(1)


def _get_llm() -> LLMProvider:
    inject_stored_api_keys()
    files = get_files()
    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))
    ok, message = ensure_provider_dependency(llm_config.provider, auto_install=True)
    if not ok:
        raise RuntimeError(message or "Provider dependency setup failed.")
    if message:
        console.print(f"[dim]{message}[/dim]")
    return create_llm_provider(llm_config)


def _read_validation_output(runtime_dir: Path, iteration: int) -> list[str]:
    log_path = runtime_dir / f"validate-{iteration}.log"
    if not log_path.exists():
        return []
    try:
        return log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _read_agent_exit_code(runtime_dir: Path, iteration: int) -> int:
    log_path = runtime_dir / f"agent-{iteration}.log"
    if not log_path.exists():
        return 0
    try:
        content = log_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in reversed(content.splitlines()):
        if "exit code" in line.lower():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return 0
    return 0


def _read_agent_output(runtime_dir: Path, iteration: int) -> list[str]:
    log_path = runtime_dir / f"agent-{iteration}.log"
    if not log_path.exists():
        return []
    try:
        return log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def render_ralph_status(_config_dict: dict[str, object]) -> list[str]:
    ralph_config = read_ralph_config(FileStorage(AGENT_DIR))

    enabled_value = ralph_config.get("enabled")
    enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False

    max_iterations_value = ralph_config.get("max_iterations")
    max_iterations = (
        int(max_iterations_value) if isinstance(max_iterations_value, int | float) else 10
    )

    sleep_seconds_value = ralph_config.get("sleep_seconds")
    sleep_seconds = int(sleep_seconds_value) if isinstance(sleep_seconds_value, int | float) else 2

    compact_mode_value = ralph_config.get("compact_mode")
    compact_mode = (
        compact_mode_value
        if isinstance(compact_mode_value, str) and compact_mode_value
        else "always"
    )

    cost_budget_value = ralph_config.get("cost_budget_usd")
    cost_budget = (
        float(cost_budget_value)
        if isinstance(cost_budget_value, int | float) and cost_budget_value >= 0
        else None
    )

    state = "enabled" if enabled else "disabled"
    recent_file = AGENT_DIR / "RECENT.md"
    last_run_timestamp: str | None = None
    last_outcome: str | None = None
    if recent_file.exists():
        try:
            current_heading: str | None = None
            for raw_line in recent_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith("## "):
                    current_heading = line.removeprefix("## ").strip()
                elif line.startswith("- Outcome:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        last_outcome = value
                        last_run_timestamp = current_heading
        except OSError:
            pass
    selected_prd_value = ralph_config.get("selected_prd_ids")
    selected_prd_ids: list[str] | None = None
    if isinstance(selected_prd_value, list):
        selected_prd_ids = [str(x) for x in selected_prd_value if x]
    selected_prd_display = (
        ", ".join(selected_prd_ids) if selected_prd_ids else "all (model decides)"
    )

    store = IterationReportStore(AGENT_DIR / "ralph")
    cost_summary = summarize_costs(store.load_all())
    cost_total_text = format_usd(cost_summary.total_cost_usd)
    if cost_budget is None:
        cost_budget_text = "-"
    else:
        cost_budget_text = format_usd(cost_budget)
    budget_alert = (
        "" if not budget_exceeded(cost_summary.total_cost_usd, cost_budget) else " (exceeded)"
    )

    coding_cli_value = ralph_config.get("coding_cli")
    coding_cli = (
        str(coding_cli_value)
        if isinstance(coding_cli_value, str) and coding_cli_value
        else "not set"
    )
    cli_model_value = ralph_config.get("cli_model")
    cli_model = (
        str(cli_model_value) if isinstance(cli_model_value, str) and cli_model_value else "not set"
    )

    lines = [
        "[bold]Ralph Loop:[/bold]",
        f"  Status: {state}",
        f"  Coding CLI:     {coding_cli}",
        f"  CLI Model:      {cli_model}",
        f"  Max iterations: {max_iterations}",
        f"  Sleep seconds:  {sleep_seconds}",
        f"  Compact mode:   {compact_mode}",
        f"  Cost total:     {cost_total_text}",
        f"  Cost budget:    {cost_budget_text}{budget_alert}",
        f"  Selected PRDs:  {selected_prd_display}",
        f"  Last run:       {last_run_timestamp or 'none'}",
        f"  Last outcome:   {last_outcome or 'none'}",
        "",
    ]
    return lines


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def _render_status_mode_a(agent_dir: Path) -> None:
    manager = RalphStateManager(agent_dir)
    state = manager.load()
    state_payload: dict[str, Any] = {}
    if manager.state_path.exists():
        try:
            payload = json.loads(manager.state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                state_payload = payload
        except (OSError, json.JSONDecodeError):
            state_payload = {}

    status_style = "green" if state.status == RalphStatus.ENABLED else "red"
    lines = [
        f"Status: [{status_style}]{state.status.value}[/{status_style}]",
        f"Current iteration: {int(state_payload.get('current_iteration') or 0)}",
        f"Total iterations: {state.total_iterations}",
        f"Successful iterations: {int(state_payload.get('successful_iterations') or 0)}",
        f"Failed iterations: {int(state_payload.get('failed_iterations') or 0)}",
    ]
    last_run_at = state_payload.get("last_run_at")
    if isinstance(last_run_at, str) and last_run_at:
        lines.append(f"Last run: {last_run_at}")
    last_outcome = state_payload.get("last_outcome")
    if isinstance(last_outcome, str) and last_outcome:
        lines.append(f"Last outcome: {last_outcome}")
    prd_path = state_payload.get("prd_path")
    if isinstance(prd_path, str) and prd_path:
        lines.append(f"PRD path: {prd_path}")

    panel = Panel.fit("\n".join(lines), title="Ralph Status")
    console.print(panel)

    items = state_payload.get("items")
    item_list = list(items) if isinstance(items, list) else []
    table = Table(title="PRD Items", box=box.SIMPLE)
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Iterations", justify="right")

    status_styles = {
        "pending": "dim",
        "in_progress": "yellow",
        "completed": "green",
        "blocked": "red",
        "skipped": "dim",
    }
    for item in item_list:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        title = _truncate(str(item.get("title") or ""), 40)
        status_raw = str(item.get("status") or "pending").lower()
        style = status_styles.get(status_raw, "dim")
        status_text = f"[{style}]{status_raw}[/{style}]"
        iterations = str(item.get("iterations") or "0")
        table.add_row(item_id, title, status_text, iterations)
    console.print(table)


def _render_status_mode_b(prd_path: Path) -> None:
    try:
        payload = json.loads(prd_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[error]Invalid PRD JSON: {exc}[/error]")
        raise typer.Exit(1)

    if not isinstance(payload, dict):
        console.print(f"[error]Invalid PRD JSON: {prd_path}[/error]")
        raise typer.Exit(1)

    project = str(payload.get("project") or "Unknown")
    version = str(payload.get("version") or "")
    items = payload.get("items")
    item_list = list(items) if isinstance(items, list) else []

    total = len(item_list)
    passed = sum(1 for item in item_list if isinstance(item, dict) and item.get("passes") is True)
    remaining = total - passed

    lines = [
        f"PRD path: {prd_path}",
        f"Project: {project}",
        f"Version: {version or 'unknown'}",
        f"Total items: {total}",
        f"Completed: [green]{passed}[/green]",
        f"Remaining: [yellow]{remaining}[/yellow]",
    ]
    console.print(Panel.fit("\n".join(lines), title="Ralph PRD"))

    table = Table(title="PRD Items", box=box.SIMPLE)
    table.add_column("ID", style="cyan")
    table.add_column("Priority", justify="right")
    table.add_column("Title")
    table.add_column("Status")

    def sort_key(item: dict[str, Any]) -> int:
        priority = item.get("priority")
        return int(priority) if isinstance(priority, int) else 999999

    for item in sorted((i for i in item_list if isinstance(i, dict)), key=sort_key):
        item_id = str(item.get("id") or "")
        priority_raw = item.get("priority")
        priority = str(priority_raw) if isinstance(priority_raw, int) else "-"
        title = _truncate(str(item.get("title") or ""), 50)
        if item.get("passes") is True:
            status_text = "[green]✓ Passed[/green]"
        else:
            status_text = "[yellow]Pending[/yellow]"
        table.add_row(item_id, priority, title, status_text)
    console.print(table)


@ralph_app.command("status")
def ralph_status() -> None:
    """Show Ralph loop status."""
    _get_theme_manager()
    prd_path = get_default_prd_path()
    if prd_path.exists():
        _render_status_mode_b(prd_path)
        return

    agent_dir = get_agent_dir()
    if not agent_dir.exists():
        console.print(
            "[error]No PRD JSON or Ralph state found. "
            "Run 'agent-recall init' or provide a PRD JSON file.[/error]"
        )
        raise typer.Exit(1)

    state_manager = RalphStateManager(agent_dir)
    if state_manager.state_path.exists():
        _render_status_mode_a(agent_dir)
        return

    files = get_files()
    config_dict = files.read_config()
    lines = render_ralph_status(config_dict)
    console.print(Panel.fit("\n".join(lines), title="Ralph Loop"))


@ralph_app.command("enable")
def ralph_enable(
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        min=1,
        help="Max iterations for Ralph loop config",
    ),
    prd_file: Path | None = typer.Option(
        None,
        "--prd-file",
        "--prd",
        "-p",
        help="Path to PRD JSON file (default: auto-detected)",
    ),
    sleep_seconds: int | None = typer.Option(
        None,
        "--sleep-seconds",
        min=0,
        help="Sleep seconds between iterations",
    ),
    compact_mode: str | None = typer.Option(
        None,
        "--compact-mode",
        help="Compact mode: always, on-failure, off",
    ),
) -> None:
    """Enable the Ralph loop."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    if compact_mode is not None:
        compact_mode = compact_mode.strip().lower()
        if compact_mode not in {"always", "on-failure", "off"}:
            console.print("[error]Invalid compact mode. Use always, on-failure, or off.[/error]")
            raise typer.Exit(1)

    agent_dir, storage, loop_files = get_ralph_components()
    loop = RalphLoop(agent_dir, storage, loop_files)
    updates: dict[str, object] = {"enabled": True}
    if max_iterations is not None:
        updates["max_iterations"] = max_iterations
    if sleep_seconds is not None:
        updates["sleep_seconds"] = sleep_seconds
    if compact_mode is not None:
        updates["compact_mode"] = compact_mode
    config_dict = write_ralph_config(files, updates)

    if prd_file is None:
        loop.enable()
        lines = render_ralph_status(config_dict)
        console.print(Panel.fit("\n".join(lines), title="Ralph Loop Updated"))
        return

    if not prd_file.exists():
        console.print(f"[error]PRD file not found: {prd_file}[/error]")
        raise typer.Exit(1)
    item_count = loop.initialize_from_prd(prd_file)
    console.print(f"[success]✓ Ralph enabled ({item_count} PRD items)[/success]")


@ralph_app.command("disable")
def ralph_disable() -> None:
    """Disable the Ralph loop."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    agent_dir, storage, loop_files = get_ralph_components()
    loop = RalphLoop(agent_dir, storage, loop_files)
    state = loop.disable()
    config_dict = write_ralph_config(files, {"enabled": False})
    console.print(f"[warning]Ralph disabled (total iterations: {state.total_iterations})[/warning]")
    lines = render_ralph_status(config_dict)
    console.print(Panel.fit("\n".join(lines), title="Ralph Loop Updated"))


@ralph_app.command("cost-report")
def ralph_cost_report(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output cost summary as JSON",
    ),
) -> None:
    """Show token and cost summary for Ralph iterations."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    ralph_config = read_ralph_config(FileStorage(AGENT_DIR))
    cost_budget_value = ralph_config.get("cost_budget_usd")
    cost_budget = (
        float(cost_budget_value)
        if isinstance(cost_budget_value, int | float) and cost_budget_value >= 0
        else None
    )
    store = IterationReportStore(AGENT_DIR / "ralph")
    summary = summarize_costs(store.load_all())

    table = Table(title="Ralph Cost Report", box=box.SIMPLE, expand=True)
    table.add_column("Item", style="table_header", no_wrap=True)
    table.add_column("Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right")

    if json_output:
        payload = {
            "total_tokens": summary.total_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "cost_budget_usd": cost_budget,
            "exceeded": budget_exceeded(summary.total_cost_usd, cost_budget),
            "items": [
                {
                    "item_id": item.item_id,
                    "item_title": item.item_title,
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                    "total_tokens": item.total_tokens,
                    "cost_usd": item.cost_usd,
                }
                for item in summary.items
            ],
        }
        console.print(json.dumps(payload, indent=2))
        return

    if not summary.items:
        console.print("[dim]No token usage recorded yet.[/dim]")
    else:
        for item in summary.items:
            label = item.item_id
            if item.item_title:
                label = f"{item.item_id} · {item.item_title}"
            table.add_row(label, str(item.total_tokens), format_usd(item.cost_usd))
        console.print(table)

    budget_text = "-" if cost_budget is None else format_usd(cost_budget)
    total_text = format_usd(summary.total_cost_usd)
    budget_note = "" if not budget_exceeded(summary.total_cost_usd, cost_budget) else " (exceeded)"
    console.print(
        "\n".join(
            [
                f"Total tokens: {summary.total_tokens}",
                f"Total cost: {total_text}",
                f"Budget: {budget_text}{budget_note}",
            ]
        )
    )


@ralph_app.command("run")
def ralph_run(
    max_mode_a_iterations: int | None = typer.Option(
        None,
        "--max",
        "-m",
        min=1,
        help="Max iterations for Python loop mode",
    ),
    item_id: str | None = typer.Option(
        None,
        "--item",
        "-i",
        help="PRD item id to run in Python loop mode",
    ),
    agent_cmd: str | None = typer.Option(
        None,
        "--agent-cmd",
        "-a",
        help="Agent command for bash loop mode",
    ),
    validate_cmd: str | None = typer.Option(
        None,
        "--validate-cmd",
        "-v",
        help="Validation command for bash loop mode",
    ),
    agent_transport: str = typer.Option(
        "pipe",
        "--agent-transport",
        help="Agent transport for bash loop mode: pipe, pty, auto",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        min=1,
        help="Max iterations for bash loop mode",
    ),
    prd_file: Path | None = typer.Option(
        None,
        "--prd-file",
        "-p",
        help="Path to PRD JSON file (default: auto-detected)",
    ),
    prompt_prd_top_n: int = typer.Option(
        8,
        "--prompt-prd-top-n",
        min=1,
        help="Top-N unpassed PRD items to include in each iteration prompt",
    ),
    rules_file: Path | None = typer.Option(
        None,
        "--rules-file",
        help="Path to RULES.md for loop prompt context (default: .agent/RULES.md)",
    ),
    compact_mode: str = typer.Option(
        "always",
        "--compact-mode",
        help="Compact mode: always, on-failure, off",
    ),
    sleep_seconds: int = typer.Option(
        2,
        "--sleep-seconds",
        min=0,
        help="Sleep seconds between iterations",
    ),
) -> None:
    """Run the Ralph loop."""
    _get_theme_manager()
    if agent_cmd is None:
        ensure_initialized()
        files = get_files()
        ralph_cfg = read_ralph_config(files)
        notify_enabled, notify_events = _read_notification_config(ralph_cfg)
        agent_dir = get_agent_dir()
        state_manager = RalphStateManager(agent_dir)
        if state_manager.load().status == RalphStatus.DISABLED:
            console.print("[error]Ralph loop is disabled. Run 'agent-recall ralph enable'.[/error]")
            raise typer.Exit(1)
        enabled_value = ralph_cfg.get("enabled")
        enabled = bool(enabled_value) if isinstance(enabled_value, bool) else False
        if not enabled:
            console.print("[error]Ralph loop is disabled. Run 'agent-recall ralph enable'.[/error]")
            raise typer.Exit(1)

        python_max_iterations = max_mode_a_iterations
        selected_value = ralph_cfg.get("selected_prd_ids")
        selected_ids: list[str] | None = None
        if isinstance(selected_value, list) and selected_value:
            selected_ids = [str(x) for x in selected_value if x]

        if python_max_iterations is None:
            max_iterations_value = ralph_cfg.get("max_iterations")
            python_max_iterations = (
                int(max_iterations_value) if isinstance(max_iterations_value, int | float) else None
            )

        def progress_callback(event: dict[str, Any]) -> None:
            event_type = str(event.get("event") or "")
            if event_type == "output_line":
                line = str(event.get("line") or "")
                if line:
                    console.print(f"[dim]{line}[/dim]")
                return
            if event_type == "iteration_started":
                iteration = event.get("iteration")
                item = event.get("item_id")
                console.print(f"[cyan]Iteration {iteration}: {item} started[/cyan]")
            elif event_type == "agent_complete":
                exit_code = int(event.get("exit_code") or 0)
                if exit_code == 0:
                    console.print("[green]✓ Agent complete[/green]")
                else:
                    console.print(f"[red]✗ Agent failed (exit {exit_code})[/red]")
            elif event_type == "validation_complete":
                success = bool(event.get("success"))
                hint = str(event.get("hint") or "")
                if success:
                    console.print("[green]✓ Validation passed[/green]")
                else:
                    message = _truncate(hint, 80) if hint else "Validation failed"
                    console.print(f"[red]✗ {message}[/red]")
                    if notify_enabled:
                        dispatch_notification(
                            RalphNotificationEvent.VALIDATION_FAILED,
                            enabled=notify_enabled,
                            enabled_events=notify_events,
                            iteration=event.get("iteration"),
                        )
            elif event_type == "iteration_complete":
                outcome = str(event.get("outcome") or "")
                duration = float(event.get("duration_seconds") or 0)
                console.print(f"[dim]Iteration complete ({outcome}) in {duration:.2f}s[/dim]")
                if notify_enabled:
                    dispatch_notification(
                        RalphNotificationEvent.ITERATION_COMPLETE,
                        enabled=notify_enabled,
                        enabled_events=notify_events,
                        iteration=event.get("iteration"),
                    )

        coding_cli_value = ralph_cfg.get("coding_cli")
        coding_cli = (
            str(coding_cli_value)
            if isinstance(coding_cli_value, str) and coding_cli_value
            else None
        )
        cli_model_value = ralph_cfg.get("cli_model")
        cli_model = (
            str(cli_model_value) if isinstance(cli_model_value, str) and cli_model_value else None
        )

        agent_dir, storage, loop_files = get_ralph_components()
        loop = RalphLoop(agent_dir, storage, loop_files)
        summary = asyncio.run(
            loop.run_loop(
                max_iterations=python_max_iterations,
                item_id=item_id,
                selected_prd_ids=selected_ids,
                progress_callback=progress_callback,
                coding_cli=coding_cli,
                cli_model=cli_model,
            )
        )
        panel_lines = [
            f"Total iterations: {summary.get('total_iterations', 0)}",
            f"Passed: [green]{summary.get('passed', 0)}[/green]",
            f"Failed: [red]{summary.get('failed', 0)}[/red]",
        ]
        console.print(Panel.fit("\n".join(panel_lines), title="Ralph Run Summary"))
        if notify_enabled:
            dispatch_notification(
                RalphNotificationEvent.LOOP_FINISHED,
                enabled=notify_enabled,
                enabled_events=notify_events,
            )
        return

    compact_mode = compact_mode.strip().lower()
    if compact_mode not in {"always", "on-failure", "off"}:
        console.print("[error]Invalid compact mode. Use always, on-failure, or off.[/error]")
        raise typer.Exit(1)
    agent_transport = agent_transport.strip().lower()
    if agent_transport not in {"pipe", "pty", "auto"}:
        console.print("[error]Invalid agent transport. Use pipe, pty, or auto.[/error]")
        raise typer.Exit(1)

    script_path = get_default_script_path()
    if not script_path.exists():
        console.print(f"[error]Ralph loop script not found: {script_path}[/error]")
        raise typer.Exit(1)

    prd_path = prd_file or get_default_prd_path()
    if not prd_path.exists():
        console.print(f"[error]PRD file not found: {prd_path}[/error]")
        raise typer.Exit(1)

    console.print(f"[dim]PRD: {prd_path}[/dim]")
    console.print(f"[dim]Max iterations: {max_iterations}[/dim]")

    pre_iteration_cmd: str | None = None
    if AGENT_DIR.exists():
        try:
            files = get_files()
            ralph_cfg = read_ralph_config(files)
            if ralph_cfg.get("coding_cli") == "claude-code":
                pre_iteration_cmd = "uv run agent-recall ralph hooks install"
        except Exception:  # noqa: BLE001
            pre_iteration_cmd = None

    cmd = [
        str(script_path),
        "--agent-cmd",
        agent_cmd,
        "--max-iterations",
        str(max_iterations),
        "--prd-file",
        str(prd_path),
        "--compact-mode",
        compact_mode,
        "--agent-transport",
        agent_transport,
        "--sleep-seconds",
        str(sleep_seconds),
        "--prompt-prd-top-n",
        str(prompt_prd_top_n),
    ]
    if validate_cmd:
        cmd.extend(["--validate-cmd", validate_cmd])
    if rules_file is not None:
        cmd.extend(["--rules-file", str(rules_file)])
    if pre_iteration_cmd:
        cmd.extend(["--pre-iteration-cmd", pre_iteration_cmd])

    try:
        returncode = run_streaming_command(
            cmd,
            cwd=Path.cwd(),
            on_emit=lambda fragment: console.print(
                fragment,
                end="",
                markup=False,
                highlight=False,
            ),
            context="ralph_cli_shell_run",
            partial_flush_ms=120,
            transport="pipe",
        )
    except KeyboardInterrupt:
        console.print("[warning]Ralph loop interrupted.[/warning]")
        raise typer.Exit(130) from None
    if stream_debug_enabled():
        console.print(f"[dim]Stream debug artifacts: {stream_debug_dir(Path.cwd())}[/dim]")

    if returncode == 0:
        console.print("[success]✓ Ralph loop completed successfully.[/success]")
    elif returncode == 2:
        console.print("[warning]Ralph loop reached max iterations.[/warning]")
    else:
        console.print(f"[error]Ralph loop failed (exit {returncode}).[/error]")
    raise typer.Exit(returncode)


@ralph_app.command("select")
def ralph_select() -> None:
    """Show instructions for selecting PRD items."""
    _get_theme_manager()
    ensure_initialized()
    console.print(
        "[dim]PRD selection is interactive in the TUI. Use one of:[/dim]\n"
        "  • [bold]agent-recall open[/bold] then palette (Ctrl+P) → 'Choose PRD items'\n"
        "  • [bold]agent-recall ralph set-prds --prds AR-001,AR-002[/bold] to set via CLI"
    )


@ralph_app.command("get-selected-prds")
def ralph_get_selected_prds() -> None:
    """Output selected PRD IDs if configured."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()
    ralph_cfg = read_ralph_config(files)
    selected_value = ralph_cfg.get("selected_prd_ids")
    if isinstance(selected_value, list) and selected_value:
        console.print(",".join(str(x) for x in selected_value if x))


@ralph_app.command("set-prds")
def ralph_set_prds(
    prds: str = typer.Option(
        ...,
        "--prds",
        help="Comma-separated PRD IDs. Omit for all items (model decides).",
    ),
    prd_file: Path | None = typer.Option(
        None,
        "--prd-file",
        "--prd",
        "-p",
        help="Path to PRD JSON file (default: auto-detected)",
    ),
) -> None:
    """Set selected PRD IDs via CLI."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    raw_ids = [x.strip() for x in re.split(r"[,\s]+", prds) if x.strip()]
    prd_path = prd_file or get_default_prd_path()
    ralph_cfg = read_ralph_config(files)
    max_iter = int(ralph_cfg.get("max_iterations") or 10)
    updates: dict[str, object]
    if raw_ids:
        valid_ids, invalid_ids = validate_prd_ids(prd_path, raw_ids, max_iter)
        if invalid_ids:
            console.print(
                f"[error]Invalid PRD IDs (not found in {prd_path}): "
                f"{', '.join(invalid_ids)}[/error]"
            )
            raise typer.Exit(1)
        if len(valid_ids) > max_iter:
            console.print(
                f"[error]Selected {len(valid_ids)} PRDs exceeds max_iterations ({max_iter}). "
                f"Select at most {max_iter} items.[/error]"
            )
            raise typer.Exit(1)
        updates = {"selected_prd_ids": valid_ids}
    else:
        updates = {"selected_prd_ids": None}
    config_dict = write_ralph_config(files, updates)
    lines = render_ralph_status(config_dict)
    console.print(Panel.fit("\n".join(lines), title="Ralph PRD Selection Updated"))


@ralph_app.command("archive-completed")
def ralph_archive_completed(
    prd_file: Path | None = typer.Option(
        None,
        "--prd-file",
        "--prd",
        "-p",
        help="Path to PRD JSON file (default: auto-detected)",
    ),
    prune_only: bool = typer.Option(
        False,
        "--prune-only",
        help="Only prune archived items from PRD, do not archive new ones.",
    ),
    iteration: int | None = typer.Option(
        None,
        "--iteration",
        "-n",
        help="Iteration number for archive metadata",
    ),
) -> None:
    """Archive completed PRD items and optionally prune them."""
    _get_theme_manager()
    ensure_initialized()
    prd_path = prd_file or get_default_prd_path()
    if not prd_path.exists():
        console.print(f"[error]PRD file not found: {prd_path}[/error]")
        raise typer.Exit(1)
    archive = PRDArchive(AGENT_DIR, get_storage() if not prune_only else None)
    if prune_only:
        pruned = archive.prune_archived_from_prd(prd_path)
        if pruned:
            console.print(f"[success]✓ Pruned {pruned} archived item(s) from PRD[/success]")
        else:
            console.print("No archived items to prune from PRD")
        return

    archived_items = archive.archive_completed_from_prd(prd_path, iteration=int(iteration or 0))
    if not archived_items:
        console.print("No new items to archive")
        return
    console.print(f"[success]✓ Archived {len(archived_items)} item(s)[/success]")
    for item in archived_items:
        console.print(f"• {item.id}: {item.title}")


@ralph_app.command("search-archive")
def ralph_search_archive(
    query: str = typer.Argument(..., help="Search query for archive"),
    top_k: int = typer.Option(
        5,
        "--top",
        "-k",
        min=1,
        help="Max search results for search-archive",
    ),
) -> None:
    """Search archived PRD items."""
    _get_theme_manager()
    ensure_initialized()
    if not query.strip():
        console.print("[error]Search query is required.[/error]")
        raise typer.Exit(1)
    archive = PRDArchive(AGENT_DIR)
    results = archive.search(query.strip(), top_k=top_k)
    if not results:
        console.print("No matching archived items found")
        return
    table = Table(title="Archived PRD Search", box=box.SIMPLE)
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Score", justify="right")
    for item, score in results:
        table.add_row(item.id, item.title, f"{score:.3f}")
    console.print(table)


@ralph_app.command("refresh-context")
def ralph_refresh_context(
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Task description for refresh-context",
    ),
    item_id: str | None = typer.Option(
        None,
        "--item",
        "-i",
        help="PRD item id for refresh-context",
    ),
    iteration: int | None = typer.Option(
        None,
        "--iteration",
        "-n",
        help="Iteration number for refresh-context",
    ),
) -> None:
    """Refresh the Ralph context bundle."""
    _get_theme_manager()
    ensure_initialized()
    storage = get_storage()
    files = get_files()
    hook = ContextRefreshHook(AGENT_DIR, storage, files)
    try:
        summary = hook.refresh(task=task, item_id=item_id, iteration=iteration)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[error]Context refresh failed: {exc}[/error]")
        raise typer.Exit(1) from None
    adapters_written = summary.get("adapters_written")
    adapter_list = adapters_written if isinstance(adapters_written, list) else []
    adapter_names = ", ".join(str(name) for name in adapter_list)
    lines = [
        "[success]✓ Context refreshed[/success]",
        f"  Context length: {summary.get('context_length', 0)}",
        f"  Adapters: {adapter_names or 'none'}",
    ]
    task_value = summary.get("task")
    if isinstance(task_value, str) and task_value.strip():
        lines.append(f"  Task: {task_value}")
    console.print("\n".join(lines))


@ralph_app.command("set-agent")
def ralph_set_agent(
    cli: str = typer.Option(
        ...,
        "--cli",
        help="Coding CLI to use: claude-code, codex, or opencode",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use with the selected coding CLI",
    ),
) -> None:
    """Set the coding CLI and model for the Ralph loop."""
    _get_theme_manager()
    ensure_initialized()
    cli_name = cli.strip().lower()
    if cli_name not in CODING_CLIS:
        valid = ", ".join(sorted(CODING_CLIS))
        console.print(f"[error]Invalid coding CLI '{cli_name}'. Choose from: {valid}[/error]")
        raise typer.Exit(1)
    files = get_files()
    updates: dict[str, object] = {"coding_cli": cli_name}
    if model is not None:
        updates["cli_model"] = model.strip() or None
    config_dict = write_ralph_config(files, updates)
    lines = render_ralph_status(config_dict)
    console.print(Panel.fit("\n".join(lines), title="Ralph Agent Updated"))


@ralph_app.command("set-budget")
def ralph_set_budget(
    cost_budget_usd: float | None = typer.Option(
        None,
        "--cost-usd",
        help="USD budget to pause Ralph loop when exceeded",
    ),
) -> None:
    """Set the Ralph loop cost budget."""
    _get_theme_manager()
    ensure_initialized()
    files = get_files()

    updates: dict[str, object] = {}
    if cost_budget_usd is not None:
        if cost_budget_usd < 0:
            console.print("[error]Cost budget must be >= 0.[/error]")
            raise typer.Exit(1)
        updates["cost_budget_usd"] = cost_budget_usd
    else:
        updates["cost_budget_usd"] = None

    config_dict = write_ralph_config(files, updates)
    console.print("[success]✓ Updated Ralph cost budget[/success]")
    lines = render_ralph_status(config_dict)
    console.print(Panel.fit("\n".join(lines), title="Ralph Loop Updated"))


@ralph_app.command("watch")
def ralph_watch(
    poll_seconds: float = typer.Option(
        DEFAULT_WATCH_POLL_SECONDS,
        "--poll-seconds",
        min=0.1,
        help="Polling interval for log watcher",
    ),
    max_events: int | None = typer.Option(
        None,
        "--max-events",
        min=1,
        help="Stop after emitting N events",
    ),
    max_seconds: float | None = typer.Option(
        None,
        "--max-seconds",
        min=1.0,
        help="Stop after N seconds",
    ),
    start_at_end: bool = typer.Option(
        True,
        "--start-at-end/--start-at-beginning",
        help="Begin watching at end of log to only capture new events",
    ),
) -> None:
    """Watch Claude Code logs and emit live activity lines."""
    _get_theme_manager()
    from agent_recall.ingest.log_watcher import LogWatcher

    watcher = LogWatcher(
        project_path=Path.cwd(),
        poll_interval=poll_seconds,
        start_at_end=start_at_end,
    )

    def on_message(message: Any) -> None:
        timestamp = message.timestamp.isoformat() if message.timestamp else ""
        preview = _truncate(" ".join(message.content.split()), 160)
        console.print(f"[dim]{timestamp}[/dim] {message.role}: {preview}")

    count = watcher.watch(
        on_message=on_message,
        max_events=max_events,
        max_seconds=max_seconds,
    )
    console.print(f"[success]✓ Watched {count} new event(s)[/success]")


@hooks_app.command("install")
def ralph_hooks_install(
    settings_path: Path | None = typer.Option(
        None,
        "--settings-path",
        help="Path to Claude Code settings.json (default: ~/.claude/settings.json)",
    ),
) -> None:
    """Install Claude Code PreToolUse/PostToolUse hooks for Ralph."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    files = get_files()
    guardrails_text = files.read_tier(KnowledgeTier.GUARDRAILS)
    hook_paths = get_hook_paths(AGENT_DIR)
    generate_pre_tool_script(guardrails_text, hook_paths.pre_tool_path)
    generate_post_tool_script(hook_paths.post_tool_path, hook_paths.events_path)
    generate_notification_script(hook_paths.notification_path)
    pre_cmd = build_hook_command(hook_paths.pre_tool_path)
    post_cmd = build_hook_command(hook_paths.post_tool_path)
    notification_cmd = build_hook_command(hook_paths.notification_path)
    settings_file = settings_path or get_claude_settings_path()
    install_hooks(settings_file, pre_cmd, post_cmd, notification_cmd)
    console.print(
        "[success]✓ Claude Code hooks installed[/success]\n"
        f"  PreToolUse: {hook_paths.pre_tool_path}\n"
        f"  PostToolUse: {hook_paths.post_tool_path}\n"
        f"  Notification: {hook_paths.notification_path}\n"
        f"  Settings: {settings_file}"
    )


@hooks_app.command("uninstall")
def ralph_hooks_uninstall(
    settings_path: Path | None = typer.Option(
        None,
        "--settings-path",
        help="Path to Claude Code settings.json (default: ~/.claude/settings.json)",
    ),
) -> None:
    """Remove Ralph Claude Code hook entries."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    settings_file = settings_path or get_claude_settings_path()
    changed = uninstall_hooks(settings_file)
    if changed:
        console.print(f"[success]✓ Removed Ralph hooks from {settings_file}[/success]")
    else:
        console.print("No Ralph hooks found to remove.")


ralph_app.add_typer(hooks_app, name="hooks")


@plugin_app.command("opencode-install")
def ralph_opencode_install(
    project_dir: Path = typer.Option(
        Path.cwd(),
        "--project-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Project directory that owns the .opencode plugin",
    ),
) -> None:
    """Install OpenCode plugin for Ralph session events."""
    _get_theme_manager()
    changed = install_opencode_plugin(project_dir)
    paths = get_opencode_plugin_paths(project_dir)
    if changed:
        console.print(
            f"[success]✓ OpenCode plugin installed[/success]\n  Plugin: {paths.plugin_path}"
        )
    else:
        console.print(f"OpenCode plugin already installed at {paths.plugin_path}")


@plugin_app.command("opencode-uninstall")
def ralph_opencode_uninstall(
    project_dir: Path = typer.Option(
        Path.cwd(),
        "--project-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Project directory that owns the .opencode plugin",
    ),
) -> None:
    """Remove OpenCode plugin for Ralph session events."""
    _get_theme_manager()
    changed = uninstall_opencode_plugin(project_dir)
    paths = get_opencode_plugin_paths(project_dir)
    if changed:
        console.print(
            f"[success]✓ OpenCode plugin removed[/success]\n  Plugin: {paths.plugin_path}"
        )
    else:
        console.print(f"No OpenCode plugin found at {paths.plugin_path}")


ralph_app.add_typer(plugin_app, name="plugin")


@ralph_app.command("create-report")
def ralph_create_report(
    iteration: int = typer.Option(..., "--iteration", "-n", min=1),
    item_id: str = typer.Option(..., "--item-id"),
    item_title: str = typer.Option(..., "--item-title"),
) -> None:
    """Create a new iteration report (current.json)."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    if not item_id.strip():
        console.print("[error]item-id cannot be empty.[/error]")
        raise typer.Exit(1)
    store = IterationReportStore(AGENT_DIR / "ralph")
    report = store.create_for_iteration(iteration, item_id=item_id, item_title=item_title)
    console.print(
        "[success]✓ Created report for iteration "
        f"{report.iteration:03d} ({report.item_id})[/success]"
    )


@ralph_app.command("finalize-report")
def ralph_finalize_report(
    validation_exit: int = typer.Option(..., "--validation-exit"),
    validation_hint: str = typer.Option("", "--validation-hint"),
) -> None:
    """Finalize the current iteration report and archive it."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    store = IterationReportStore(AGENT_DIR / "ralph")
    report = store.finalize_current(validation_exit, validation_hint or None)
    if report is None:
        console.print("[warning]No current report to finalize.[/warning]")
        return
    archived_path = store.iterations_dir / f"{report.iteration:03d}.json"
    console.print(
        "[success]✓ Archived report to "
        f"{archived_path} (outcome: {report.outcome or 'UNKNOWN'})[/success]"
    )


@ralph_app.command("extract-iteration")
def ralph_extract_iteration(
    iteration: int = typer.Option(..., "--iteration", "-n", min=1),
    runtime_dir: Path = typer.Option(..., "--runtime-dir"),
) -> None:
    """Extract heuristic artifacts and update current.json."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    if not runtime_dir.exists():
        console.print(f"[error]Runtime dir not found: {runtime_dir}[/error]")
        raise typer.Exit(1)
    store = IterationReportStore(AGENT_DIR / "ralph")
    report = store.load_current()
    if report is None:
        console.print("[warning]No current report to update.[/warning]")
        return
    validation_output = _read_validation_output(runtime_dir, iteration)
    validation_exit = report.validation_exit_code or 0
    if report.validation_hint is None:
        report.validation_hint = "\n".join(validation_output) if validation_output else None
    agent_output = _read_agent_output(runtime_dir, iteration)
    agent_exit = _read_agent_exit_code(runtime_dir, iteration)
    artifacts = extract_from_artifacts(
        validation_exit=validation_exit,
        validation_output=validation_output,
        agent_exit=agent_exit,
        elapsed=0.0,
        timeout=0.0,
        repo_dir=Path.cwd(),
    )
    token_usage, token_model = extract_token_usage(agent_output)
    report.token_usage = token_usage
    report.token_model = token_model
    outcome = artifacts.get("outcome")
    if isinstance(outcome, IterationOutcome):
        report.outcome = outcome
    elif report.outcome is None:
        report.outcome = extract_outcome(validation_exit, agent_exit, 0.0, 0.0)
    failure_reason = artifacts.get("failure_reason")
    report.failure_reason = (
        failure_reason if isinstance(failure_reason, str) and failure_reason else None
    )
    validation_hint = artifacts.get("validation_hint")
    report.validation_hint = (
        validation_hint if isinstance(validation_hint, str) else report.validation_hint
    )
    files_changed = artifacts.get("files_changed")
    if isinstance(files_changed, list):
        report.files_changed = [str(item) for item in files_changed if item]
    git_diff = artifacts.get("git_diff")
    if isinstance(git_diff, str):
        store.save_current_diff(report, git_diff)
    commit_hash = artifacts.get("commit_hash")
    if isinstance(commit_hash, str) and commit_hash:
        report.commit_hash = commit_hash
    store.save_current(report)
    console.print("[success]✓ Updated current report with extracted artifacts[/success]")


@ralph_app.command("view-diff")
def ralph_view_diff(
    iteration: int | None = typer.Option(None, "--iteration", "-n"),
) -> None:
    """View the latest Ralph iteration diff."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    store = IterationReportStore(AGENT_DIR / "ralph")
    target_iteration: int | None = iteration
    if target_iteration is None:
        reports = store.load_recent(count=1)
        if not reports:
            console.print("[warning]No archived iteration reports found.[/warning]")
            return
        target_iteration = reports[0].iteration
    diff_text = store.load_diff_for_iteration(target_iteration)
    if not diff_text:
        console.print(f"[warning]No diff stored for iteration {target_iteration:03d}.[/warning]")
        return
    console.print(Panel(diff_text, title=f"Iteration {target_iteration:03d} Diff", box=box.SQUARE))


@ralph_app.command("rebuild-forecast")
def ralph_rebuild_forecast(
    use_llm: bool = typer.Option(False, "--use-llm"),
) -> None:
    """Rebuild RECENT.md from iteration reports."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    files = get_files()
    config = load_config(AGENT_DIR)
    forecast_cfg = config.ralph.forecast
    cfg = ForecastConfig(
        window=forecast_cfg.window,
        use_llm=forecast_cfg.use_llm if not use_llm else True,
        llm_on_consecutive_failures=forecast_cfg.llm_on_consecutive_failures,
        llm_model=forecast_cfg.llm_model,
    )
    generator = ForecastGenerator(AGENT_DIR / "ralph", files, config=cfg)
    llm = None
    if use_llm or cfg.use_llm:
        try:
            llm = _get_llm()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[warning]LLM unavailable, using heuristic forecast: {exc}[/warning]")
            llm = None
    content = generator.write_forecast(llm=llm)
    console.print(f"[success]✓ Forecast rebuilt ({len(content)} chars)[/success]")


@ralph_app.command("synthesize-climate")
def ralph_synthesize_climate(
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Synthesize GUARDRAILS.md and STYLE.md from iteration reports."""
    _get_theme_manager()
    _ensure_agent_dir_exists()
    config = load_config(AGENT_DIR)
    llm: LLMProvider | None = None
    try:
        llm = ralph_cli_get_llm()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[warning]LLM unavailable, using heuristic synthesis: {exc}[/warning]")
        llm = None
    synth_cfg = config.ralph.synthesis
    synthesizer = ClimateSynthesizer(
        AGENT_DIR / "ralph",
        get_files(),
        llm=llm,
        config=SynthesisConfig(
            max_guardrails=synth_cfg.max_guardrails,
            max_style=synth_cfg.max_style,
            auto_after_loop=synth_cfg.auto_after_loop,
        ),
    )
    if not force and not synthesizer.should_synthesize():
        console.print("No new iterations since last synthesis.")
        return
    results = asyncio.run(synthesizer.synthesize())
    console.print(
        "[success]✓ Synthesis complete "
        f"(guardrails: {results['guardrails']}, style: {results['style']})[/success]"
    )


def ralph_cli_get_llm():
    return _get_llm()


@ralph_app.command("notify")
def ralph_notify(
    event: str = typer.Option(
        "iteration_complete",
        "--event",
        "-e",
        help=(
            "Notification event: iteration_complete, validation_failed, "
            "loop_finished, budget_exceeded"
        ),
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        "-t",
        help="Override notification title",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Override notification message",
    ),
) -> None:
    """Send a Ralph desktop notification (macOS/Linux)."""
    _get_theme_manager()
    try:
        event_enum = RalphNotificationEvent(event)
    except ValueError:
        valid = ", ".join(e.value for e in RalphNotificationEvent)
        console.print(f"[error]Unknown event '{event}'. Choose from: {valid}[/error]")
        raise typer.Exit(1)

    info = build_notification_content(event_enum)
    final_title = title.strip() if isinstance(title, str) and title.strip() else info.title
    final_message = (
        message.strip() if isinstance(message, str) and message.strip() else info.message
    )
    success = dispatch_claude_notification({"title": final_title, "message": final_message})
    if success:
        console.print(f"[success]✓ Notification sent: {final_title}[/success]")
        return
    console.print("[warning]Notification unavailable on this platform.[/warning]")
