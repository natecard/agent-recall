from __future__ import annotations

import asyncio
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_recall.cli.theme import DEFAULT_THEME, ThemeManager
from agent_recall.core.config import load_config
from agent_recall.ralph.context_refresh import ContextRefreshHook
from agent_recall.ralph.loop import RalphLoop, RalphStateManager, RalphStatus
from agent_recall.ralph.prd_archive import PRDArchive
from agent_recall.storage import create_storage_backend
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage
from agent_recall.storage.remote import resolve_shared_db_path
from agent_recall.storage.sqlite import SQLiteStorage

ralph_app = typer.Typer(help="Manage Ralph loop configuration")

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"

CODING_CLIS: dict[str, dict[str, str]] = {
    "claude-code": {"binary": "claude", "model_flag": "--model"},
    "codex": {"binary": "codex", "model_flag": "--model"},
    "opencode": {"binary": "opencode", "model_flag": "--model"},
}

CLI_DEFAULT_MODELS: dict[str, list[str]] = {
    "claude-code": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "codex": [
        "o4-mini",
        "gpt-4o",
        "gpt-5.3-codex",
        "gpt-5-codex",
    ],
    "opencode": [],
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
    candidates = [
        Path(".agent/ralph/prd.json"),
        Path("agent_recall/ralph/prd.json"),
        Path("prd.json"),
    ]
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
            elif event_type == "iteration_complete":
                outcome = str(event.get("outcome") or "")
                duration = float(event.get("duration_seconds") or 0)
                console.print(f"[dim]Iteration complete ({outcome}) in {duration:.2f}s[/dim]")

        agent_dir, storage, loop_files = get_ralph_components()
        loop = RalphLoop(agent_dir, storage, loop_files)
        summary = asyncio.run(
            loop.run_loop(
                max_iterations=python_max_iterations,
                item_id=item_id,
                selected_prd_ids=selected_ids,
                progress_callback=progress_callback,
            )
        )
        panel_lines = [
            f"Total iterations: {summary.get('total_iterations', 0)}",
            f"Passed: [green]{summary.get('passed', 0)}[/green]",
            f"Failed: [red]{summary.get('failed', 0)}[/red]",
        ]
        console.print(Panel.fit("\n".join(panel_lines), title="Ralph Run Summary"))
        return

    compact_mode = compact_mode.strip().lower()
    if compact_mode not in {"always", "on-failure", "off"}:
        console.print("[error]Invalid compact mode. Use always, on-failure, or off.[/error]")
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
        "--sleep-seconds",
        str(sleep_seconds),
    ]
    if validate_cmd:
        cmd.extend(["--validate-cmd", validate_cmd])

    try:
        result = subprocess.run(cmd, cwd=Path.cwd(), check=False)
    except KeyboardInterrupt:
        console.print("[warning]Ralph loop interrupted.[/warning]")
        raise typer.Exit(130) from None

    if result.returncode == 0:
        console.print("[success]✓ Ralph loop completed successfully.[/success]")
    elif result.returncode == 2:
        console.print("[warning]Ralph loop reached max iterations.[/warning]")
    else:
        console.print(f"[error]Ralph loop failed (exit {result.returncode}).[/error]")
    raise typer.Exit(result.returncode)


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
