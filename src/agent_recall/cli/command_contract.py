from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from typer.main import get_command as get_typer_command


@dataclass(frozen=True)
class CommandContract:
    command: str
    aliases: tuple[str, ...]
    description: str
    surfaces: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class TuiCommandManifestEntry:
    command: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    notes: str = ""


_TUI_COMMAND_MANIFEST: tuple[TuiCommandManifestEntry, ...] = (
    TuiCommandManifestEntry(
        command="help",
        description="Show slash command help",
    ),
    TuiCommandManifestEntry(
        command="status",
        description="Show repository status and source availability",
    ),
    TuiCommandManifestEntry(
        command="sources",
        description="List detected conversation sources",
    ),
    TuiCommandManifestEntry(
        command="sessions",
        description="List discovered conversations",
    ),
    TuiCommandManifestEntry(
        command="sync",
        description="Ingest conversations (optionally skip compaction)",
    ),
    TuiCommandManifestEntry(
        command="sync background",
        description="Run background sync mode",
    ),
    TuiCommandManifestEntry(
        command="sync reset",
        description="Reset sync markers",
    ),
    TuiCommandManifestEntry(
        command="run",
        description="Alias for sync (includes synthesis by default)",
    ),
    TuiCommandManifestEntry(
        command="view",
        aliases=("menu",),
        description="Switch TUI dashboard views",
    ),
    TuiCommandManifestEntry(
        command="settings",
        aliases=("preferences",),
        description="Open the TUI settings view",
    ),
    TuiCommandManifestEntry(
        command="layout",
        description="Customise dashboard layout",
    ),
    TuiCommandManifestEntry(
        command="compact",
        description="Synthesize knowledge from recent logs",
    ),
    TuiCommandManifestEntry(
        command="feedback",
        description="Capture and evaluate retrieval feedback",
    ),
    TuiCommandManifestEntry(
        command="feedback add",
        description="Record up/down feedback for a retrieved chunk",
    ),
    TuiCommandManifestEntry(
        command="feedback list",
        description="List retrieval feedback rows",
    ),
    TuiCommandManifestEntry(
        command="feedback evaluate",
        description="Compare baseline vs feedback-aware retrieval quality",
    ),
    TuiCommandManifestEntry(
        command="tiers",
        description="Manage tier file workflows",
    ),
    TuiCommandManifestEntry(
        command="tiers compact",
        description="Compact tier files (GUARDRAILS/STYLE/RECENT)",
    ),
    TuiCommandManifestEntry(
        command="tiers lint",
        description="Lint tier files for structure issues",
    ),
    TuiCommandManifestEntry(
        command="tiers stats",
        description="Show tier file statistics",
    ),
    TuiCommandManifestEntry(
        command="tiers write guardrails",
        aliases=("tiers write guardrails-failure",),
        description="Append guardrails entries",
    ),
    TuiCommandManifestEntry(
        command="tiers write style",
        description="Append style entries",
    ),
    TuiCommandManifestEntry(
        command="tiers write recent",
        description="Append recent session summary entries",
    ),
    TuiCommandManifestEntry(
        command="config setup",
        description="Run repository setup",
    ),
    TuiCommandManifestEntry(
        command="config model",
        description="Configure provider and model defaults",
    ),
    TuiCommandManifestEntry(
        command="config settings",
        aliases=("config preferences", "config prefs"),
        description="Configure workspace settings",
    ),
    TuiCommandManifestEntry(
        command="theme list",
        description="List available themes",
    ),
    TuiCommandManifestEntry(
        command="theme show",
        description="Show active theme",
    ),
    TuiCommandManifestEntry(
        command="theme set",
        description="Apply a theme by name",
    ),
    TuiCommandManifestEntry(
        command="ralph status",
        description="Show Ralph loop status",
    ),
    TuiCommandManifestEntry(
        command="ralph enable",
        description="Enable Ralph loop",
    ),
    TuiCommandManifestEntry(
        command="ralph disable",
        description="Disable Ralph loop",
    ),
    TuiCommandManifestEntry(
        command="ralph select",
        description="Choose which PRD items to include in Ralph loop",
    ),
    TuiCommandManifestEntry(
        command="ralph view-diff",
        description="View the latest Ralph iteration diff",
    ),
    TuiCommandManifestEntry(
        command="ralph config",
        description="Open Ralph configuration modal",
    ),
    TuiCommandManifestEntry(
        command="ralph terminal",
        description="Toggle the embedded terminal panel",
    ),
    TuiCommandManifestEntry(
        command="ralph notify",
        aliases=("ralph notifications",),
        description="Toggle Ralph notifications",
    ),
    TuiCommandManifestEntry(
        command="ralph watch",
        description="Watch Claude Code logs and emit live activity",
    ),
    TuiCommandManifestEntry(
        command="ralph hooks install",
        description="Install Claude Code hooks for Ralph guardrails",
    ),
    TuiCommandManifestEntry(
        command="ralph hooks uninstall",
        description="Remove Claude Code hooks installed by Ralph",
    ),
    TuiCommandManifestEntry(
        command="ralph plugin opencode-install",
        description="Install OpenCode plugin for Ralph session events",
    ),
    TuiCommandManifestEntry(
        command="ralph plugin opencode-uninstall",
        description="Remove OpenCode plugin for Ralph session events",
    ),
    TuiCommandManifestEntry(
        command="context refresh",
        description="Refresh context bundle outputs",
    ),
    TuiCommandManifestEntry(
        command="quit",
        aliases=("exit", "q"),
        description="Exit the TUI",
    ),
)


def command_parity_report(
    *,
    cli_commands: set[str],
    tui_commands: set[str],
) -> dict[str, set[str]]:
    expected_cli: set[str] = set()
    expected_tui: set[str] = set()
    for contract in get_command_contract():
        if "cli" in contract.surfaces:
            expected_cli.add(contract.command)
        if "tui" in contract.surfaces:
            expected_tui.add(contract.command)
            expected_tui.update(contract.aliases)
    missing_in_tui = expected_tui - tui_commands
    missing_in_cli = expected_cli - cli_commands
    extra_in_tui = tui_commands - expected_tui
    extra_in_cli = cli_commands - expected_cli
    return {
        "missing_in_tui": missing_in_tui,
        "missing_in_cli": missing_in_cli,
        "extra_in_tui": extra_in_tui,
        "extra_in_cli": extra_in_cli,
    }


def clear_command_contract_cache() -> None:
    _cached_command_contract.cache_clear()


def get_command_contract() -> list[CommandContract]:
    return list(_cached_command_contract())


def get_registered_cli_command_paths() -> list[str]:
    app = _resolve_cli_app()
    return [path for path, _command in _walk_registered_cli_commands(app)]


@lru_cache(maxsize=1)
def _cached_command_contract() -> tuple[CommandContract, ...]:
    app = _resolve_cli_app()
    by_command: dict[str, CommandContract] = {}

    for path, click_command in _walk_registered_cli_commands(app):
        by_command[path] = CommandContract(
            command=path,
            aliases=(),
            description=_command_help(click_command),
            surfaces=("cli",),
            notes="",
        )

    for entry in _TUI_COMMAND_MANIFEST:
        existing = by_command.get(entry.command)
        if existing is None:
            by_command[entry.command] = CommandContract(
                command=entry.command,
                aliases=entry.aliases,
                description=entry.description,
                surfaces=("tui",),
                notes=entry.notes,
            )
            continue

        merged_aliases = tuple(dict.fromkeys([*existing.aliases, *entry.aliases]))
        merged_surfaces = tuple(dict.fromkeys([*existing.surfaces, "tui"]))
        description = entry.description or existing.description
        notes = entry.notes or existing.notes
        by_command[entry.command] = CommandContract(
            command=entry.command,
            aliases=merged_aliases,
            description=description,
            surfaces=merged_surfaces,
            notes=notes,
        )

    contracts = sorted(by_command.values(), key=lambda item: item.command)
    return tuple(contracts)


def _resolve_cli_app() -> Any:
    from agent_recall.cli.main import app

    return app


def _walk_registered_cli_commands(cli_app: Any) -> list[tuple[str, Any]]:
    root = get_typer_command(cli_app)
    seen: set[str] = set()
    discovered: list[tuple[str, Any]] = []

    def _walk(command: Any, prefix: str = "") -> None:
        mapping = getattr(command, "commands", None)
        if not isinstance(mapping, dict):
            return

        for name in sorted(mapping):
            child = mapping[name]
            if getattr(child, "hidden", False):
                continue
            full = f"{prefix} {name}".strip()
            if full in seen:
                continue
            seen.add(full)
            discovered.append((full, child))
            _walk(child, full)

    _walk(root)
    return discovered


def _command_help(click_command: Any) -> str:
    for attr_name in ("help", "short_help"):
        value = getattr(click_command, attr_name, None)
        if isinstance(value, str) and value.strip():
            return _first_help_line(value)
    return ""


def _first_help_line(text: str) -> str:
    line = next(iter(_non_empty_lines(text)), "")
    return line


def _non_empty_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            yield line
