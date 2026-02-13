from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandContract:
    command: str
    aliases: tuple[str, ...]
    description: str
    surfaces: tuple[str, ...]
    notes: str = ""


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
            expected_cli.update(contract.aliases)
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


def get_command_contract() -> list[CommandContract]:
    return [
        CommandContract(
            command="help",
            aliases=(),
            description="Show slash command help",
            surfaces=("tui",),
        ),
        CommandContract(
            command="status",
            aliases=(),
            description="Show repository status and source availability",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="sources",
            aliases=(),
            description="List detected conversation sources",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="sessions",
            aliases=(),
            description="List discovered conversations",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="sync",
            aliases=("run",),
            description="Ingest conversations (optionally skip compaction)",
            surfaces=("cli", "tui"),
            notes="TUI palette has dedicated run alias for sync + compact.",
        ),
        CommandContract(
            command="view",
            aliases=("menu",),
            description="Switch TUI dashboard views",
            surfaces=("tui",),
        ),
        CommandContract(
            command="settings",
            aliases=("preferences",),
            description="Open the TUI settings view",
            surfaces=("tui",),
        ),
        CommandContract(
            command="compact",
            aliases=(),
            description="Synthesize knowledge from recent logs",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="compact-tiers",
            aliases=(),
            description="Compact tier files (GUARDRAILS/STYLE/RECENT)",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="open",
            aliases=("tui",),
            description="Open the TUI dashboard",
            surfaces=("cli",),
            notes="TUI already running; /open is blocked in slash commands.",
        ),
        CommandContract(
            command="command-inventory",
            aliases=(),
            description="Print command inventory for CLI, TUI, and palette",
            surfaces=("cli",),
        ),
        CommandContract(
            command="config setup",
            aliases=(),
            description="Run repository setup",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="config model",
            aliases=("config-llm",),
            description="Configure provider and model defaults",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="config settings",
            aliases=("config preferences",),
            description="Configure workspace settings",
            surfaces=("tui",),
            notes="Config settings command is not yet available in CLI.",
        ),
        CommandContract(
            command="theme list",
            aliases=(),
            description="List available themes",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="theme show",
            aliases=(),
            description="Show active theme",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="theme set",
            aliases=(),
            description="Apply a theme by name",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="ralph status",
            aliases=(),
            description="Show Ralph loop status",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="ralph enable",
            aliases=(),
            description="Enable Ralph loop",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="ralph disable",
            aliases=(),
            description="Disable Ralph loop",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="refresh-context",
            aliases=(),
            description="Refresh context bundle outputs",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="sync-background",
            aliases=(),
            description="Run background sync mode",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="write-guardrails",
            aliases=("write-guardrails-failure",),
            description="Append guardrails entries",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="write-style",
            aliases=(),
            description="Append style entries",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="write-recent",
            aliases=(),
            description="Append recent session summary entries",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="lint-tiers",
            aliases=(),
            description="Lint tier files for structure issues",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="tier-stats",
            aliases=(),
            description="Show tier file statistics",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="reset-sync",
            aliases=(),
            description="Reset sync markers",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="quit",
            aliases=("exit", "q"),
            description="Exit the TUI",
            surfaces=("tui",),
        ),
    ]
