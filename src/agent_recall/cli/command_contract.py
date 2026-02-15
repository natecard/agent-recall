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
            command="init",
            aliases=(),
            description="Initialize agent memory in the current repository",
            surfaces=("cli",),
        ),
        CommandContract(
            command="splash",
            aliases=(),
            description="Display the Agent Recall splash banner",
            surfaces=("cli",),
        ),
        CommandContract(
            command="start",
            aliases=(),
            description="Start a new session and output context",
            surfaces=("cli",),
        ),
        CommandContract(
            command="log",
            aliases=(),
            description="Log an observation or learning",
            surfaces=("cli",),
        ),
        CommandContract(
            command="end",
            aliases=(),
            description="End the current session",
            surfaces=("cli",),
        ),
        CommandContract(
            command="context",
            aliases=(),
            description="Output current context bundle",
            surfaces=("cli",),
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
            aliases=(),
            description="Ingest conversations (optionally skip compaction)",
            surfaces=("cli", "tui"),
            notes="TUI palette has dedicated run alias for sync + compact.",
        ),
        CommandContract(
            command="run",
            aliases=(),
            description="Alias for sync (includes synthesis by default)",
            surfaces=("tui",),
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
            command="retrieve",
            aliases=(),
            description="Retrieve relevant memory chunks",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ingest",
            aliases=(),
            description="Ingest a JSONL transcript into log entries",
            surfaces=("cli",),
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
            command="providers",
            aliases=(),
            description="List available LLM providers",
            surfaces=("cli",),
        ),
        CommandContract(
            command="test-llm",
            aliases=(),
            description="Test the configured LLM provider",
            surfaces=("cli",),
        ),
        CommandContract(
            command="config",
            aliases=(),
            description="Manage onboarding and model configuration",
            surfaces=("cli",),
        ),
        CommandContract(
            command="curation",
            aliases=(),
            description="Review and approve extracted learnings",
            surfaces=("cli",),
        ),
        CommandContract(
            command="curation list",
            aliases=(),
            description="List log entries by curation status",
            surfaces=("cli",),
        ),
        CommandContract(
            command="curation approve",
            aliases=(),
            description="Approve a pending log entry",
            surfaces=("cli",),
        ),
        CommandContract(
            command="curation reject",
            aliases=(),
            description="Reject a pending log entry",
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
            aliases=(),
            description="Configure provider and model defaults",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="config adapters",
            aliases=(),
            description="Configure context adapter payloads",
            surfaces=("cli",),
        ),
        CommandContract(
            command="config settings",
            aliases=("config preferences",),
            description="Configure workspace settings",
            surfaces=("tui",),
            notes="Config settings command is not yet available in CLI.",
        ),
        CommandContract(
            command="theme",
            aliases=(),
            description="Manage CLI themes",
            surfaces=("cli",),
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
            command="ralph",
            aliases=(),
            description="Manage Ralph loop configuration",
            surfaces=("cli",),
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
            command="ralph run",
            aliases=(),
            description="Run Ralph loop (python or bash)",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph watch",
            aliases=(),
            description="Watch Claude Code logs and emit live activity",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph select",
            aliases=(),
            description="Choose which PRD items to include in the Ralph loop",
            surfaces=("tui",),
        ),
        CommandContract(
            command="ralph set-prds",
            aliases=(),
            description="Set selected PRD IDs via CLI (e.g. --prds AR-001,AR-002)",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph set-agent",
            aliases=(),
            description="Set coding CLI and model for Ralph loop",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph hooks install",
            aliases=(),
            description="Install Claude Code hooks for Ralph guardrails",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph hooks uninstall",
            aliases=(),
            description="Remove Claude Code hooks installed by Ralph",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph create-report",
            aliases=(),
            description="Create iteration report (current.json)",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph finalize-report",
            aliases=(),
            description="Finalize and archive current report",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph extract-iteration",
            aliases=(),
            description="Extract heuristic artifacts for current report",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph view-diff",
            aliases=(),
            description="View the latest Ralph iteration diff",
            surfaces=("cli", "tui"),
        ),
        CommandContract(
            command="ralph rebuild-forecast",
            aliases=(),
            description="Rebuild RECENT.md forecast from iteration reports",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph synthesize-climate",
            aliases=(),
            description="Synthesize GUARDRAILS.md and STYLE.md from iterations",
            surfaces=("cli",),
        ),
        CommandContract(
            command="ralph config",
            aliases=(),
            description="Open Ralph configuration modal",
            surfaces=("tui",),
        ),
        CommandContract(
            command="ralph notify",
            aliases=("ralph notifications",),
            description="Send a Ralph notification on supported platforms",
            surfaces=("cli",),
        ),
        CommandContract(
            command="refresh-context",
            aliases=(),
            description="Refresh context bundle outputs",
            surfaces=("cli", "tui"),
            notes="Optional adapter payloads can be generated for supported agents.",
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
