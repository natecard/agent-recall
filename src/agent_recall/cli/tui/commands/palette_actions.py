from __future__ import annotations

import shlex
from dataclasses import dataclass

from agent_recall.cli.command_contract import get_command_contract


def _build_command_suggestions(cli_commands: list[str]) -> list[str]:
    base = ["help", "run", "sync --no-compact", "settings", "preferences", "quit"]
    for contract in get_command_contract():
        if "tui" not in contract.surfaces:
            continue
        base.append(contract.command)
        base.extend(contract.aliases)
    base.extend(
        [
            "config setup --force",
            "config setup --quick",
            "config model --provider ollama --model llama3.1",
            "config model --temperature 0.2 --max-tokens 8192",
            "view overview",
            "view sources",
            "view llm",
            "view knowledge",
            "view settings",
            "view console",
            "view all",
            "menu overview",
        ]
    )

    suggestions: list[str] = []
    seen: set[str] = set()

    for value in [*base, *cli_commands]:
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned.startswith("/"):
            cleaned = cleaned[1:].strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        suggestions.append(cleaned)

    return suggestions


def _normalize_palette_command(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("/"):
        cleaned = cleaned[1:].strip()
    return cleaned


def _is_palette_cli_command_redundant(command: str) -> bool:
    normalized = _normalize_palette_command(command)
    if not normalized:
        return True

    exact = {
        "open",
        "status",
        "run",
        "sync",
        "compact",
        "sources",
        "sessions",
        "settings",
        "preferences",
        "setup",
        "model",
        "theme",
    }
    if normalized in exact:
        return True

    prefixes = (
        "view ",
        "menu ",
        "theme ",
        "ralph ",
        "config setup",
        "config model",
        "config settings",
        "config preferences",
    )
    return normalized.startswith(prefixes)


def _is_knowledge_run_command(command: str) -> bool:
    value = command.strip()
    if not value:
        return False
    try:
        parts = shlex.split(value)
    except ValueError:
        return False
    if not parts:
        return False
    action = parts[0].lower()
    if action in {"run", "compact"}:
        return True
    if action == "sync" and "--no-compact" not in parts:
        return True
    return False


@dataclass(frozen=True)
class PaletteAction:
    action_id: str
    title: str
    description: str
    group: str
    shortcut: str = ""
    binding: str = ""
    keywords: str = ""


def get_palette_actions() -> list[PaletteAction]:
    return [
        PaletteAction(
            "setup",
            "Setup",
            "Configure sources and model defaults for this repo",
            "Core",
            shortcut="configure repository setup",
            keywords="onboarding setup",
        ),
        PaletteAction(
            "knowledge-run",
            "Run Knowledge Update",
            "Ingest conversations and synthesize GUARDRAILS, STYLE, and RECENT",
            "Core",
            shortcut="ingest + synthesize",
            binding="Ctrl+K",
            keywords="run compact synthesis llm",
        ),
        PaletteAction(
            "run:select",
            "Run Selected Conversations",
            "Choose specific conversations for a targeted knowledge update",
            "Core",
            shortcut="targeted run",
            keywords="sessions select run llm",
        ),
        PaletteAction(
            "sync",
            "Sync Conversations",
            "Ingest from enabled sources without running synthesis",
            "Core",
            shortcut="ingest only",
            binding="Ctrl+Y",
            keywords="sync ingest",
        ),
        PaletteAction(
            "status",
            "Refresh Dashboard",
            "Reload all dashboard panels now",
            "Core",
            shortcut="refresh now",
            binding="Ctrl+R",
            keywords="status dashboard refresh",
        ),
        PaletteAction(
            "ralph-enable",
            "Start Ralph Loop",
            "Enable Ralph loop automation for this repository",
            "Core",
            shortcut="enable loop",
            keywords="ralph enable start loop",
        ),
        PaletteAction(
            "ralph-disable",
            "Stop Ralph Loop",
            "Disable Ralph loop automation for this repository",
            "Core",
            shortcut="disable loop",
            keywords="ralph disable stop loop",
        ),
        PaletteAction(
            "ralph-status",
            "Ralph Loop Status",
            "Show current Ralph status and last run outcome",
            "Core",
            shortcut="loop status",
            keywords="ralph status last run outcome",
        ),
        PaletteAction(
            "ralph-select",
            "Select PRD Items",
            "Choose which PRD items to include in the Ralph loop",
            "Core",
            shortcut="ralph select",
            keywords="ralph prd select items",
        ),
        PaletteAction(
            "ralph-run",
            "Run Ralph Loop",
            "Execute the Ralph loop and stream CLI output to console",
            "Core",
            shortcut="ralph run",
            keywords="ralph run execute loop agent",
        ),
        PaletteAction(
            "sources",
            "Source Health",
            "Check source availability and discovered conversation counts",
            "Sessions",
            shortcut="availability + counts",
            keywords="sources cursor claude",
        ),
        PaletteAction(
            "theme",
            "Theme",
            "Switch themes instantly with arrows and Enter",
            "Sessions",
            shortcut="preview + apply",
            binding="Ctrl+T",
            keywords="theme list set",
        ),
        PaletteAction(
            "sessions",
            "Conversations",
            "Browse discovered conversations for this repository",
            "Sessions",
            shortcut="browse conversation list",
            keywords="sessions conversations history",
        ),
        PaletteAction(
            "view:overview",
            "Overview",
            "High-level repository status and health",
            "Views",
            keywords="view overview",
        ),
        PaletteAction(
            "view:sources",
            "Sources View",
            "Source connectivity and ingestion status",
            "Views",
            keywords="view sources",
        ),
        PaletteAction(
            "view:llm",
            "LLM View",
            "Provider, model, and synthesis configuration",
            "Views",
            keywords="view llm",
        ),
        PaletteAction(
            "view:knowledge",
            "Knowledge View",
            "Knowledge base artifacts and indexed chunks",
            "Views",
            keywords="view knowledge",
        ),
        PaletteAction(
            "view:settings",
            "Settings View",
            "Runtime and interface settings",
            "Views",
            keywords="view settings",
        ),
        PaletteAction(
            "view:timeline",
            "Timeline View",
            "Iteration outcomes and summaries",
            "Views",
            keywords="view timeline iterations",
        ),
        PaletteAction(
            "view:console",
            "Console View",
            "Recent command output and activity history",
            "Views",
            keywords="view console",
        ),
        PaletteAction(
            "view:all",
            "All Views",
            "Show all dashboard panels together",
            "Views",
            keywords="view all",
        ),
        PaletteAction(
            "model",
            "Model Preferences",
            "Adjust provider, model, base URL, and generation defaults",
            "Settings",
            shortcut="provider + model config",
            keywords="provider model temperature max tokens",
        ),
        PaletteAction(
            "settings",
            "Workspace Preferences",
            "Change default view, refresh speed, and workspace scope",
            "Settings",
            shortcut="refresh/view/workspaces",
            binding="Ctrl+G",
            keywords="settings preferences",
        ),
        PaletteAction(
            "ralph-config",
            "Ralph Configuration",
            "Configure Ralph loop agent, model, and behavior",
            "Settings",
            shortcut="ralph config",
            keywords="ralph config setup agent coding cli model",
        ),
        PaletteAction(
            "ralph-hooks-install",
            "Install Claude Hooks",
            "Install Claude Code hooks for Ralph guardrails",
            "Settings",
            shortcut="ralph hooks install",
            keywords="ralph hooks install claude settings",
        ),
        PaletteAction(
            "ralph-hooks-uninstall",
            "Remove Claude Hooks",
            "Remove Claude Code hooks installed by Ralph",
            "Settings",
            shortcut="ralph hooks uninstall",
            keywords="ralph hooks uninstall claude settings",
        ),
        PaletteAction(
            "ralph-opencode-install",
            "Install OpenCode Plugin",
            "Install OpenCode plugin for Ralph session events",
            "Settings",
            shortcut="ralph plugin opencode-install",
            keywords="ralph plugin opencode install",
        ),
        PaletteAction(
            "ralph-opencode-uninstall",
            "Remove OpenCode Plugin",
            "Remove OpenCode plugin for Ralph session events",
            "Settings",
            shortcut="ralph plugin opencode-uninstall",
            keywords="ralph plugin opencode uninstall",
        ),
        PaletteAction(
            "ralph-watch",
            "Watch Claude Logs",
            "Stream Claude Code JSONL events in the activity panel",
            "Settings",
            shortcut="ralph watch",
            keywords="ralph watch claude logs live",
        ),
        PaletteAction(
            "ralph-view-diff",
            "View Last Diff",
            "Review the most recent Ralph iteration diff",
            "Settings",
            shortcut="ralph view-diff",
            keywords="ralph diff viewer iteration changes",
        ),
        PaletteAction(
            "ralph-notifications",
            "Toggle Notifications",
            "Enable or disable Ralph desktop notifications",
            "Settings",
            shortcut="ralph notifications",
            keywords="ralph notify notifications alert",
        ),
        PaletteAction(
            "ralph-terminal",
            "Toggle Terminal Panel",
            "Show or hide the embedded terminal panel",
            "Settings",
            shortcut="ralph terminal",
            keywords="ralph terminal panel toggle",
        ),
        PaletteAction(
            "quit",
            "Quit",
            "Exit the TUI",
            "System",
            binding="Ctrl+Q",
            keywords="quit exit",
        ),
    ]
