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
            "Settings",
            shortcut="configure repository setup",
            keywords="onboarding setup",
        ),
        PaletteAction(
            "knowledge-run",
            "Run Knowledge Update",
            "Ingest conversations and synthesize GUARDRAILS, STYLE, and RECENT",
            "Memory",
            shortcut="ingest + synthesize",
            binding="Ctrl+K",
            keywords="run compact synthesis llm",
        ),
        PaletteAction(
            "run:select",
            "Run Selected Conversations",
            "Choose specific conversations for a targeted knowledge update",
            "Memory",
            shortcut="targeted run",
            keywords="sessions select run llm",
        ),
        PaletteAction(
            "sync",
            "Sync Conversations",
            "Ingest from enabled sources without running synthesis",
            "Memory",
            shortcut="ingest only",
            binding="Ctrl+Y",
            keywords="sync ingest",
        ),
        PaletteAction(
            "status",
            "Refresh Dashboard",
            "Reload all dashboard panels now",
            "Dashboard",
            shortcut="refresh now",
            binding="Ctrl+R",
            keywords="status dashboard refresh",
        ),
        PaletteAction(
            "ralph-enable",
            "Start Ralph Loop",
            "Enable Ralph loop automation for this repository",
            "Ralph",
            shortcut="enable loop",
            keywords="ralph enable start loop",
        ),
        PaletteAction(
            "ralph-disable",
            "Stop Ralph Loop",
            "Disable Ralph loop automation for this repository",
            "Ralph",
            shortcut="disable loop",
            keywords="ralph disable stop loop",
        ),
        PaletteAction(
            "ralph-status",
            "Ralph Loop Status",
            "Show current Ralph status and last run outcome",
            "Ralph",
            shortcut="loop status",
            keywords="ralph status last run outcome",
        ),
        PaletteAction(
            "ralph-select",
            "Select PRD Items",
            "Choose which PRD items to include in the Ralph loop",
            "Ralph",
            shortcut="ralph select",
            keywords="ralph prd select items",
        ),
        PaletteAction(
            "ralph-run",
            "Run Ralph Loop",
            "Execute the Ralph loop and stream CLI output to console",
            "Ralph",
            shortcut="ralph run",
            keywords="ralph run execute loop agent",
        ),
        PaletteAction(
            "theme",
            "Theme",
            "Switch themes instantly with arrows and Enter",
            "Settings",
            shortcut="preview + apply",
            binding="Ctrl+T",
            keywords="theme list set",
        ),
        PaletteAction(
            "sessions",
            "View Sessions & Sources",
            "Browse discovered conversations for this repository",
            "Memory",
            shortcut="browse conversation list",
            keywords="sessions conversations history sources",
        ),
        PaletteAction(
            "view-select",
            "Change View",
            "Switch the active dashboard view or display all panels",
            "Settings",
            shortcut="change view",
            binding="Ctrl+V",
            keywords="view layout switch panel",
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
            "layout",
            "Customise Layout",
            "Toggle visible dashboard widgets and banner size",
            "Settings",
            shortcut="layout widgets banner",
            binding="Ctrl+L",
            keywords="layout widgets banner",
        ),
        PaletteAction(
            "ralph-config",
            "Ralph Configuration",
            "Configure Ralph loop agent, model, and behavior",
            "Settings",
            shortcut="ralph config",
            keywords="ralph config setup agent coding cli model plugins hooks",
        ),
        PaletteAction(
            "ralph-view-diff",
            "Iteration History",
            "Browse Ralph iteration diffs with side-by-side viewer and navigation",
            "Ralph",
            shortcut="ralph view-diff",
            keywords="ralph diff history iteration changes side-by-side navigate",
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
