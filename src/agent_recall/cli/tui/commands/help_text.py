from __future__ import annotations

from agent_recall.cli.command_contract import get_command_contract


def get_all_cli_commands() -> list[str]:
    """Return a list of all available CLI commands for autocomplete suggestions.

    Returns commands with their / prefix for slash command matching.
    """
    commands: list[str] = []

    # Add all TUI-facing commands from command contract
    for contract in get_command_contract():
        if "tui" in contract.surfaces:
            commands.append(f"/{contract.command}")
            for alias in contract.aliases:
                commands.append(f"/{alias}")

    # Add view commands
    views = [
        "overview",
        "sources",
        "llm",
        "knowledge",
        "settings",
        "timeline",
        "ralph",
        "console",
        "all",
    ]
    for view in views:
        commands.append(f"/view {view}")

    # Add additional common commands
    additional = [
        "/help",
        "/run",
        "/sync",
        "/sync --no-compact",
        "/settings",
        "/preferences",
        "/quit",
        "/refresh",
        "/menu overview",
        "/config setup --force",
        "/config setup --quick",
        "/config model --provider ollama --model llama3.1",
        "/config model --temperature 0.2 --max-tokens 8192",
    ]
    commands.extend(additional)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_commands: list[str] = []
    for cmd in commands:
        normalized = cmd.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_commands.append(cmd)

    return unique_commands


def filter_command_suggestions(
    input_value: str, commands: list[str], max_results: int = 8
) -> list[str]:
    """Filter commands based on input value.

    Performs case-insensitive prefix and substring matching.
    Prefix matches are prioritized over substring matches.
    """
    if not input_value or not input_value.startswith("/"):
        return []

    input_lower = input_value.lower().strip()
    prefix_matches: list[str] = []
    substring_matches: list[str] = []

    for cmd in commands:
        cmd_lower = cmd.lower()
        if cmd_lower.startswith(input_lower):
            prefix_matches.append(cmd)
        elif input_lower in cmd_lower:
            substring_matches.append(cmd)

    # Combine prefix matches first, then substring matches
    all_matches = prefix_matches + substring_matches
    return all_matches[:max_results]


def build_tui_help_lines() -> list[str]:
    lines = ["[bold]Slash Commands[/bold]"]
    for contract in get_command_contract():
        if "tui" not in contract.surfaces:
            continue
        lines.append(f"[dim]/{contract.command}[/dim] - {contract.description}")
    lines.append(
        "[dim]/view overview|sources|llm|knowledge|settings|timeline|ralph|console|all[/dim]"
        " - switch TUI view"
    )
    lines.append("[dim]/run[/dim] - Alias for /sync (includes synthesis by default)")
    lines.append("[dim]/settings[/dim] - Open settings view")
    lines.append("[dim]/quit[/dim] - Exit the TUI")
    return lines
