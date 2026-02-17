from __future__ import annotations

from agent_recall.cli.command_contract import get_command_contract


def build_tui_help_lines() -> list[str]:
    lines = ["[bold]Slash Commands[/bold]"]
    for contract in get_command_contract():
        if "tui" not in contract.surfaces:
            continue
        lines.append(f"[dim]/{contract.command}[/dim] - {contract.description}")
    lines.append(
        "[dim]/view overview|sources|llm|knowledge|settings|timeline|console|all[/dim]"
        " - switch TUI view"
    )
    lines.append("[dim]/run[/dim] - Alias for /sync (includes synthesis by default)")
    lines.append("[dim]/settings[/dim] - Open settings view")
    lines.append("[dim]/quit[/dim] - Exit the TUI")
    return lines
