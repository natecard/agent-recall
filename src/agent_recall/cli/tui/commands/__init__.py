from agent_recall.cli.tui.commands.help_text import build_tui_help_lines
from agent_recall.cli.tui.commands.local_router import handle_local_command
from agent_recall.cli.tui.commands.palette_actions import (
    PaletteAction,
    _build_command_suggestions,
    _is_knowledge_run_command,
    _is_palette_cli_command_redundant,
    _normalize_palette_command,
    get_palette_actions,
)
from agent_recall.cli.tui.commands.palette_router import handle_palette_action

__all__ = [
    "PaletteAction",
    "_build_command_suggestions",
    "_is_knowledge_run_command",
    "_is_palette_cli_command_redundant",
    "_normalize_palette_command",
    "build_tui_help_lines",
    "get_palette_actions",
    "handle_local_command",
    "handle_palette_action",
]
