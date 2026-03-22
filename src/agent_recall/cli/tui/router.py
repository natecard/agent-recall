from __future__ import annotations

import os
import re
import shlex
import sys
import time
from collections.abc import Callable

from rich.markup import escape
from typer.testing import CliRunner

_ansi_escape_pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_box_drawing_chars = set("│┃─━┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿")
_box_drawing_translation = str.maketrans(
    {
        "│": " ",
        "┃": " ",
        "─": " ",
        "━": " ",
        "┄": " ",
        "┅": " ",
        "┆": " ",
        "┇": " ",
        "┈": " ",
        "┉": " ",
        "┊": " ",
        "┋": " ",
        "┌": " ",
        "┍": " ",
        "┎": " ",
        "┏": " ",
        "┐": " ",
        "┑": " ",
        "┒": " ",
        "┓": " ",
        "└": " ",
        "┕": " ",
        "┖": " ",
        "┗": " ",
        "┘": " ",
        "┙": " ",
        "┚": " ",
        "┛": " ",
        "├": " ",
        "┝": " ",
        "┞": " ",
        "┟": " ",
        "┠": " ",
        "┡": " ",
        "┢": " ",
        "┣": " ",
        "┤": " ",
        "┥": " ",
        "┦": " ",
        "┧": " ",
        "┨": " ",
        "┩": " ",
        "┪": " ",
        "┫": " ",
        "┬": " ",
        "┭": " ",
        "┮": " ",
        "┯": " ",
        "┰": " ",
        "┱": " ",
        "┲": " ",
        "┳": " ",
        "┴": " ",
        "┵": " ",
        "┶": " ",
        "┷": " ",
        "┸": " ",
        "┹": " ",
        "┺": " ",
        "┻": " ",
        "┼": " ",
        "┽": " ",
        "┾": " ",
        "┿": " ",
    }
)


def normalize_tui_command(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    return text if text.startswith("/") else f"/{text}"


def read_tui_command(timeout_seconds: float) -> str | None:
    if not sys.stdin.isatty():
        time.sleep(timeout_seconds)
        return None

    if os.name == "nt":
        time.sleep(timeout_seconds)
        return None

    try:
        import select

        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        time.sleep(timeout_seconds)
        return None

    if not ready:
        return None

    line = sys.stdin.readline()
    if line == "":
        return "/quit"
    return line.rstrip("\n")


def normalize_tui_output_line(line: str) -> str:
    without_ansi = _ansi_escape_pattern.sub("", line)
    without_box_chars = without_ansi.translate(_box_drawing_translation)
    return " ".join(without_box_chars.split())


def strip_ansi_control_sequences(line: str) -> str:
    return _ansi_escape_pattern.sub("", line)


def looks_like_table_output(lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    table_like_rows = 0
    for line in lines:
        if any(char in _box_drawing_chars for char in line):
            table_like_rows += 1
    return table_like_rows >= 2


def execute_tui_slash_command(
    raw: str,
    *,
    app,
    slash_runner: CliRunner,
    get_help_lines: Callable[[], list[str]],
    run_onboarding_setup: Callable[[bool, bool], None],
    terminal_width: int | None = None,
    terminal_height: int | None = None,
) -> tuple[bool, list[str]]:
    value = normalize_tui_command(raw)
    if not value:
        return False, []

    if not value.startswith("/"):
        return False, ["[warning]Commands must start with '/'. Try /help.[/warning]"]

    command_text = value[1:].strip()
    if not command_text:
        return False, ["[warning]Empty command. Try /help.[/warning]"]

    try:
        parts = shlex.split(command_text)
    except ValueError as exc:
        return False, [f"[error]Invalid command: {escape(str(exc))}[/error]"]

    original_parts = list(parts)
    command_name = parts[0].lower()
    if command_name in {"q", "quit", "exit"}:
        return True, ["[dim]Leaving TUI...[/dim]"]

    if command_name in {"help", "h", "?"}:
        return False, get_help_lines()

    if command_name == "settings":
        return False, [
            "[success]✓ Switched to settings view[/success]",
            "[dim]Use /view settings[/dim]",
        ]

    if command_name == "config" and len(parts) >= 2 and parts[1].lower() == "setup":
        force = "--force" in parts or "-f" in parts
        quick = "--quick" in parts
        run_onboarding_setup(force, quick)
        return False, ["[success]✓ Setup flow completed[/success]"]

    if (
        command_name == "ralph"
        and len(parts) >= 2
        and parts[1].lower()
        in {
            "enable",
            "disable",
            "status",
            "select",
            "set-prds",
            "get-selected-prds",
        }
    ):
        command = ["ralph", *parts[1:]]
        result = slash_runner.invoke(app, command)
        lines = []
        if result.exit_code == 0:
            lines.append(f"[success]✓ /{escape(' '.join(parts))}[/success]")
        else:
            lines.append(f"[error]✗ /{escape(' '.join(parts))} (exit {result.exit_code})[/error]")
        output = result.output.strip()
        if output:
            for line in output.splitlines():
                if line.strip():
                    lines.append(f"[dim]{escape(line.strip())}[/dim]")
        return False, lines

    if command_name in {"tui", "open"}:
        return False, ["[warning]/open is already running.[/warning]"]

    if command_name == "run":
        parts = ["sync", "--compact", "--verbose", *parts[1:]]
        command_name = "sync"

    if command_name == "sync" and "--verbose" not in parts and "-v" not in parts:
        parts.append("--verbose")

    invoke_env: dict[str, str] | None = None
    if terminal_width is not None or terminal_height is not None:
        invoke_env = {}
        if terminal_width is not None and terminal_width > 0:
            invoke_env["COLUMNS"] = str(int(terminal_width))
        if terminal_height is not None and terminal_height > 0:
            invoke_env["LINES"] = str(int(terminal_height))

    runner_terminal_width = (
        terminal_width if terminal_width is not None and terminal_width > 0 else 120
    )
    result = slash_runner.invoke(
        app,
        parts,
        env=invoke_env,
        terminal_width=max(runner_terminal_width, 80),
    )
    command_label_parts = original_parts if original_parts and original_parts[0] == "run" else parts
    command_label = "/" + " ".join(command_label_parts)

    lines: list[str] = []
    if result.exit_code == 0:
        lines.append(f"[success]✓ {escape(command_label)}[/success]")
    else:
        lines.append(f"[error]✗ {escape(command_label)} (exit {result.exit_code})[/error]")

    output = result.output.strip()
    if output:
        raw_output_lines = output.splitlines()
        ansi_stripped_lines = [
            strip_ansi_control_sequences(raw_line).rstrip()
            for raw_line in raw_output_lines
            if raw_line.strip()
        ]
        if looks_like_table_output(ansi_stripped_lines):
            output_lines = ansi_stripped_lines
        else:
            meaningful_lines = []
            for raw_line in ansi_stripped_lines:
                normalized = normalize_tui_output_line(raw_line)
                if not normalized:
                    continue
                if not any(char.isalnum() for char in normalized):
                    continue
                meaningful_lines.append(normalized)
            output_lines = meaningful_lines if meaningful_lines else ansi_stripped_lines
        for line in output_lines:
            lines.append(f"[dim]{escape(line)}[/dim]")

    return False, lines


def handle_tui_view_command(raw: str, current_view: str) -> tuple[bool, str, list[str]]:
    value = normalize_tui_command(raw)
    if not value.startswith("/"):
        return False, current_view, []

    try:
        parts = shlex.split(value[1:].strip())
    except ValueError as exc:
        return True, current_view, [f"[error]Invalid command: {escape(str(exc))}[/error]"]

    if not parts:
        return False, current_view, []

    command = parts[0].lower()
    if command not in {"view", "menu"}:
        return False, current_view, []

    if len(parts) < 2:
        return (
            True,
            current_view,
            ["[warning]Usage: /view <overview|knowledge|settings|timeline|console|all>[/warning]"],
        )

    target = parts[1].strip().lower()
    valid = {"overview", "knowledge", "settings", "timeline", "console", "all"}
    if target not in valid:
        valid_text = ", ".join(sorted(valid))
        return (
            True,
            current_view,
            [f"[warning]Unknown view '{escape(target)}'. Valid: {valid_text}[/warning]"],
        )

    if target == current_view:
        return True, current_view, [f"[dim]Already on {target} view.[/dim]"]

    return True, target, [f"[success]✓ Switched to {target} view[/success]"]
