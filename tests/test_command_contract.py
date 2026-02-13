from __future__ import annotations

import agent_recall.cli.main as cli_main
from agent_recall.cli.command_contract import command_parity_report
from agent_recall.cli.textual_tui import _build_command_suggestions


def test_command_parity_report_separates_surfaces() -> None:
    report = command_parity_report(
        cli_commands={"status", "sources", "sync"},
        tui_commands={"status", "sources", "sync", "settings"},
    )
    assert "settings" not in report["missing_in_cli"]
    assert "help" in report["missing_in_tui"]
    assert report["extra_in_cli"] == set()


def test_command_parity_report_flags_untracked_surfaces() -> None:
    report = command_parity_report(
        cli_commands={"status", "sources", "sync", "mystery"},
        tui_commands={"status", "sources", "sync"},
    )
    assert report["extra_in_cli"] == {"mystery"}


def test_command_contract_tui_commands_present_in_suggestions() -> None:
    cli_commands = cli_main._collect_cli_commands_for_palette()
    suggestions = set(_build_command_suggestions(cli_commands))

    report = command_parity_report(
        cli_commands=set(cli_commands),
        tui_commands=suggestions,
    )

    assert report["missing_in_cli"] == set()
    assert report["missing_in_tui"] == set()
