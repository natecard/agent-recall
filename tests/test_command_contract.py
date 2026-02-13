from __future__ import annotations

from agent_recall.cli.command_contract import command_parity_report


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
