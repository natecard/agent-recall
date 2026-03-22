from __future__ import annotations

from agent_recall.cli.tui.logic.select_compat import select_empty_value
from agent_recall.cli.tui.ui.modals.coding_agent_step import _normalize_cli_value


def test_normalize_cli_value_rejects_non_string_values() -> None:
    assert _normalize_cli_value(False) == ""
    assert _normalize_cli_value(None) == ""
    assert _normalize_cli_value(select_empty_value()) == ""


def test_normalize_cli_value_accepts_known_cli_options() -> None:
    assert _normalize_cli_value("codex") == "codex"
    assert _normalize_cli_value("  OPENCODE ") == "opencode"
    assert _normalize_cli_value("claude-code") == "claude-code"
    assert _normalize_cli_value("unknown") == ""
