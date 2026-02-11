from __future__ import annotations

from agent_recall.cli.textual_tui import (
    _build_command_suggestions,
    _clean_optional_text,
    _is_palette_cli_command_redundant,
)


def test_build_command_suggestions_excludes_slash_variants() -> None:
    suggestions = _build_command_suggestions(["status", "config setup", "sync"])

    assert "status" in suggestions
    assert "config setup" in suggestions
    assert "sync" in suggestions
    assert "settings" in suggestions
    assert "config settings" in suggestions


def test_clean_optional_text_handles_none_variants() -> None:
    assert _clean_optional_text(None) == ""
    assert _clean_optional_text("None") == ""
    assert _clean_optional_text("null") == ""
    assert _clean_optional_text(" value ") == "value"


def test_palette_cli_command_redundancy_filter() -> None:
    assert _is_palette_cli_command_redundant("status") is True
    assert _is_palette_cli_command_redundant("sources") is True
    assert _is_palette_cli_command_redundant("sessions") is True
    assert _is_palette_cli_command_redundant("theme list") is True
    assert _is_palette_cli_command_redundant("theme show") is False
    assert _is_palette_cli_command_redundant("config model") is True
    assert _is_palette_cli_command_redundant("providers") is False
