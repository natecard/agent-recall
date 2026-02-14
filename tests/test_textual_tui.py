from __future__ import annotations

from agent_recall.cli.textual_tui import (
    _build_command_suggestions,
    _clean_optional_text,
    _is_knowledge_run_command,
    _is_palette_cli_command_redundant,
    get_palette_actions,
)


def test_build_command_suggestions_excludes_slash_variants() -> None:
    suggestions = _build_command_suggestions(["status", "config setup", "sync"])

    assert "status" in suggestions
    assert "config setup" in suggestions
    assert "sync" in suggestions
    assert "settings" in suggestions
    assert "config settings" in suggestions
    assert "config model" in suggestions
    assert "ralph enable" in suggestions
    assert "ralph disable" in suggestions
    assert "ralph status" in suggestions


def test_palette_contains_ralph_loop_controls() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-enable" in action_ids
    assert "ralph-disable" in action_ids
    assert "ralph-status" in action_ids


def test_palette_contains_ralph_config_action() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-config" in action_ids


def test_palette_contains_ralph_run_action() -> None:
    action_ids = {action.action_id for action in get_palette_actions()}
    assert "ralph-run" in action_ids


def test_clean_optional_text_handles_none_variants() -> None:
    assert _clean_optional_text(None) == ""
    assert _clean_optional_text("None") == ""
    assert _clean_optional_text("null") == ""
    assert _clean_optional_text(" value ") == "value"


def test_palette_cli_command_redundancy_filter() -> None:
    assert _is_palette_cli_command_redundant("open") is True
    assert _is_palette_cli_command_redundant("status") is True
    assert _is_palette_cli_command_redundant("run") is True
    assert _is_palette_cli_command_redundant("sources") is True
    assert _is_palette_cli_command_redundant("sessions") is True
    assert _is_palette_cli_command_redundant("theme list") is True
    assert _is_palette_cli_command_redundant("theme show") is False
    assert _is_palette_cli_command_redundant("config model") is True
    assert _is_palette_cli_command_redundant("providers") is False


def test_is_knowledge_run_command() -> None:
    assert _is_knowledge_run_command("run")
    assert _is_knowledge_run_command("compact")
    assert _is_knowledge_run_command("sync")
    assert _is_knowledge_run_command("sync --source cursor")
    assert not _is_knowledge_run_command("sync --no-compact")
    assert not _is_knowledge_run_command("status")
