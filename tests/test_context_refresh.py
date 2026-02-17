"""Unit tests for ContextRefreshHook."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_recall.ralph.context_refresh import ContextRefreshHook
from agent_recall.storage.files import FileStorage
from agent_recall.storage.sqlite import SQLiteStorage


def _make_agent_dir(tmp_path: Path) -> Path:
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "RULES.md").write_text("# Rules\n- Keep changes small.\n")
    (agent_dir / "GUARDRAILS.md").write_text("# Guardrails\n")
    (agent_dir / "STYLE.md").write_text("# Style\n")
    (agent_dir / "RECENT.md").write_text("# Recent\n")
    (agent_dir / "config.yaml").write_text("llm:\n  provider: openai\n  model: gpt-4o-mini\n")
    return agent_dir


def test_context_refresh_hook_task_format_item_and_iteration(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={"codex": agent_dir / "codex" / "context.json"},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh(item_id="AR-001", iteration=3)

    assert result["task"] == "[Ralph Iteration 3] AR-001: Continue work"
    assert "context_length" in result
    assert "adapters_written" in result
    assert "refreshed_at" in result
    mock_write.assert_called_once()
    call_kwargs = mock_write.call_args[1]
    assert call_kwargs["task"] == "[Ralph Iteration 3] AR-001: Continue work"
    assert call_kwargs["active_session_id"] == "ralph-iteration-3"


def test_context_refresh_hook_task_format_item_only(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh(item_id="AR-002")

    assert result["task"] == "AR-002: Continue work"
    call_kwargs = mock_write.call_args[1]
    assert call_kwargs["active_session_id"] is None


def test_context_refresh_hook_task_format_iteration_only(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ):
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh(iteration=5)

    assert result["task"] == "[Ralph Iteration 5] Continue work"


def test_context_refresh_hook_task_format_explicit_task(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ):
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh(task="Implement JWT auth")

    assert result["task"] == "Implement JWT auth"


def test_context_refresh_hook_task_format_item_iteration_and_task(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ):
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh(task="Build feature", item_id="AR-003", iteration=2)

    assert result["task"] == "[Ralph Iteration 2] AR-003: Build feature"


def test_context_refresh_hook_return_dict_structure(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={"codex": Path("x"), "cursor": Path("y")},
    ):
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "x" * 100

        result = hook.refresh()

    assert "context_length" in result
    assert result["context_length"] == 100
    assert "adapters_written" in result
    adapters = result["adapters_written"]
    assert isinstance(adapters, list)
    assert set(adapters) == {"codex", "cursor"}
    assert "refreshed_at" in result
    assert "task" in result


def test_context_refresh_hook_refresh_for_prd_item_extracts_and_delegates(
    tmp_path: Path,
) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh_for_prd_item(
            {
                "id": "AR-010",
                "title": "Add auth",
                "description": "JWT implementation",
            },
            iteration=1,
        )

    assert result["task"] == "[Ralph Iteration 1] AR-010: Add auth - JWT implementation"
    call_kwargs = mock_write.call_args[1]
    assert call_kwargs["task"] == result["task"]


def test_context_refresh_hook_refresh_for_prd_item_title_only(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ):
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        result = hook.refresh_for_prd_item(
            {"id": "AR-011", "title": "Minimal item"},
        )

    assert result["task"] == "AR-011: Minimal item"


def test_context_refresh_hook_forward_token_budget(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.assembler = MagicMock()
        hook.assembler.assemble.return_value = "context"

        hook.refresh(token_budget=500)

    call_kwargs = mock_write.call_args[1]
    assert call_kwargs["token_budget"] == 500


def test_context_refresh_reads_latest_tier_file_state(tmp_path: Path) -> None:
    """Refresh uses current tier file content; no stale cache."""
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)
    guardrails_path = agent_dir / "GUARDRAILS.md"

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)

        guardrails_path.write_text("# Guardrails\n\n- initial")
        hook.refresh()
        ctx1 = mock_write.call_args[1]["context"]

        guardrails_path.write_text("# Guardrails\n\n- updated-after-compaction")
        hook.refresh()
        ctx2 = mock_write.call_args[1]["context"]

    assert "initial" in ctx1
    assert "updated-after-compaction" not in ctx1
    assert "updated-after-compaction" in ctx2
    assert "initial" not in ctx2


def test_context_refresh_includes_rules_file(tmp_path: Path) -> None:
    agent_dir = _make_agent_dir(tmp_path)
    storage = SQLiteStorage(agent_dir / "state.db")
    files = FileStorage(agent_dir)
    (agent_dir / "RULES.md").write_text("# Rules\n- Always run local checks.\n")

    with patch(
        "agent_recall.ralph.context_refresh.write_adapter_payloads",
        return_value={},
    ) as mock_write:
        hook = ContextRefreshHook(agent_dir, storage, files)
        hook.refresh()

    context = mock_write.call_args[1]["context"]
    assert "## Rules" in context
    assert "Always run local checks." in context
