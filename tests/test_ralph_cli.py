from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.cli.ralph import build_agent_cmd_from_ralph_config

runner = CliRunner()


def test_cli_ralph_enable_creates_state_file() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "enable"])
        assert result.exit_code == 0
        state_path = Path(".agent") / "ralph" / "ralph_state.json"
        assert state_path.exists()
        payload = json.loads(state_path.read_text())
        assert payload["status"] == "ENABLED"


def test_cli_ralph_enable_with_prd_counts_items() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path("prd.json")
        prd_path.write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "AR-1", "passes": False},
                        {"id": "AR-2", "passes": False},
                    ]
                }
            )
        )
        result = runner.invoke(cli_main.app, ["ralph", "enable", "--prd", "prd.json"])
        assert result.exit_code == 0
        assert "2 PRD items" in result.output


def test_cli_ralph_enable_missing_prd_errors() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "enable", "--prd", "missing.json"])
        assert result.exit_code == 1
        assert "PRD file not found" in result.output


def test_cli_ralph_disable_updates_state() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["ralph", "enable"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "disable"])
        assert result.exit_code == 0
        state_path = Path(".agent") / "ralph" / "ralph_state.json"
        payload = json.loads(state_path.read_text())
        assert payload["status"] == "DISABLED"


def test_cli_ralph_set_agent_stores_in_config() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["ralph", "set-agent", "--cli", "claude-code", "--model", "claude-opus-4-6"],
        )
        assert result.exit_code == 0
        import yaml

        config_path = Path(".agent") / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        assert config["ralph"]["coding_cli"] == "claude-code"
        assert config["ralph"]["cli_model"] == "claude-opus-4-6"


def test_cli_ralph_set_agent_rejects_invalid() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["ralph", "set-agent", "--cli", "invalid-tool"])
        assert result.exit_code == 1
        assert "Invalid coding CLI" in result.output


def test_build_agent_cmd_from_ralph_config_with_model() -> None:
    cmd = build_agent_cmd_from_ralph_config({"coding_cli": "codex", "cli_model": "gpt-5.3-codex"})
    assert cmd == "codex --print --model gpt-5.3-codex"


def test_build_agent_cmd_from_ralph_config_missing_cli_returns_none() -> None:
    cmd = build_agent_cmd_from_ralph_config({"cli_model": "gpt-5.3-codex"})
    assert cmd is None


def test_ralph_run_loop_emits_output_line_when_no_cli() -> None:
    """When no coding_cli is set, run_loop emits an output_line event."""
    import asyncio

    from agent_recall.ralph.loop import RalphLoop
    from agent_recall.storage.files import FileStorage
    from agent_recall.storage.sqlite import SQLiteStorage

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path(".agent") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            json.dumps({"items": [{"id": "T-1", "title": "Test", "passes": False}]})
        )
        agent_dir = Path(".agent")
        storage = SQLiteStorage(agent_dir / "state.db")
        files = FileStorage(agent_dir)
        loop = RalphLoop(agent_dir, storage, files)

        # Enable the loop first.
        loop.enable()

        events: list[dict] = []

        def cb(event: dict) -> None:
            events.append(event)

        summary = asyncio.run(
            loop.run_loop(
                progress_callback=cb,
                coding_cli=None,
                cli_model=None,
            )
        )
        assert summary["total_iterations"] == 1
        output_events = [e for e in events if e.get("event") == "output_line"]
        assert len(output_events) >= 1
        assert "No coding CLI configured" in output_events[0]["line"]


def test_ralph_run_loop_passes_coding_cli_params() -> None:
    """run_loop accepts coding_cli and cli_model without error."""
    import asyncio

    from agent_recall.ralph.loop import RalphLoop
    from agent_recall.storage.files import FileStorage
    from agent_recall.storage.sqlite import SQLiteStorage

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        prd_path = Path(".agent") / "ralph" / "prd.json"
        prd_path.parent.mkdir(parents=True, exist_ok=True)
        prd_path.write_text(
            json.dumps({"items": [{"id": "T-2", "title": "Test 2", "passes": False}]})
        )
        agent_dir = Path(".agent")
        storage = SQLiteStorage(agent_dir / "state.db")
        files = FileStorage(agent_dir)
        loop = RalphLoop(agent_dir, storage, files)
        loop.enable()

        events: list[dict] = []

        def cb(event: dict) -> None:
            events.append(event)

        # Use a non-existent binary — should emit output_line with error.
        _summary = asyncio.run(
            loop.run_loop(
                progress_callback=cb,
                coding_cli="claude-code",
                cli_model="claude-sonnet-4-20250514",
            )
        )
        # The agent should have "failed" since the binary isn't on the test PATH.
        output_events = [e for e in events if e.get("event") == "output_line"]
        assert len(output_events) >= 1
        agent_events = [e for e in events if e.get("event") == "agent_complete"]
        assert len(agent_events) == 1
