from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import agent_recall.cli.main as cli_main

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
