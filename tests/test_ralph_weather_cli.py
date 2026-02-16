from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
import agent_recall.cli.ralph as ralph_cli
from agent_recall.llm.base import LLMProvider, LLMResponse, Message

runner = CliRunner()


def _write_validate_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def test_ralph_create_report_writes_current_json() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "create-report",
                "--iteration",
                "1",
                "--item-id",
                "AUTH-001",
                "--item-title",
                "JWT",
            ],
        )
        assert result.exit_code == 0
        current_path = Path(".agent") / "ralph" / "iterations" / "current.json"
        assert current_path.exists()
        payload = json.loads(current_path.read_text())
        assert payload["iteration"] == 1
        assert payload["item_id"] == "AUTH-001"


def test_ralph_finalize_report_archives_and_removes_current() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        create = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "create-report",
                "--iteration",
                "1",
                "--item-id",
                "AUTH-002",
                "--item-title",
                "JWT",
            ],
        )
        assert create.exit_code == 0
        finalize = runner.invoke(
            cli_main.app,
            ["ralph", "finalize-report", "--validation-exit", "0"],
        )
        assert finalize.exit_code == 0
        current_path = Path(".agent") / "ralph" / "iterations" / "current.json"
        assert not current_path.exists()
        archived_path = Path(".agent") / "ralph" / "iterations" / "001.json"
        assert archived_path.exists()


def test_ralph_extract_iteration_updates_current_report() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        create = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "create-report",
                "--iteration",
                "1",
                "--item-id",
                "AUTH-003",
                "--item-title",
                "JWT",
            ],
        )
        assert create.exit_code == 0
        runtime_dir = Path(".agent") / "ralph" / ".runtime"
        _write_validate_log(runtime_dir / "validate-1.log", ["E   AssertionError: boom"])
        _write_validate_log(
            runtime_dir / "agent-1.log",
            [
                '{"usage": {"prompt_tokens": 3, "completion_tokens": 2}, "model": "gpt-test"}',
            ],
        )
        current_path = Path(".agent") / "ralph" / "iterations" / "current.json"
        payload = json.loads(current_path.read_text())
        payload["validation_exit_code"] = 1
        current_path.write_text(json.dumps(payload, indent=2))
        with patch("agent_recall.ralph.extraction.subprocess.run") as mocked_run:

            def _fake_run(*_args: object, **_kwargs: object):
                class Result:
                    returncode = 0

                    def __init__(self, stdout: str):
                        self.stdout = stdout

                command = _args[0] if _args else []
                if isinstance(command, list) and len(command) >= 2 and command[1] == "diff":
                    if "--name-only" in command:
                        return Result("changed.txt\n")
                    return Result("diff --git a/foo b/foo\n+add\n")
                return Result("")

            mocked_run.side_effect = _fake_run
            extract = runner.invoke(
                cli_main.app,
                [
                    "ralph",
                    "extract-iteration",
                    "--iteration",
                    "1",
                    "--runtime-dir",
                    str(runtime_dir),
                ],
            )
        assert extract.exit_code == 0
        current_path = Path(".agent") / "ralph" / "iterations" / "current.json"
        payload = json.loads(current_path.read_text())
        assert payload["validation_hint"] == "E   AssertionError: boom"
        assert payload["token_usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
        assert payload["token_model"] == "gpt-test"
        diff_path = Path(".agent") / "ralph" / "iterations" / "001.diff"
        assert diff_path.exists()


def test_ralph_rebuild_forecast_overwrites_recent() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        agent_dir = Path(".agent")
        recent_path = agent_dir / "RECENT.md"
        recent_path.write_text("stale")
        result = runner.invoke(cli_main.app, ["ralph", "rebuild-forecast"])
        assert result.exit_code == 0
        updated = recent_path.read_text()
        assert updated != "stale"
        assert "# Current Situation" in updated


def test_ralph_synthesize_climate_writes_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "fake"

        @property
        def model_name(self) -> str:
            return "fake"

        async def generate(
            self,
            messages: list[Message],
            temperature: float = 0.3,
            max_tokens: int = 4096,
        ) -> LLMResponse:
            content = (
                "# Guardrails\n\n- fake"
                if "guardrails" in messages[0].content.lower()
                else "# Style Guide\n\n- fake"
            )
            return LLMResponse(content=content, model="fake")

        def validate(self) -> tuple[bool, str]:
            return True, "ok"

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        # Ensure at least one archived report exists
        create = runner.invoke(
            cli_main.app,
            [
                "ralph",
                "create-report",
                "--iteration",
                "1",
                "--item-id",
                "AUTH-004",
                "--item-title",
                "JWT",
            ],
        )
        assert create.exit_code == 0
        finalize = runner.invoke(
            cli_main.app,
            ["ralph", "finalize-report", "--validation-exit", "0"],
        )
        assert finalize.exit_code == 0
        monkeypatch.setattr(ralph_cli, "_get_llm", lambda: FakeLLM())
        result = runner.invoke(cli_main.app, ["ralph", "synthesize-climate", "--force"])
        assert result.exit_code == 0
        guardrails = Path(".agent") / "GUARDRAILS.md"
        style = Path(".agent") / "STYLE.md"
        assert guardrails.exists()
        assert style.exists()


def test_ralph_rebuild_forecast_use_llm_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        monkeypatch.setattr(
            ralph_cli, "_get_llm", lambda: (_ for _ in ()).throw(RuntimeError("no llm"))
        )
        result = runner.invoke(cli_main.app, ["ralph", "rebuild-forecast", "--use-llm"])
        assert result.exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["ralph", "create-report", "--iteration", "1", "--item-id", "X", "--item-title", "Y"],
        ["ralph", "finalize-report", "--validation-exit", "0"],
        [
            "ralph",
            "extract-iteration",
            "--iteration",
            "1",
            "--runtime-dir",
            ".agent/ralph/.runtime",
        ],
        ["ralph", "rebuild-forecast"],
        ["ralph", "synthesize-climate"],
    ],
)
def test_ralph_weather_commands_require_agent_dir(args: list[str]) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli_main.app, args)
        assert result.exit_code == 1
