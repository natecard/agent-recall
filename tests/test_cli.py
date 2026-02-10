from __future__ import annotations

from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.llm.base import LLMProvider, LLMResponse, Message

runner = CliRunner()


class DummyProvider(LLMProvider):
    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        return LLMResponse(content="NONE", model="dummy")


def test_cli_init_creates_agent_dir() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli_main.app, ["init"])
        assert result.exit_code == 0


def test_cli_session_flow() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["start", "build feature"]).exit_code == 0
        assert (
            runner.invoke(
                cli_main.app,
                [
                    "log",
                    "Remember to backfill default values",
                    "--label",
                    "gotcha",
                    "--tags",
                    "db,migration",
                ],
            ).exit_code
            == 0
        )
        assert runner.invoke(cli_main.app, ["end", "finished implementation"]).exit_code == 0

        context_result = runner.invoke(cli_main.app, ["context", "--task", "migration"])
        assert context_result.exit_code == 0

        status_result = runner.invoke(cli_main.app, ["status"])
        assert status_result.exit_code == 0
        assert "Log entries:" in status_result.output


def test_cli_invalid_label() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["log", "bad label", "--label", "not-a-label"],
        )
        assert result.exit_code == 1
        assert "Invalid label" in result.output


def test_cli_compact(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "create_llm_provider", lambda *_args, **_kwargs: DummyProvider())

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["start", "task"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["log", "note", "--label", "gotcha"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["end", "done"]).exit_code == 0

        result = runner.invoke(cli_main.app, ["compact"])
        assert result.exit_code == 0
        assert "Compaction complete" in result.output
