from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.llm.base import LLMProvider, LLMResponse, Message
from agent_recall.storage.sqlite import SQLiteStorage

runner = CliRunner()


class DummyProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def model_name(self) -> str:
        return "dummy-model"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        return LLMResponse(content="NONE", model="dummy")

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


class FakeIngester:
    def __init__(self, source_name: str, sessions: list[Path]):
        self.source_name = source_name
        self._sessions = sessions

    def discover_sessions(self, since=None):
        _ = since
        return self._sessions


def test_cli_help_import_sanity() -> None:
    result = runner.invoke(cli_main.app, ["--help"])
    assert result.exit_code == 0
    assert "Agent Memory System" in result.output


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


def test_cli_sync_no_compact(monkeypatch) -> None:
    called = {"sync": 0, "sync_and_compact": 0}

    class FakeAutoSync:
        def __init__(self, *_args, **_kwargs):
            pass

        async def sync(self, since=None, sources=None):
            _ = (since, sources)
            called["sync"] += 1
            return {
                "sessions_discovered": 1,
                "sessions_processed": 1,
                "sessions_skipped": 0,
                "learnings_extracted": 0,
                "by_source": {},
                "errors": [],
            }

        async def sync_and_compact(self, since=None, sources=None, force_compact=False):
            _ = (since, sources, force_compact)
            called["sync_and_compact"] += 1
            return {
                "sessions_discovered": 1,
                "sessions_processed": 1,
                "sessions_skipped": 0,
                "learnings_extracted": 1,
                "by_source": {},
                "errors": [],
                "compaction": {
                    "guardrails_updated": False,
                    "style_updated": True,
                    "chunks_indexed": 1,
                },
            }

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(cli_main, "get_default_ingesters", lambda **_kwargs: [])

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["sync", "--no-compact"])
        assert result.exit_code == 0
        assert "Sessions discovered:" in result.output
        assert called["sync"] == 1
        assert called["sync_and_compact"] == 0


def test_cli_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("a")]),
            FakeIngester("claude-code", []),
        ],
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["sources"])
        assert result.exit_code == 0
        assert "cursor" in result.output
        assert "claude-code" in result.output


def test_cli_providers_command() -> None:
    result = runner.invoke(cli_main.app, ["providers"])
    assert result.exit_code == 0
    assert "openai-compatible" in result.output
    assert "ollama" in result.output


def test_cli_config_llm_no_validate() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["config-llm", "--provider", "ollama", "--model", "llama3.1", "--no-validate"],
        )
        assert result.exit_code == 0
        assert "LLM Configuration Updated" in result.output


def test_cli_test_llm(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["test-llm"])
        assert result.exit_code == 0
        assert "Generation successful" in result.output


def test_cli_sync_cursor_override_wiring(monkeypatch) -> None:
    captured = {
        "cursor_db_path": None,
        "workspace_storage_dir": None,
        "cursor_all_workspaces": None,
    }

    class FakeAutoSync:
        def __init__(self, *_args, **_kwargs):
            pass

        async def sync(self, since=None, sources=None):
            _ = (since, sources)
            return {
                "sessions_discovered": 1,
                "sessions_processed": 1,
                "sessions_skipped": 0,
                "empty_sessions": 0,
                "learnings_extracted": 0,
                "by_source": {},
                "errors": [],
            }

    def fake_get_ingester(
        source,
        project_path=None,
        cursor_db_path=None,
        workspace_storage_dir=None,
        cursor_all_workspaces=False,
    ):
        _ = (source, project_path)
        captured["cursor_db_path"] = cursor_db_path
        captured["workspace_storage_dir"] = workspace_storage_dir
        captured["cursor_all_workspaces"] = cursor_all_workspaces
        return FakeIngester("cursor", [Path("session")])

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(cli_main, "get_ingester", fake_get_ingester)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        db_override = Path("custom.vscdb")
        storage_override = Path("workspace-storage")
        result = runner.invoke(
            cli_main.app,
            [
                "sync",
                "--source",
                "cursor",
                "--no-compact",
                "--cursor-db-path",
                str(db_override),
                "--cursor-storage-dir",
                str(storage_override),
                "--all-cursor-workspaces",
            ],
        )
        assert result.exit_code == 0
        assert captured["cursor_db_path"] == db_override
        assert captured["workspace_storage_dir"] == storage_override
        assert captured["cursor_all_workspaces"] is True


def test_cli_sources_all_cursor_workspaces_flag(monkeypatch) -> None:
    captured = {"cursor_all_workspaces": None}

    def fake_get_default_ingesters(**kwargs):
        captured["cursor_all_workspaces"] = kwargs.get("cursor_all_workspaces")
        return [
            FakeIngester("cursor", [Path("session")]),
            FakeIngester("claude-code", []),
        ]

    monkeypatch.setattr(cli_main, "get_default_ingesters", fake_get_default_ingesters)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["sources", "--all-cursor-workspaces"])
        assert result.exit_code == 0
        assert captured["cursor_all_workspaces"] is True


def test_cli_reset_sync_all_markers() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0

        storage = SQLiteStorage(Path(".agent") / "state.db")
        storage.mark_session_processed("cursor-workspace-1")
        storage.mark_session_processed("claude-code-session-1")

        result = runner.invoke(cli_main.app, ["reset-sync"])
        assert result.exit_code == 0
        assert "Cleared processed session markers: 2" in result.output

        refreshed = SQLiteStorage(Path(".agent") / "state.db")
        assert refreshed.is_session_processed("cursor-workspace-1") is False
        assert refreshed.is_session_processed("claude-code-session-1") is False


def test_cli_reset_sync_by_source() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0

        storage = SQLiteStorage(Path(".agent") / "state.db")
        storage.mark_session_processed("cursor-workspace-2")
        storage.mark_session_processed("claude-code-session-2")

        result = runner.invoke(cli_main.app, ["reset-sync", "--source", "cursor"])
        assert result.exit_code == 0
        assert "Cleared processed session markers: 1" in result.output

        refreshed = SQLiteStorage(Path(".agent") / "state.db")
        assert refreshed.is_session_processed("cursor-workspace-2") is False
        assert refreshed.is_session_processed("claude-code-session-2") is True


def test_cli_tui_single_iteration(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("state.vscdb")]),
            FakeIngester("claude-code", []),
        ],
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["tui", "--iterations", "1", "--refresh-seconds", "0.2"],
        )
        assert result.exit_code == 0


def test_cli_compact_uses_spinner(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_with_spinner(description: str, action):
        calls.append(description)
        return action()

    monkeypatch.setattr(cli_main, "run_with_spinner", fake_run_with_spinner)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["start", "task"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["log", "note", "--label", "gotcha"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["end", "done"]).exit_code == 0

        result = runner.invoke(cli_main.app, ["compact"])
        assert result.exit_code == 0
        assert any("compaction" in description.lower() for description in calls)


def test_cli_test_llm_uses_spinners(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_with_spinner(description: str, action):
        calls.append(description)
        return action()

    monkeypatch.setattr(cli_main, "run_with_spinner", fake_run_with_spinner)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["test-llm"])
        assert result.exit_code == 0
        assert any("validating" in description.lower() for description in calls)
        assert any("generation" in description.lower() for description in calls)


def test_cli_ingest_uses_spinner(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_with_spinner(description: str, action):
        calls.append(description)
        return action()

    monkeypatch.setattr(cli_main, "run_with_spinner", fake_run_with_spinner)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        transcript = Path("session.jsonl")
        transcript.write_text('{"content": "hello from transcript"}\n')
        result = runner.invoke(cli_main.app, ["ingest", str(transcript)])
        assert result.exit_code == 0
        assert any("ingesting" in description.lower() for description in calls)
