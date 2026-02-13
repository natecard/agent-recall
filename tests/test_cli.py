from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

import agent_recall.cli.main as cli_main
from agent_recall.ingest.base import RawMessage, RawSession
from agent_recall.llm.base import LLMProvider, LLMResponse, Message
from agent_recall.storage.files import FileStorage
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel
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

    def get_session_id(self, path: Path) -> str:
        return f"{self.source_name}-{path.stem}"

    def parse_session(self, path: Path) -> RawSession:
        return RawSession(
            source=self.source_name,
            session_id=self.get_session_id(path),
            title=f"Session {path.stem}",
            project_path=path.parent,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            messages=[
                RawMessage(role="user", content=f"Inspect {path.stem}"),
                RawMessage(role="assistant", content=f"Reviewed {path.stem}"),
            ],
        )


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


def test_cli_context_uses_retrieval_config_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(
            self,
            _storage,
            backend: str = "fts5",
            fusion_k: int = 60,
            rerank_enabled: bool = False,
            rerank_candidate_k: int = 20,
        ):
            captured["backend"] = backend
            captured["fusion_k"] = fusion_k
            captured["rerank_enabled"] = rerank_enabled
            captured["rerank_candidate_k"] = rerank_candidate_k

        def search(
            self,
            query: str,
            top_k: int = 5,
            labels=None,
            backend=None,
            rerank=None,
            rerank_candidate_k=None,
        ) -> list[Chunk]:
            _ = (labels, backend, rerank, rerank_candidate_k)
            captured["query"] = query
            captured["top_k"] = top_k
            return [
                Chunk(
                    source=ChunkSource.MANUAL,
                    source_ids=[],
                    content="Hybrid retrieval context match",
                    label=SemanticLabel.PATTERN,
                )
            ]

    monkeypatch.setattr(cli_main, "Retriever", FakeRetriever)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        files = FileStorage(Path(".agent"))
        config = files.read_config()
        config["retrieval"] = {
            "backend": "hybrid",
            "top_k": 3,
            "fusion_k": 31,
            "rerank_enabled": True,
            "rerank_candidate_k": 9,
        }
        files.write_config(config)

        result = runner.invoke(cli_main.app, ["context", "--task", "retry strategy"])
        assert result.exit_code == 0
        assert captured["backend"] == "hybrid"
        assert captured["fusion_k"] == 31
        assert captured["rerank_enabled"] is True
        assert captured["rerank_candidate_k"] == 9
        assert captured["query"] == "retry strategy"
        assert captured["top_k"] == 3
        assert 'Relevant to "retry strategy"' in result.output


def test_cli_context_reads_shared_tier_files_when_shared_backend_enabled() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        files = FileStorage(Path(".agent"))
        config = files.read_config()

        shared_dir = Path("shared-memory").resolve()
        shared_dir.mkdir(parents=True)
        config["storage"] = {
            "backend": "shared",
            "shared": {
                "base_url": f"file://{shared_dir}",
                "api_key_env": "AGENT_RECALL_SHARED_API_KEY",
                "timeout_seconds": 10.0,
                "retry_attempts": 2,
            },
        }
        files.write_config(config)

        (Path(".agent") / "GUARDRAILS.md").write_text(
            "# Guardrails\n\nLOCAL_ONLY_GUARDRAIL\n"
        )
        (shared_dir / "GUARDRAILS.md").write_text(
            "# Guardrails\n\nSHARED_TEAM_GUARDRAIL\n"
        )

        result = runner.invoke(cli_main.app, ["context"])

        assert result.exit_code == 0
        assert "SHARED_TEAM_GUARDRAIL" in result.output
        assert "LOCAL_ONLY_GUARDRAIL" not in result.output


def test_cli_retrieve_accepts_backend_and_tuning_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(
            self,
            _storage,
            backend: str = "fts5",
            fusion_k: int = 60,
            rerank_enabled: bool = False,
            rerank_candidate_k: int = 20,
        ):
            captured["backend"] = backend
            captured["fusion_k"] = fusion_k
            captured["rerank_enabled"] = rerank_enabled
            captured["rerank_candidate_k"] = rerank_candidate_k

        def search(
            self,
            query: str,
            top_k: int = 5,
            labels=None,
            backend=None,
            rerank=None,
            rerank_candidate_k=None,
        ) -> list[Chunk]:
            _ = (labels, backend, rerank, rerank_candidate_k)
            captured["query"] = query
            captured["top_k"] = top_k
            return [
                Chunk(
                    source=ChunkSource.MANUAL,
                    source_ids=[],
                    content="Result",
                    label=SemanticLabel.PATTERN,
                )
            ]

    monkeypatch.setattr(cli_main, "Retriever", FakeRetriever)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        files = FileStorage(Path(".agent"))
        config = files.read_config()
        config["retrieval"] = {
            "backend": "hybrid",
            "top_k": 9,
            "fusion_k": 41,
            "rerank_enabled": False,
            "rerank_candidate_k": 12,
        }
        files.write_config(config)

        result = runner.invoke(
            cli_main.app,
            [
                "retrieve",
                "bounded retries",
                "--backend",
                "fts5",
                "--top-k",
                "2",
                "--fusion-k",
                "11",
                "--rerank",
                "--rerank-candidate-k",
                "4",
            ],
        )

        assert result.exit_code == 0
        assert captured["backend"] == "fts5"
        assert captured["fusion_k"] == 11
        assert captured["rerank_enabled"] is True
        assert captured["rerank_candidate_k"] == 4
        assert captured["query"] == "bounded retries"
        assert captured["top_k"] == 2


def test_cli_retrieve_uses_retrieval_config_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(
            self,
            _storage,
            backend: str = "fts5",
            fusion_k: int = 60,
            rerank_enabled: bool = False,
            rerank_candidate_k: int = 20,
        ):
            captured["backend"] = backend
            captured["fusion_k"] = fusion_k
            captured["rerank_enabled"] = rerank_enabled
            captured["rerank_candidate_k"] = rerank_candidate_k

        def search(
            self,
            query: str,
            top_k: int = 5,
            labels=None,
            backend=None,
            rerank=None,
            rerank_candidate_k=None,
        ) -> list[Chunk]:
            _ = (labels, backend, rerank, rerank_candidate_k)
            captured["query"] = query
            captured["top_k"] = top_k
            return [
                Chunk(
                    source=ChunkSource.MANUAL,
                    source_ids=[],
                    content="Config default result",
                    label=SemanticLabel.PATTERN,
                )
            ]

    monkeypatch.setattr(cli_main, "Retriever", FakeRetriever)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        files = FileStorage(Path(".agent"))
        config = files.read_config()
        config["retrieval"] = {
            "backend": "hybrid",
            "top_k": 7,
            "fusion_k": 25,
            "rerank_enabled": True,
            "rerank_candidate_k": 10,
        }
        files.write_config(config)

        result = runner.invoke(cli_main.app, ["retrieve", "semantic query"])
        assert result.exit_code == 0
        assert captured["backend"] == "hybrid"
        assert captured["fusion_k"] == 25
        assert captured["rerank_enabled"] is True
        assert captured["rerank_candidate_k"] == 10
        assert captured["query"] == "semantic query"
        assert captured["top_k"] == 7


def test_cli_refresh_context_writes_bundle_using_active_task() -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        assert runner.invoke(cli_main.app, ["start", "implement oauth callback"]).exit_code == 0

        result = runner.invoke(cli_main.app, ["refresh-context"])
        assert result.exit_code == 0
        assert "Context bundle refreshed" in result.output
        assert "implement oauth callback" in result.output

        bundle_dir = Path(".agent") / "context"
        markdown_path = bundle_dir / "context.md"
        json_path = bundle_dir / "context.json"
        assert markdown_path.exists()
        assert json_path.exists()
        assert "## Guardrails" in markdown_path.read_text()

        payload = json.loads(json_path.read_text())
        assert payload["task"] == "implement oauth callback"
        assert payload["active_session_id"] is not None
        assert payload["repo_path"]
        assert payload["refreshed_at"]


def test_cli_refresh_context_retries_transient_failure(monkeypatch) -> None:
    attempts = {"count": 0}

    class FlakyContextAssembler:
        def __init__(self, *_args, **_kwargs):
            pass

        def assemble(self, task: str | None = None) -> str:
            _ = task
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary context failure")
            return "## Guardrails\nRecovered context"

    monkeypatch.setattr(cli_main, "ContextAssembler", FlakyContextAssembler)
    monkeypatch.setattr(cli_main.time, "sleep", lambda _seconds: None)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["refresh-context", "--task", "retry diagnostics"])
        assert result.exit_code == 0
        assert attempts["count"] == 3
        assert "Retries:  2" in result.output


def test_cli_refresh_context_reports_retry_diagnostics_on_failure(monkeypatch) -> None:
    attempts = {"count": 0}

    class BrokenContextAssembler:
        def __init__(self, *_args, **_kwargs):
            pass

        def assemble(self, task: str | None = None) -> str:
            _ = task
            attempts["count"] += 1
            raise ValueError("assemble exploded")

    monkeypatch.setattr(cli_main, "ContextAssembler", BrokenContextAssembler)
    monkeypatch.setattr(cli_main.time, "sleep", lambda _seconds: None)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["refresh-context", "--task", "failing task"])
        assert result.exit_code == 1
        assert attempts["count"] == 3
        assert "Context refresh failed after 3 attempt(s)." in result.output
        assert "Attempt 3/3 failed: ValueError: assemble exploded" in result.output


def test_cli_sync_background_prints_failure_diagnostics(monkeypatch) -> None:
    class FakeBackgroundSyncManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def can_start_sync(self) -> tuple[bool, str | None]:
            return True, None

        async def run_sync(self, sources=None, max_sessions=None, compact=True):
            _ = (sources, max_sessions, compact)
            return type(
                "Result",
                (),
                {
                    "success": False,
                    "was_already_running": False,
                    "error_message": (
                        "Background sync failed after 3 attempt(s). Last error: "
                        "RuntimeError: transport unavailable"
                    ),
                    "diagnostics": [
                        (
                            "Attempt 1/3 failed (sources=all, max_sessions=auto, compact=yes): "
                            "RuntimeError: transport unavailable"
                        ),
                        (
                            "Attempt 2/3 failed (sources=all, max_sessions=auto, compact=yes): "
                            "RuntimeError: transport unavailable"
                        ),
                        (
                            "Attempt 3/3 failed (sources=all, max_sessions=auto, compact=yes): "
                            "RuntimeError: transport unavailable"
                        ),
                    ],
                    "sessions_processed": 0,
                    "learnings_extracted": 0,
                },
            )()

    monkeypatch.setattr(cli_main, "BackgroundSyncManager", FakeBackgroundSyncManager)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(cli_main, "get_default_ingesters", lambda **_kwargs: [])
    monkeypatch.setattr(cli_main, "AutoSync", lambda *_args, **_kwargs: object())

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["sync-background"])
        assert result.exit_code == 1
        assert "Sync failed" in result.output
        assert "Failure diagnostics:" in result.output
        assert "Attempt 3/3 failed" in result.output


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
    monkeypatch.setattr(
        cli_main,
        "ensure_provider_dependency",
        lambda *_args, **_kwargs: (True, None),
    )

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

        async def sync(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, sources, session_ids, max_sessions)
            called["sync"] += 1
            return {
                "sessions_discovered": 1,
                "sessions_processed": 1,
                "sessions_skipped": 0,
                "learnings_extracted": 0,
                "by_source": {},
                "errors": [],
            }

        async def sync_and_compact(
            self,
            since=None,
            sources=None,
            session_ids=None,
            max_sessions=None,
            force_compact=False,
        ):
            _ = (since, sources, session_ids, max_sessions, force_compact)
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


def test_cli_sync_session_filters_wiring(monkeypatch) -> None:
    captured: dict[str, object] = {
        "session_ids": None,
        "max_sessions": None,
    }

    class FakeAutoSync:
        def __init__(self, *_args, **_kwargs):
            pass

        async def sync(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, sources)
            captured["session_ids"] = session_ids
            captured["max_sessions"] = max_sessions
            return {
                "sessions_discovered": 2,
                "sessions_processed": 2,
                "sessions_skipped": 0,
                "empty_sessions": 0,
                "learnings_extracted": 0,
                "by_source": {},
                "errors": [],
            }

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(cli_main, "get_default_ingesters", lambda **_kwargs: [])

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            [
                "sync",
                "--no-compact",
                "--session-id",
                "cursor-a",
                "--session-id",
                "cursor-b",
                "--max-sessions",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert captured["session_ids"] == ["cursor-a", "cursor-b"]
        assert captured["max_sessions"] == 2


def test_cli_sync_selected_sessions_already_processed_feedback(monkeypatch) -> None:
    class FakeAutoSync:
        def __init__(self, *_args, **_kwargs):
            pass

        async def sync(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, sources, session_ids, max_sessions)
            return {
                "sessions_discovered": 1,
                "sessions_processed": 0,
                "sessions_skipped": 1,
                "sessions_already_processed": 1,
                "empty_sessions": 0,
                "learnings_extracted": 0,
                "by_source": {},
                "errors": [],
                "session_diagnostics": [],
            }

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(cli_main, "get_default_ingesters", lambda **_kwargs: [])

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            [
                "sync",
                "--no-compact",
                "--session-id",
                "cursor-123",
            ],
        )
        assert result.exit_code == 0
        assert "already processed" in result.output
        assert "reset-sync --session-id <id>" in result.output


def test_cli_sessions_lists_titles(monkeypatch) -> None:
    captured: dict[str, object] = {
        "session_ids": None,
        "max_sessions": None,
    }

    class FakeAutoSync:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_sessions(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, sources)
            captured["session_ids"] = session_ids
            captured["max_sessions"] = max_sessions
            return {
                "sessions_discovered": 1,
                "by_source": {"cursor": {"discovered": 1, "listed": 1}},
                "errors": [],
                "sessions": [
                    {
                        "source": "cursor",
                        "session_id": "cursor-session-1",
                        "title": "Refactor theme setup",
                        "started_at": None,
                        "ended_at": None,
                        "message_count": 2,
                        "processed": False,
                        "session_path": Path("state.vscdb"),
                    }
                ],
            }

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_default_ingesters", lambda **_kwargs: [])

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            [
                "sessions",
                "--session-id",
                "cursor-session-1",
                "--max-sessions",
                "1",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert '"title": "Refactor theme setup"' in result.output
        assert '"session_id": "cursor-session-1"' in result.output
        assert captured["session_ids"] == ["cursor-session-1"]
        assert captured["max_sessions"] == 1


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
        assert "Discovered Conversations" in result.output
        assert "Session a" in result.output
        assert "Conversation" in result.output
        assert "Session ID" not in result.output
        assert "Processed" in result.output


def test_cli_sources_conversation_table_responsive_columns(monkeypatch) -> None:
    long_title = (
        "Downloads view progress and season grouping now includes "
        "expanded metadata visibility"
    )

    class LongTitleIngester(FakeIngester):
        def parse_session(self, path: Path) -> RawSession:
            return RawSession(
                source=self.source_name,
                session_id=self.get_session_id(path),
                title=long_title,
                project_path=path.parent,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                messages=[
                    RawMessage(role="user", content=f"Inspect {path.stem}"),
                    RawMessage(role="assistant", content=f"Reviewed {path.stem}"),
                ],
            )

    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [LongTitleIngester("cursor", [Path("a")])],
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0

        monkeypatch.setattr(cli_main, "_conversation_table_mode", lambda _width: "wide")
        wide = runner.invoke(cli_main.app, ["sources"])

        monkeypatch.setattr(cli_main, "_conversation_table_mode", lambda _width: "compact")
        compact = runner.invoke(cli_main.app, ["sources"])

        assert wide.exit_code == 0
        assert compact.exit_code == 0
        assert "Ref" in wide.output
        assert "Ref" not in compact.output
        assert long_title.split()[-1] in compact.output


def test_cli_providers_command() -> None:
    result = runner.invoke(cli_main.app, ["providers"])
    assert result.exit_code == 0
    assert "openai-compatible" in result.output
    assert "ollama" in result.output


def test_cli_config_llm_no_validate(monkeypatch) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        monkeypatch.setattr(
            cli_main,
            "ensure_provider_dependency",
            lambda *_args, **_kwargs: (True, None),
        )
        result = runner.invoke(
            cli_main.app,
            ["config-llm", "--provider", "ollama", "--model", "llama3.1", "--no-validate"],
        )
        assert result.exit_code == 0
        assert "LLM Configuration Updated" in result.output


def test_cli_config_llm_generation_settings(monkeypatch) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        monkeypatch.setattr(
            cli_main,
            "ensure_provider_dependency",
            lambda *_args, **_kwargs: (True, None),
        )
        result = runner.invoke(
            cli_main.app,
            ["config-llm", "--temperature", "0.2", "--max-tokens", "8192", "--no-validate"],
        )
        assert result.exit_code == 0
        assert "Temperature: 0.2" in result.output
        assert "Max tokens: 8192" in result.output

        config = FileStorage(Path(".agent")).read_config()
        assert config["llm"]["temperature"] == 0.2
        assert config["llm"]["max_tokens"] == 8192


def test_cli_config_model_subcommand(monkeypatch) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        monkeypatch.setattr(
            cli_main,
            "ensure_provider_dependency",
            lambda *_args, **_kwargs: (True, None),
        )
        result = runner.invoke(
            cli_main.app,
            ["config", "model", "--provider", "ollama", "--model", "llama3.1", "--no-validate"],
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

        async def sync(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, sources, session_ids, max_sessions)
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
    monkeypatch.setattr(
        "agent_recall.core.onboarding.ensure_provider_dependency",
        lambda *_args, **_kwargs: (True, None),
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["tui", "--iterations", "1", "--refresh-seconds", "0.2"],
        )
        assert result.exit_code == 0


def test_cli_open_single_iteration(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("state.vscdb")]),
            FakeIngester("claude-code", []),
        ],
    )
    monkeypatch.setattr(
        "agent_recall.core.onboarding.ensure_provider_dependency",
        lambda *_args, **_kwargs: (True, None),
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(
            cli_main.app,
            ["open", "--iterations", "1", "--refresh-seconds", "0.2"],
        )
        assert result.exit_code == 0


def test_cli_open_injects_stored_api_keys(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_inject() -> int:
        calls["count"] += 1
        return 0

    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("state.vscdb")]),
            FakeIngester("claude-code", []),
        ],
    )
    monkeypatch.setattr(cli_main, "inject_stored_api_keys", fake_inject)

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["open", "--iterations", "1"])
        assert result.exit_code == 0
        assert calls["count"] == 1


def test_cli_onboard_quick_persists_repo_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("cursor-session")]),
            FakeIngester("claude-code", [Path("claude-session")]),
        ],
    )
    monkeypatch.setattr(
        "agent_recall.core.onboarding.ensure_provider_dependency",
        lambda *_args, **_kwargs: (True, None),
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["onboard", "--quick"])
        assert result.exit_code == 0

        config = FileStorage(Path(".agent")).read_config()
        onboarding = config.get("onboarding", {})

        assert onboarding.get("repository_verified") is True
        assert onboarding.get("selected_agents") == [
            "cursor",
            "claude-code",
            "opencode",
            "codex",
        ]
        assert config.get("llm", {}).get("provider") == "anthropic"
        assert config.get("llm", {}).get("temperature") == 0.3
        assert config.get("llm", {}).get("max_tokens") == 4096


def test_cli_config_setup_quick_persists_repo_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("cursor-session")]),
            FakeIngester("claude-code", [Path("claude-session")]),
        ],
    )
    monkeypatch.setattr(
        "agent_recall.core.onboarding.ensure_provider_dependency",
        lambda *_args, **_kwargs: (True, None),
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0
        result = runner.invoke(cli_main.app, ["config", "setup", "--quick"])
        assert result.exit_code == 0

        config = FileStorage(Path(".agent")).read_config()
        onboarding = config.get("onboarding", {})

        assert onboarding.get("repository_verified") is True
        assert onboarding.get("selected_agents") == [
            "cursor",
            "claude-code",
            "opencode",
            "codex",
        ]
        assert config.get("llm", {}).get("provider") == "anthropic"


def test_cli_sync_uses_onboarding_selected_sources(monkeypatch) -> None:
    captured: dict[str, object] = {"sources": None, "ingesters": []}

    class FakeAutoSync:
        def __init__(self, _storage, _files, _llm, ingesters):
            captured["ingesters"] = [ingester.source_name for ingester in ingesters]

        async def sync(self, since=None, sources=None, session_ids=None, max_sessions=None):
            _ = (since, session_ids, max_sessions)
            captured["sources"] = sources
            return {
                "sessions_discovered": 1,
                "sessions_processed": 1,
                "sessions_skipped": 0,
                "learnings_extracted": 1,
                "by_source": {},
                "errors": [],
            }

        async def sync_and_compact(
            self,
            since=None,
            sources=None,
            session_ids=None,
            max_sessions=None,
            force_compact=False,
        ):
            _ = (since, sources, session_ids, max_sessions, force_compact)
            raise AssertionError("sync_and_compact should not be called when --no-compact is set")

    monkeypatch.setattr(cli_main, "AutoSync", FakeAutoSync)
    monkeypatch.setattr(cli_main, "get_llm", lambda: DummyProvider())
    monkeypatch.setattr(
        cli_main,
        "get_default_ingesters",
        lambda **_kwargs: [
            FakeIngester("cursor", [Path("a")]),
            FakeIngester("claude-code", [Path("b")]),
        ],
    )

    with runner.isolated_filesystem():
        assert runner.invoke(cli_main.app, ["init"]).exit_code == 0

        files = FileStorage(Path(".agent"))
        config = files.read_config()
        config["onboarding"] = {
            "completed_at": "2026-02-01T00:00:00+00:00",
            "repository_path": str(Path.cwd().resolve()),
            "selected_agents": ["cursor"],
        }
        files.write_config(config)

        result = runner.invoke(cli_main.app, ["sync", "--no-compact"])
        assert result.exit_code == 0
        assert captured["sources"] == ["cursor"]
        assert captured["ingesters"] == ["cursor"]
        assert "Using configured sources: cursor" in result.output


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


def test_tui_slash_quit_command() -> None:
    should_exit, lines = cli_main._execute_tui_slash_command("/quit")
    assert should_exit is True
    assert any("Leaving TUI" in line for line in lines)


def test_tui_slash_compact_dispatch(monkeypatch) -> None:
    class FakeResult:
        exit_code = 0
        output = "Compaction complete\nChunks indexed: 3\n"

    captured = {"terminal_width": None, "columns": None, "lines": None}

    def fake_invoke(*_args, **kwargs):
        captured["terminal_width"] = kwargs.get("terminal_width")
        env = kwargs.get("env") or {}
        captured["columns"] = env.get("COLUMNS")
        captured["lines"] = env.get("LINES")
        return FakeResult()

    monkeypatch.setattr(cli_main._slash_runner, "invoke", fake_invoke)

    should_exit, lines = cli_main._execute_tui_slash_command(
        "/compact --force",
        terminal_width=132,
        terminal_height=44,
    )

    assert should_exit is False
    assert any("/compact --force" in line for line in lines)
    assert any("Compaction complete" in line for line in lines)
    assert captured["terminal_width"] == 132
    assert captured["columns"] == "132"
    assert captured["lines"] == "44"


def test_tui_slash_run_alias_dispatch(monkeypatch) -> None:
    captured = {"args": None}

    class FakeResult:
        exit_code = 0
        output = "Sync complete\n"

    def fake_invoke(_app, args, **_kwargs):
        captured["args"] = args
        return FakeResult()

    monkeypatch.setattr(cli_main._slash_runner, "invoke", fake_invoke)

    should_exit, lines = cli_main._execute_tui_slash_command("/run --max-sessions 2")

    assert should_exit is False
    assert captured["args"] == ["sync", "--max-sessions", "2"]
    assert any("/run --max-sessions 2" in line for line in lines)


def test_tui_slash_preserves_table_output(monkeypatch) -> None:
    class FakeResult:
        exit_code = 0
        output = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃ Source        Sessions  Status             ┃\n"
            "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
            "│ cursor        62        ✓ Available        │\n"
            "└─────────────────────────────────────────────┘\n"
        )

    monkeypatch.setattr(
        cli_main._slash_runner,
        "invoke",
        lambda *_args, **_kwargs: FakeResult(),
    )

    should_exit, lines = cli_main._execute_tui_slash_command("/sources")

    assert should_exit is False
    assert any("cursor" in line.lower() for line in lines)
    assert any("┏" in line or "└" in line for line in lines)


def test_tui_slash_normalizes_theme_table_rows(monkeypatch) -> None:
    class FakeResult:
        exit_code = 0
        output = (
            "\x1b[36m┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓\x1b[0m\n"
            "\x1b[36m┃ Name                           ┃ Status     ┃\x1b[0m\n"
            "\x1b[36m┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩\x1b[0m\n"
            "\x1b[37m│ dark+                          │ ✓ Current  │\x1b[0m\n"
            "\x1b[37m│ light+                         │            │\x1b[0m\n"
            "\x1b[36m└────────────────────────────────┴────────────┘\x1b[0m\n"
        )

    monkeypatch.setattr(
        cli_main._slash_runner,
        "invoke",
        lambda *_args, **_kwargs: FakeResult(),
    )

    should_exit, lines = cli_main._execute_tui_slash_command("/theme list")

    assert should_exit is False
    assert any("dark+" in line for line in lines)
    assert any("light+" in line for line in lines)
    assert any("│" in line or "┏" in line or "┗" in line for line in lines)


def test_tui_slash_does_not_truncate_output_lines(monkeypatch) -> None:
    class FakeResult:
        exit_code = 0
        output = "\n".join(f"line-{index}" for index in range(1, 13)) + "\n"

    monkeypatch.setattr(
        cli_main._slash_runner,
        "invoke",
        lambda *_args, **_kwargs: FakeResult(),
    )

    should_exit, lines = cli_main._execute_tui_slash_command("/status")

    assert should_exit is False
    assert any("line-12" in line for line in lines)
    assert not any("... and " in line for line in lines)


def test_tui_slash_disallows_nested_tui() -> None:
    should_exit, lines = cli_main._execute_tui_slash_command("/tui")
    assert should_exit is False
    assert any("already running" in line for line in lines)

    should_exit, lines = cli_main._execute_tui_slash_command("/open")
    assert should_exit is False
    assert any("already running" in line for line in lines)


def test_tui_view_command_switches_view() -> None:
    handled, next_view, lines = cli_main._handle_tui_view_command("/view llm", "overview")
    assert handled is True
    assert next_view == "llm"
    assert any("Switched to llm" in line for line in lines)


def test_tui_view_command_rejects_unknown_view() -> None:
    handled, next_view, lines = cli_main._handle_tui_view_command("/view unknown", "overview")
    assert handled is True
    assert next_view == "overview"
    assert any("Unknown view" in line for line in lines)


def test_tui_normalize_command_accepts_non_slash() -> None:
    assert cli_main._normalize_tui_command("config setup --quick") == "/config setup --quick"
    assert cli_main._normalize_tui_command("/status") == "/status"


def test_conversation_table_mode_thresholds() -> None:
    assert cli_main._conversation_table_mode(160) == "wide"
    assert cli_main._conversation_table_mode(130) == "medium"
    assert cli_main._conversation_table_mode(100) == "compact"


def test_tui_slash_config_setup_uses_direct_flow(monkeypatch) -> None:
    captured = {"force": None, "quick": None}

    def fake_setup(force: bool, quick: bool) -> None:
        captured["force"] = force
        captured["quick"] = quick

    monkeypatch.setattr(cli_main, "_run_onboarding_setup", fake_setup)

    should_exit, lines = cli_main._execute_tui_slash_command("config setup --force --quick")
    assert should_exit is False
    assert captured["force"] is True
    assert captured["quick"] is True
    assert any("Setup flow completed" in line for line in lines)


def test_tui_settings_command_helper() -> None:
    should_exit, lines = cli_main._execute_tui_slash_command("/settings")
    assert should_exit is False
    assert any("settings view" in line.lower() for line in lines)


def test_tui_palette_command_catalog_includes_cli_commands() -> None:
    commands = cli_main._collect_cli_commands_for_palette()
    assert "status" in commands
    assert "config setup" in commands
    assert "config model" in commands
