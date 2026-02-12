from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_recall.ingest import get_default_ingesters, get_ingester
from agent_recall.ingest.base import RawMessage, RawSession
from agent_recall.ingest.claude_code import ClaudeCodeIngester
from agent_recall.ingest.cursor import CursorIngester
from agent_recall.ingest.opencode import OpenCodeIngester
from agent_recall.llm.base import LLMProvider, LLMResponse, Message


class TestCursorIngester:
    def test_source_name(self) -> None:
        ingester = CursorIngester()
        assert ingester.source_name == "cursor"

    def test_discover_sessions_no_workspace(self, tmp_path: Path) -> None:
        ingester = CursorIngester(project_path=tmp_path)
        ingester.storage_dir = tmp_path / "nonexistent"

        sessions = ingester.discover_sessions()
        assert sessions == []

    def test_parse_session_empty_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        ingester = CursorIngester(project_path=tmp_path)
        session = ingester.parse_session(db_path)

        assert session.source == "cursor"
        assert session.messages == []

    def test_discover_sessions_with_cursor_db_override(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        ingester = CursorIngester(project_path=tmp_path, cursor_db_path=db_path)
        sessions = ingester.discover_sessions()

        assert sessions == [db_path]

    def test_workspace_match_from_code_workspace(self, tmp_path: Path) -> None:
        project_path = tmp_path / "repo"
        project_path.mkdir()

        workspace_file = tmp_path / "test.code-workspace"
        workspace_file.write_text(
            json.dumps(
                {
                    "folders": [
                        {"path": str(project_path)},
                    ]
                }
            )
        )

        storage_dir = tmp_path / "workspaceStorage"
        workspace_dir = storage_dir / "abc123"
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "state.vscdb").write_text("")
        (workspace_dir / "workspace.json").write_text(
            json.dumps({"workspace": f"file://{workspace_file}"})
        )

        ingester = CursorIngester(
            project_path=project_path,
            workspace_storage_dir=storage_dir,
        )

        sessions = ingester.discover_sessions()
        assert sessions == [workspace_dir / "state.vscdb"]

    def test_workspace_match_from_folder_uri(self, tmp_path: Path) -> None:
        project_path = tmp_path / "repo"
        project_path.mkdir()

        storage_dir = tmp_path / "workspaceStorage"
        workspace_dir = storage_dir / "xyz789"
        workspace_dir.mkdir(parents=True)
        (workspace_dir / "state.vscdb").write_text("")
        (workspace_dir / "workspace.json").write_text(
            json.dumps({"folder": f"file://{project_path}"})
        )

        ingester = CursorIngester(
            project_path=project_path,
            workspace_storage_dir=storage_dir,
        )

        sessions = ingester.discover_sessions()
        assert sessions == [workspace_dir / "state.vscdb"]

    def test_discover_all_workspaces_mode(self, tmp_path: Path) -> None:
        storage_dir = tmp_path / "workspaceStorage"
        first = storage_dir / "ws1"
        second = storage_dir / "ws2"
        first.mkdir(parents=True)
        second.mkdir(parents=True)

        first_db = first / "state.vscdb"
        second_db = second / "state.vscdb"
        first_db.write_text("")
        second_db.write_text("")

        (first / "workspace.json").write_text(json.dumps({"folder": "file:///tmp/repo-one"}))
        (second / "workspace.json").write_text(json.dumps({"folder": "file:///tmp/repo-two"}))

        ingester = CursorIngester(
            project_path=tmp_path,
            workspace_storage_dir=storage_dir,
            include_all_workspaces=True,
        )
        sessions = sorted(ingester.discover_sessions())
        assert sessions == sorted([first_db, second_db])

    def test_discover_sessions_recursively_finds_nested_workspace(self, tmp_path: Path) -> None:
        project_path = tmp_path / "repo"
        project_path.mkdir()

        storage_dir = tmp_path / "workspaceStorage"
        nested_workspace = storage_dir / "level1" / "level2" / "workspace-hash"
        nested_workspace.mkdir(parents=True)
        db_path = nested_workspace / "state.vscdb"
        db_path.write_text("")
        (nested_workspace / "workspace.json").write_text(
            json.dumps({"folder": f"file://{project_path}"})
        )

        ingester = CursorIngester(
            project_path=project_path,
            workspace_storage_dir=storage_dir,
        )
        sessions = ingester.discover_sessions()
        assert sessions == [db_path]

    def test_discover_sessions_splits_by_composer(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "composer.composerData",
                json.dumps(
                    {
                        "allComposers": [
                            {
                                "composerId": "composer-a",
                                "createdAt": 1_740_000_000_000,
                                "lastUpdatedAt": 1_740_000_001_000,
                            },
                            {
                                "composerId": "composer-b",
                                "createdAt": 1_740_000_100_000,
                                "lastUpdatedAt": 1_740_000_101_000,
                            },
                        ]
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        ingester = CursorIngester(
            project_path=tmp_path,
            cursor_db_path=db_path,
            global_storage_db_path=tmp_path / "globalStorage" / "state.vscdb",
        )
        sessions = ingester.discover_sessions()

        assert len(sessions) == 2
        session_ids = {ingester.get_session_id(path) for path in sessions}
        assert any("composer-a" in session_id for session_id in session_ids)
        assert any("composer-b" in session_id for session_id in session_ids)

    def test_parse_composer_session_from_global_cursor_kv(self, tmp_path: Path) -> None:
        workspace_db = tmp_path / "state.vscdb"
        workspace_conn = sqlite3.connect(str(workspace_db))
        workspace_conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")
        workspace_conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "composer.composerData",
                json.dumps(
                    {
                        "allComposers": [
                            {
                                "composerId": "composer-a",
                                "createdAt": 1_740_000_000_000,
                                "lastUpdatedAt": 1_740_000_001_000,
                            }
                        ]
                    }
                ),
            ),
        )
        workspace_conn.commit()
        workspace_conn.close()

        global_db = tmp_path / "globalStorage" / "state.vscdb"
        global_db.parent.mkdir(parents=True)
        global_conn = sqlite3.connect(str(global_db))
        global_conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)")
        global_conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "composerData:composer-a",
                json.dumps(
                    {
                        "composerId": "composer-a",
                        "name": "Theme cleanup session",
                        "createdAt": 1_740_000_000_000,
                        "lastUpdatedAt": 1_740_000_001_000,
                        "fullConversationHeadersOnly": [
                            {"bubbleId": "bubble-user", "type": 1},
                            {"bubbleId": "bubble-assistant", "type": 2},
                        ],
                    }
                ),
            ),
        )
        global_conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:composer-a:bubble-user",
                json.dumps(
                    {
                        "type": 1,
                        "text": "Please move color settings into a theme module.",
                        "createdAt": "2025-02-19T12:00:00Z",
                        "requestId": "req-1",
                    }
                ),
            ),
        )
        global_conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
            (
                "bubbleId:composer-a:bubble-assistant",
                json.dumps(
                    {
                        "type": 2,
                        "text": "Done. Theme values now come from a shared config.",
                        "createdAt": "2025-02-19T12:00:01Z",
                        "toolFormerData": {
                            "name": "edit_file_v2",
                            "status": "completed",
                            "params": "{\"path\":\"src/theme.py\"}",
                        },
                    }
                ),
            ),
        )
        global_conn.commit()
        global_conn.close()

        ingester = CursorIngester(
            project_path=tmp_path,
            cursor_db_path=workspace_db,
            global_storage_db_path=global_db,
        )
        sessions = ingester.discover_sessions()
        assert len(sessions) == 1

        session = ingester.parse_session(sessions[0])
        assert session.source == "cursor"
        assert session.title == "Theme cleanup session"
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "theme module" in session.messages[0].content
        assert session.messages[1].role == "assistant"
        assert "shared config" in session.messages[1].content
        assert session.messages[1].tool_calls[0].tool == "edit_file_v2"

    def test_parse_ai_service_prompts_and_generations(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "aiService.prompts",
                json.dumps(
                    [
                        {
                            "text": "We should use theme variables for all colors.",
                        }
                    ]
                ),
            ),
        )
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "aiService.generations",
                json.dumps(
                    [
                        {
                            "unixMs": 1_740_000_000_000,
                            "textDescription": "Implemented theme presets and config wiring.",
                        }
                    ]
                ),
            ),
        )
        conn.commit()
        conn.close()

        ingester = CursorIngester(project_path=tmp_path, cursor_db_path=db_path)
        session = ingester.parse_session(db_path)

        assert len(session.messages) == 2
        assert session.title is not None
        assert "theme variables" in session.title
        assert session.messages[0].role == "user"
        assert "theme variables" in session.messages[0].content
        assert session.messages[1].role == "assistant"
        assert "theme presets" in session.messages[1].content

    def test_parse_aichat_chatdata_bubbles(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (
                "workbench.panel.aichat.chatdata.test",
                json.dumps(
                    {
                        "tabs": [
                            {
                                "bubbles": [
                                    {"type": "user", "text": "Please refactor theme colors"},
                                    {"type": "ai", "rawText": "Done, variables are centralized."},
                                ]
                            }
                        ]
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        ingester = CursorIngester(project_path=tmp_path, cursor_db_path=db_path)
        session = ingester.parse_session(db_path)

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "refactor theme" in session.messages[0].content
        assert session.messages[1].role == "assistant"
        assert "variables are centralized" in session.messages[1].content


class TestIngesterRegistry:
    def test_get_default_ingesters_includes_opencode(self, tmp_path: Path) -> None:
        names = [item.source_name for item in get_default_ingesters(project_path=tmp_path)]
        assert "cursor" in names
        assert "claude-code" in names
        assert "opencode" in names

    def test_get_ingester_normalizes_aliases(self, tmp_path: Path) -> None:
        assert get_ingester("claudecode", project_path=tmp_path).source_name == "claude-code"
        assert get_ingester("open_code", project_path=tmp_path).source_name == "opencode"


class TestClaudeCodeIngester:
    def test_source_name(self) -> None:
        ingester = ClaudeCodeIngester()
        assert ingester.source_name == "claude-code"

    def test_parse_session_jsonl(self, tmp_path: Path) -> None:
        session_file = tmp_path / "test-session.jsonl"

        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00Z"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2024-01-01T10:00:01Z"},
        ]
        session_file.write_text("\n".join(json.dumps(msg) for msg in messages) + "\n")

        ingester = ClaudeCodeIngester(project_path=tmp_path)
        session = ingester.parse_session(session_file)

        assert session.source == "claude-code"
        assert session.title == "Hello"
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"
        assert session.messages[1].role == "assistant"

    def test_parse_session_with_tool_calls(self, tmp_path: Path) -> None:
        session_file = tmp_path / "test-session.jsonl"

        messages = [
            {
                "role": "assistant",
                "content": "I'll read that file.",
                "tool_calls": [
                    {"name": "Read", "args": {"file": "test.py"}, "result": "# code"}
                ],
                "timestamp": "2024-01-01T10:00:00Z",
            }
        ]
        session_file.write_text("\n".join(json.dumps(msg) for msg in messages) + "\n")

        ingester = ClaudeCodeIngester(project_path=tmp_path)
        session = ingester.parse_session(session_file)

        assert len(session.messages) == 1
        assert session.title == "test session"
        assert len(session.messages[0].tool_calls) == 1
        assert session.messages[0].tool_calls[0].tool == "Read"


class TestOpenCodeIngester:
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_source_name(self, tmp_path: Path) -> None:
        ingester = OpenCodeIngester(project_path=tmp_path, opencode_dir=tmp_path / "opencode")
        assert ingester.source_name == "opencode"

    def test_discover_sessions_matches_project(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        other_repo = tmp_path / "other-repo"
        other_repo.mkdir()

        opencode_dir = tmp_path / "opencode"
        storage = opencode_dir / "storage"

        self._write_json(
            storage / "project" / "proj-main.json",
            {
                "id": "proj-main",
                "worktree": str(repo_path),
                "time": {"created": 1_766_000_000_000, "updated": 1_766_000_500_000},
            },
        )
        self._write_json(
            storage / "project" / "proj-other.json",
            {
                "id": "proj-other",
                "worktree": str(other_repo),
                "time": {"created": 1_766_000_000_000, "updated": 1_766_000_500_000},
            },
        )

        wanted = storage / "session" / "proj-main" / "ses_main.json"
        other = storage / "session" / "proj-other" / "ses_other.json"

        self._write_json(
            wanted,
            {
                "id": "ses_main",
                "projectID": "proj-main",
                "directory": str(repo_path),
                "title": "Main session",
                "time": {"created": 1_766_000_000_000, "updated": 1_766_000_500_000},
            },
        )
        self._write_json(
            other,
            {
                "id": "ses_other",
                "projectID": "proj-other",
                "directory": str(other_repo),
                "title": "Other session",
                "time": {"created": 1_766_001_000_000, "updated": 1_766_001_500_000},
            },
        )

        ingester = OpenCodeIngester(project_path=repo_path, opencode_dir=opencode_dir)

        sessions = ingester.discover_sessions()
        assert sessions == [wanted]

        since = datetime.fromtimestamp(1_766_000_600, tz=UTC)
        filtered = ingester.discover_sessions(since=since)
        assert filtered == []

    def test_parse_session_extracts_messages_and_tools(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        opencode_dir = tmp_path / "opencode"
        storage = opencode_dir / "storage"

        self._write_json(
            storage / "project" / "proj-main.json",
            {
                "id": "proj-main",
                "worktree": str(repo_path),
                "time": {"created": 1_766_000_000_000, "updated": 1_766_000_900_000},
            },
        )

        session_path = storage / "session" / "proj-main" / "ses_main.json"
        self._write_json(
            session_path,
            {
                "id": "ses_main",
                "projectID": "proj-main",
                "directory": str(repo_path),
                "title": "Playback fixes",
                "time": {"created": 1_766_000_000_000, "updated": 1_766_000_900_000},
            },
        )

        self._write_json(
            storage / "message" / "ses_main" / "msg_user.json",
            {
                "id": "msg_user",
                "sessionID": "ses_main",
                "role": "user",
                "time": {"created": 1_766_000_010_000},
            },
        )
        self._write_json(
            storage / "message" / "ses_main" / "msg_assistant.json",
            {
                "id": "msg_assistant",
                "sessionID": "ses_main",
                "role": "assistant",
                "time": {"created": 1_766_000_020_000, "completed": 1_766_000_030_000},
            },
        )
        self._write_json(
            storage / "message" / "ses_main" / "msg_noise.json",
            {
                "id": "msg_noise",
                "sessionID": "ses_main",
                "role": "assistant",
                "time": {"created": 1_766_000_015_000},
            },
        )

        self._write_json(
            storage / "part" / "msg_user" / "prt_user_text.json",
            {
                "id": "prt_user_text",
                "type": "text",
                "text": "Please fix duplicate playback when opening from multiple screens.",
                "time": {"start": 1_766_000_010_000, "end": 1_766_000_010_001},
            },
        )
        self._write_json(
            storage / "part" / "msg_user" / "prt_user_file.json",
            {
                "id": "prt_user_file",
                "type": "file",
                "filename": "PlexPlayer/Shared/PrimaryTabView.swift",
            },
        )

        self._write_json(
            storage / "part" / "msg_assistant" / "prt_assistant_text.json",
            {
                "id": "prt_assistant_text",
                "type": "text",
                "text": (
                    "Implemented a single-player ownership guard and verified playback handoff."
                ),
                "time": {"start": 1_766_000_020_500, "end": 1_766_000_020_501},
            },
        )
        self._write_json(
            storage / "part" / "msg_assistant" / "prt_assistant_tool.json",
            {
                "id": "prt_assistant_tool",
                "type": "tool",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "xcodebuild -scheme PlexPlayer build"},
                    "output": "Build succeeded",
                    "time": {"start": 1_766_000_021_000, "end": 1_766_000_022_000},
                },
            },
        )
        self._write_json(
            storage / "part" / "msg_assistant" / "prt_assistant_patch.json",
            {
                "id": "prt_assistant_patch",
                "type": "patch",
                "hash": "abc123",
                "files": ["PlexPlayer/Features/Player/ViewModels/GlobalPlayerManager.swift"],
            },
        )

        self._write_json(
            storage / "part" / "msg_noise" / "prt_noise_text.json",
            {
                "id": "prt_noise_text",
                "type": "text",
                "text": (
                    "Called the Read tool with the following input: "
                    "{\"filePath\":\"foo.swift\"}"
                ),
            },
        )

        ingester = OpenCodeIngester(project_path=repo_path, opencode_dir=opencode_dir)
        session = ingester.parse_session(session_path)

        assert session.source == "opencode"
        assert session.session_id == "opencode-ses_main"
        assert session.title == "Playback fixes"
        assert session.project_path == repo_path
        assert len(session.messages) == 2

        user_message = session.messages[0]
        assert user_message.role == "user"
        assert "duplicate playback" in user_message.content
        assert "Attached files:" in user_message.content
        assert "PrimaryTabView.swift" in user_message.content

        assistant_message = session.messages[1]
        assert assistant_message.role == "assistant"
        assert "single-player ownership guard" in assistant_message.content
        assert len(assistant_message.tool_calls) == 2
        assert assistant_message.tool_calls[0].tool == "bash"
        assert (
            assistant_message.tool_calls[0].args["command"]
            == "xcodebuild -scheme PlexPlayer build"
        )
        assert assistant_message.tool_calls[0].duration_ms == 1000
        assert assistant_message.tool_calls[1].tool == "patch"


class MockLLM(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        _ = (messages, temperature, max_tokens)
        return LLMResponse(
            content=(
                "[{\"label\": \"pattern\", \"content\": \"Test pattern\", "
                "\"tags\": [\"test\"], \"confidence\": 0.8}]"
            ),
            model="mock",
            usage=None,
        )

    def validate(self) -> tuple[bool, str]:
        return True, "ok"


class TestTranscriptExtractor:
    @pytest.mark.asyncio
    async def test_extract_basic(self) -> None:
        from agent_recall.core.extract import TranscriptExtractor

        extractor = TranscriptExtractor(MockLLM())

        session = RawSession(
            source="test",
            session_id="test-123",
            started_at=datetime.now(UTC),
            messages=[
                RawMessage(
                    role="user",
                    content=(
                        "Do something useful with schema migration ordering and transaction "
                        "safety checks before deploy."
                    ),
                ),
                RawMessage(
                    role="assistant",
                    content=(
                        "Done and verified. I added transaction boundaries, rollback checks, "
                        "and documented the migration sequence."
                    ),
                ),
            ],
        )

        entries = await extractor.extract(session)

        assert len(entries) == 1
        assert entries[0].content == "Test pattern"
        assert entries[0].label.value == "pattern"

    @pytest.mark.asyncio
    async def test_extract_filters_non_functional_workflow_learning(self) -> None:
        from agent_recall.core.extract import TranscriptExtractor

        class WorkflowLLM(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "workflow"

            @property
            def model_name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[Message],
                temperature: float = 0.3,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                _ = (messages, temperature, max_tokens)
                return LLMResponse(
                    content=(
                        "[{\"label\":\"preference\",\"content\":\"Do not modify plan files; mark "
                        "existing to-dos as in_progress first\",\"tags\":[\"workflow\"],"
                        "\"confidence\":0.9}]"
                    ),
                    model="mock",
                )

            def validate(self) -> tuple[bool, str]:
                return True, "ok"

        extractor = TranscriptExtractor(WorkflowLLM())
        session = RawSession(
            source="test",
            session_id="test-456",
            started_at=datetime.now(UTC),
            messages=[
                RawMessage(role="user", content="Follow the plan instructions."),
                RawMessage(role="assistant", content="Done."),
            ],
        )

        entries = await extractor.extract(session)
        assert entries == []

    @pytest.mark.asyncio
    async def test_extract_sanitizes_thinking_tokens_and_code_fences(self) -> None:
        from agent_recall.core.extract import TranscriptExtractor

        class ThinkingLLM(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "thinking"

            @property
            def model_name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[Message],
                temperature: float = 0.3,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                _ = (messages, temperature, max_tokens)
                return LLMResponse(
                    content=(
                        "<think>I should reason here but not output it</think>\n"
                        "```json\n"
                        "[{\"label\":\"pattern\","
                        "\"content\":\"Use retry with backoff for flaky API calls\","
                        "\"tags\":[\"api\",\"reliability\"],"
                        "\"confidence\":0.8,"
                        "\"evidence\":\"Retries stabilized failing calls\"}]\n"
                        "```"
                    ),
                    model="mock",
                )

            def validate(self) -> tuple[bool, str]:
                return True, "ok"

        extractor = TranscriptExtractor(ThinkingLLM())
        session = RawSession(
            source="test",
            session_id="test-thinking",
            started_at=datetime.now(UTC),
            messages=[
                RawMessage(
                    role="user",
                    content=(
                        "Fix flaky API behavior in retries. The issue appears during intermittent "
                        "network failures where responses timeout and must be retried safely."
                    ),
                ),
                RawMessage(
                    role="assistant",
                    content=(
                        "Added retry guards with bounded exponential backoff, timeout handling, "
                        "and idempotent request checks to avoid duplicate side effects."
                    ),
                ),
            ],
        )

        entries = await extractor.extract(session)
        assert len(entries) == 1
        assert entries[0].label.value == "pattern"
        assert "retry with backoff" in entries[0].content

    @pytest.mark.asyncio
    async def test_extract_handles_object_wrapped_payload(self) -> None:
        from agent_recall.core.extract import TranscriptExtractor

        class WrappedLLM(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "wrapped"

            @property
            def model_name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[Message],
                temperature: float = 0.3,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                _ = (messages, temperature, max_tokens)
                return LLMResponse(
                    content=(
                        "Result:\n"
                        "{\"learnings\":[{\"label\":\"gotcha\","
                        "\"content\":\"Cursor ItemTable stores prompt/generation arrays, "
                        "not only chatdata\","
                        "\"tags\":[\"cursor\",\"storage\"],"
                        "\"confidence\":0.8,"
                        "\"evidence\":\"Found aiService.prompts and aiService.generations keys\"}"
                        "]}"
                    ),
                    model="mock",
                )

            def validate(self) -> tuple[bool, str]:
                return True, "ok"

        extractor = TranscriptExtractor(WrappedLLM())
        session = RawSession(
            source="test",
            session_id="test-wrapped",
            started_at=datetime.now(UTC),
            messages=[
                RawMessage(
                    role="user",
                    content=(
                        "Investigate Cursor local storage layout and determine where conversation "
                        "prompts and generations are persisted for ingestion."
                    ),
                ),
                RawMessage(
                    role="assistant",
                    content=(
                        "Mapped ItemTable keys and confirmed aiService payload arrays are the "
                        "source of prompt and generation content for extraction."
                    ),
                ),
            ],
        )

        entries = await extractor.extract(session)
        assert len(entries) == 1
        assert entries[0].label.value == "gotcha"
        assert "ItemTable" in entries[0].content

    @pytest.mark.asyncio
    async def test_extract_batches_large_sessions_and_reports_progress(self) -> None:
        from agent_recall.core.extract import TranscriptExtractor

        class BatchedLLM(LLMProvider):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def provider_name(self) -> str:
                return "batched"

            @property
            def model_name(self) -> str:
                return "mock"

            async def generate(
                self,
                messages: list[Message],
                temperature: float = 0.3,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                _ = (messages, temperature, max_tokens)
                self.calls += 1
                return LLMResponse(
                    content=(
                        "[{\"label\":\"pattern\",\"content\":\"Batch learning "
                        + str(self.calls)
                        + "\",\"tags\":[\"batch\"],\"confidence\":0.8}]"
                    ),
                    model="mock",
                )

            def validate(self) -> tuple[bool, str]:
                return True, "ok"

        llm = BatchedLLM()
        extractor = TranscriptExtractor(llm, messages_per_batch=100)
        session = RawSession(
            source="test",
            session_id="test-batched",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            messages=[
                RawMessage(
                    role="user" if index % 2 == 0 else "assistant",
                    content=(
                        f"Message {index + 1}: update migration ordering, add retry guards, "
                        "and validate rollback safety for transaction boundaries."
                    ),
                )
                for index in range(205)
            ],
        )

        progress_events: list[dict[str, object]] = []
        entries = await extractor.extract(session, progress_callback=progress_events.append)

        assert llm.calls == 3
        assert len(entries) == 3
        assert len(progress_events) == 3
        assert progress_events[0]["messages_processed"] == 100
        assert progress_events[1]["messages_processed"] == 200
        assert progress_events[2]["messages_processed"] == 205
        assert progress_events[2]["messages_total"] == 205
        assert progress_events[2]["batch_count"] == 3
