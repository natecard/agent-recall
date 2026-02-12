from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from agent_recall.ingest.codex import CodexIngester


class TestCodexFixtures:
    """Test Codex parser against fixture corpus with various edge cases."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        """Return the path to Codex fixtures directory."""
        return Path(__file__).parent / "fixtures" / "codex"

    @pytest.fixture
    def temp_project(self, tmp_path: Path) -> Path:
        """Create a temporary project directory."""
        project = tmp_path / "project"
        project.mkdir()
        return project

    def test_valid_jsonl_parses_correctly(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Valid JSONL session should parse with all messages and tool calls."""
        fixture_path = fixtures_dir / "valid_session.jsonl"

        # Copy to a temp codex sessions directory with proper structure
        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "valid_session.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        assert session.source == "codex"
        assert "valid-session-001" in session.session_id
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "authentication" in session.messages[0].content
        assert session.messages[1].role == "assistant"
        assert len(session.messages[1].tool_calls) == 1
        assert session.messages[1].tool_calls[0].tool == "read_file"
        assert session.messages[1].tool_calls[0].result is not None

    def test_legacy_json_parses_correctly(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Legacy JSON format should parse correctly."""
        fixture_path = fixtures_dir / "legacy_session.json"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "legacy_session.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        assert session.source == "codex"
        assert "legacy-session-001" in session.session_id
        assert len(session.messages) == 2
        assert "traceback" in session.messages[0].content.lower()

    def test_minimal_content_parses(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Minimal content variations should parse without errors."""
        fixture_path = fixtures_dir / "minimal_content.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "minimal_content.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        # Short "Hi" message is filtered (under 3 chars), only assistant message remains
        assert len(session.messages) == 1
        assert "Hello" in session.messages[0].content

    def test_custom_tools_parses(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Custom tool calls should be parsed and linked correctly."""
        fixture_path = fixtures_dir / "custom_tools.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "custom_tools.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        assert len(session.messages) == 2
        assistant_msg = session.messages[1]
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0].tool == "apply_patch"
        assert "success" in str(assistant_msg.tool_calls[0].result).lower()

    def test_various_timestamps_parsed(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Various timestamp formats should be handled correctly."""
        fixture_path = fixtures_dir / "various_timestamps.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "various_timestamps.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        # Should parse without errors and have correct message count
        assert len(session.messages) >= 3
        assert session.started_at is not None
        assert isinstance(session.started_at, datetime)

    def test_malformed_lines_gracefully_handled(
        self, fixtures_dir: Path, temp_project: Path
    ) -> None:
        """Malformed JSON lines should not crash parser."""
        fixture_path = fixtures_dir / "malformed_lines.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "malformed_lines.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)

        # Should not raise an exception
        session = ingester.parse_session(session_path)

        # Should still extract valid messages
        assert len(session.messages) >= 1
        assert any("still work" in msg.content for msg in session.messages)

    def test_missing_fields_graceful(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Missing optional fields should not crash parser."""
        fixture_path = fixtures_dir / "missing_fields.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "missing_fields.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)

        # Should not raise an exception
        session = ingester.parse_session(session_path)

        # Should extract the valid assistant message
        assert any(msg.role == "assistant" for msg in session.messages)

    def test_tool_ordering_handled(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Out-of-order tool calls should be handled correctly."""
        fixture_path = fixtures_dir / "tool_ordering.jsonl"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "tool_ordering.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        # Should have user and assistant messages
        assert any(msg.role == "user" for msg in session.messages)
        assert any(msg.role == "assistant" for msg in session.messages)

    def test_legacy_minimal_parses(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Minimal legacy JSON should parse."""
        fixture_path = fixtures_dir / "legacy_minimal.json"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "legacy_minimal.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        assert len(session.messages) == 2
        assert session.messages[0].content == "Simple message"

    def test_legacy_missing_fields_graceful(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Legacy JSON with missing fields should be handled gracefully."""
        fixture_path = fixtures_dir / "legacy_missing_fields.json"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "legacy_missing_fields.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)

        # Should not raise an exception
        session = ingester.parse_session(session_path)

        # Should still extract valid messages
        assert any("Structured content" in msg.content for msg in session.messages)

    def test_malformed_legacy_returns_error(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Malformed legacy JSON should raise appropriate error."""
        fixture_path = fixtures_dir / "malformed_legacy.json"

        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "malformed_legacy.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_path, session_path)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)

        # Should raise ValueError for invalid JSON
        with pytest.raises(ValueError, match="Invalid Codex session file"):
            ingester.parse_session(session_path)

    def test_empty_jsonl_file(self, temp_project: Path) -> None:
        """Empty JSONL file should be handled gracefully."""
        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "empty.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text("")

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        assert session.messages == []

    def test_partial_line_at_end(self, fixtures_dir: Path, temp_project: Path) -> None:
        """Partial/truncated line at end should not crash parser."""
        codex_dir = temp_project.parent / ".codex"
        session_path = codex_dir / "sessions" / "partial_end.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)

        # Write valid content followed by partial line
        content = (
            json.dumps(
                {
                    "timestamp": "2026-02-12T17:00:00Z",
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": "Valid message"},
                }
            )
            + '\n{"partial": "line without closing'
        )

        session_path.write_text(content)

        ingester = CodexIngester(project_path=temp_project, codex_dir=codex_dir)
        session = ingester.parse_session(session_path)

        # Should parse the valid line
        assert len(session.messages) == 1
        assert "Valid message" in session.messages[0].content


class TestCodexDiscoveryAndSync:
    """Test Codex source discovery and session management."""

    def test_discover_sessions_filters_by_project(self, tmp_path: Path) -> None:
        """Session discovery should only return sessions matching the project."""
        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        project_b.mkdir()

        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"

        # Create session for project A
        session_a = sessions_dir / "2026" / "02" / "12" / "session_a.jsonl"
        session_a.parent.mkdir(parents=True, exist_ok=True)
        session_a.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-12T10:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "session-a",
                        "cwd": str(project_a),
                        "timestamp": "2026-02-12T10:00:00Z",
                    },
                }
            )
            + "\n"
        )

        # Create session for project B
        session_b = sessions_dir / "2026" / "02" / "12" / "session_b.jsonl"
        session_b.parent.mkdir(parents=True, exist_ok=True)
        session_b.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-12T10:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "session-b",
                        "cwd": str(project_b),
                        "timestamp": "2026-02-12T10:00:00Z",
                    },
                }
            )
            + "\n"
        )

        ingester_a = CodexIngester(project_path=project_a, codex_dir=codex_dir)
        sessions_a = ingester_a.discover_sessions()

        assert len(sessions_a) == 1
        assert sessions_a[0].name == "session_a.jsonl"

    def test_discover_sessions_respects_since_parameter(self, tmp_path: Path) -> None:
        """Session discovery should filter by since timestamp."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"

        # Create old session
        old_session = sessions_dir / "old.jsonl"
        old_session.parent.mkdir(parents=True, exist_ok=True)
        old_session.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-10T10:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "old-session",
                        "cwd": str(project),
                        "timestamp": "2026-02-10T10:00:00Z",
                    },
                }
            )
            + "\n"
        )

        # Create recent session
        new_session = sessions_dir / "new.jsonl"
        new_session.parent.mkdir(parents=True, exist_ok=True)
        new_session.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-12T10:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "new-session",
                        "cwd": str(project),
                        "timestamp": "2026-02-12T10:00:00Z",
                    },
                }
            )
            + "\n"
        )

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)
        since = datetime.fromisoformat("2026-02-11T00:00:00+00:00")
        sessions = ingester.discover_sessions(since=since)

        assert len(sessions) == 1
        assert sessions[0].name == "new.jsonl"

    def test_session_id_fallback_to_filename(self, tmp_path: Path) -> None:
        """Session ID should fallback to filename when no meta available."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        session_path = codex_dir / "sessions" / "my_session.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-12T10:00:00Z",
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": "Test"},
                }
            )
            + "\n"
        )

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)
        session_id = ingester.get_session_id(session_path)

        assert "my_session" in session_id

    def test_get_session_id_from_meta(self, tmp_path: Path) -> None:
        """Session ID should use meta id when available."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        session_path = codex_dir / "sessions" / "file_name.jsonl"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-02-12T10:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": "meta-session-id",
                        "cwd": str(project),
                        "timestamp": "2026-02-12T10:00:00Z",
                    },
                }
            )
            + "\n"
        )

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)
        session_id = ingester.get_session_id(session_path)

        assert "meta-session-id" in session_id
        assert "file_name" not in session_id


class TestCodexErrorHandling:
    """Test Codex parser error handling and diagnostics."""

    def test_nonexistent_session_directory(self, tmp_path: Path) -> None:
        """Non-existent sessions directory should return empty list."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)

        sessions = ingester.discover_sessions()
        assert sessions == []

    def test_unsupported_file_extension(self, tmp_path: Path) -> None:
        """Unsupported file extensions should be ignored during discovery."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)

        # Create various file types
        (sessions_dir / "valid.jsonl").write_text(
            '{"type":"session_meta","payload":{"id":"v","cwd":"' + str(project) + '"}}\n'
        )
        (sessions_dir / "valid.json").write_text(
            '{"session":{"id":"v","cwd":"' + str(project) + '"}}'
        )
        (sessions_dir / "invalid.txt").write_text("not a session")
        (sessions_dir / "invalid.yaml").write_text("not: a session")

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)
        sessions = ingester.discover_sessions()

        assert len(sessions) == 2
        assert all(s.suffix in {".json", ".jsonl"} for s in sessions)

    def test_empty_legacy_json(self, tmp_path: Path) -> None:
        """Empty legacy JSON should raise ValueError."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        session_path = codex_dir / "sessions" / "empty.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text("")

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)

        with pytest.raises(ValueError, match="Invalid Codex session file"):
            ingester.parse_session(session_path)

    def test_invalid_legacy_json_structure(self, tmp_path: Path) -> None:
        """Invalid legacy JSON structure should raise ValueError."""
        project = tmp_path / "project"
        project.mkdir()

        codex_dir = tmp_path / ".codex"
        session_path = codex_dir / "sessions" / "invalid.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text('{"not": "a valid session"}')

        ingester = CodexIngester(project_path=project, codex_dir=codex_dir)

        # Should not crash, but return empty messages
        session = ingester.parse_session(session_path)
        assert session.messages == []
