from __future__ import annotations

from pathlib import Path

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester
from agent_recall.ingest.claude_code import ClaudeCodeIngester
from agent_recall.ingest.codex import CodexIngester
from agent_recall.ingest.cursor import CursorIngester
from agent_recall.ingest.opencode import OpenCodeIngester
from agent_recall.ingest.sources import VALID_SOURCE_NAMES, normalize_source_name

__all__ = [
    "SessionIngester",
    "RawSession",
    "RawMessage",
    "RawToolCall",
    "CursorIngester",
    "ClaudeCodeIngester",
    "OpenCodeIngester",
    "CodexIngester",
    "VALID_SOURCE_NAMES",
    "normalize_source_name",
    "get_default_ingesters",
    "get_ingester",
]


def get_default_ingesters(
    project_path: Path | None = None,
    cursor_db_path: Path | None = None,
    workspace_storage_dir: Path | None = None,
    cursor_all_workspaces: bool = False,
    opencode_dir: Path | None = None,
    codex_dir: Path | None = None,
) -> list[SessionIngester]:
    """Return ingesters for all supported native session sources."""
    return [
        CursorIngester(
            project_path=project_path,
            cursor_db_path=cursor_db_path,
            workspace_storage_dir=workspace_storage_dir,
            include_all_workspaces=cursor_all_workspaces,
        ),
        ClaudeCodeIngester(project_path),
        OpenCodeIngester(project_path=project_path, opencode_dir=opencode_dir),
        CodexIngester(project_path=project_path, codex_dir=codex_dir),
    ]


def get_ingester(
    source: str,
    project_path: Path | None = None,
    cursor_db_path: Path | None = None,
    workspace_storage_dir: Path | None = None,
    cursor_all_workspaces: bool = False,
    opencode_dir: Path | None = None,
    codex_dir: Path | None = None,
) -> SessionIngester:
    """Return a specific ingester by source name."""
    normalized = normalize_source_name(source)
    factories = {
        "cursor": lambda: CursorIngester(
            project_path=project_path,
            cursor_db_path=cursor_db_path,
            workspace_storage_dir=workspace_storage_dir,
            include_all_workspaces=cursor_all_workspaces,
        ),
        "claude-code": lambda: ClaudeCodeIngester(project_path),
        "opencode": lambda: OpenCodeIngester(
            project_path=project_path,
            opencode_dir=opencode_dir,
        ),
        "codex": lambda: CodexIngester(
            project_path=project_path,
            codex_dir=codex_dir,
        ),
    }
    factory = factories.get(normalized)
    if factory is not None:
        return factory()
    available = ", ".join(VALID_SOURCE_NAMES)
    raise ValueError(f"Unknown ingester: {source}. Available: {available}")
