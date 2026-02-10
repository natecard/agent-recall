from __future__ import annotations

from pathlib import Path

from agent_recall.ingest.base import RawMessage, RawSession, RawToolCall, SessionIngester
from agent_recall.ingest.claude_code import ClaudeCodeIngester
from agent_recall.ingest.cursor import CursorIngester

__all__ = [
    "SessionIngester",
    "RawSession",
    "RawMessage",
    "RawToolCall",
    "CursorIngester",
    "ClaudeCodeIngester",
    "get_default_ingesters",
    "get_ingester",
]


def get_default_ingesters(
    project_path: Path | None = None,
    cursor_db_path: Path | None = None,
    workspace_storage_dir: Path | None = None,
    cursor_all_workspaces: bool = False,
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
    ]


def get_ingester(
    source: str,
    project_path: Path | None = None,
    cursor_db_path: Path | None = None,
    workspace_storage_dir: Path | None = None,
    cursor_all_workspaces: bool = False,
) -> SessionIngester:
    """Return a specific ingester by source name."""
    match source.lower():
        case "cursor":
            return CursorIngester(
                project_path=project_path,
                cursor_db_path=cursor_db_path,
                workspace_storage_dir=workspace_storage_dir,
                include_all_workspaces=cursor_all_workspaces,
            )
        case "claude-code" | "claudecode" | "claude_code":
            return ClaudeCodeIngester(project_path)
        case _:
            raise ValueError(
                f"Unknown ingester: {source}. Available: cursor, claude-code"
            )
