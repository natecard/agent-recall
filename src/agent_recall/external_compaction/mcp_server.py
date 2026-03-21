from __future__ import annotations

import importlib
from typing import Any

from agent_recall.external_compaction.service import ExternalCompactionService, WriteTarget


def run_external_compaction_mcp(
    service: ExternalCompactionService,
    *,
    write_target: WriteTarget = "runtime",
) -> None:
    """Run an MCP server that exposes external compaction tools."""
    try:
        fastmcp_module = importlib.import_module("mcp.server.fastmcp")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MCP support requires the optional 'mcp' dependency. "
            'Install with: pip install "agent-recall[mcp]"'
        ) from exc

    fastmcp_cls = getattr(fastmcp_module, "FastMCP")
    server = fastmcp_cls("agent-recall-external-compaction")

    @server.tool()
    def list_imported_conversations(
        limit: int = 20,
        pending_only: bool = True,
    ) -> dict[str, Any]:
        conversations = service.list_imported_conversations(limit=limit, pending_only=pending_only)
        return {"conversations": conversations}

    @server.tool()
    def get_imported_conversation(source_session_id: str, limit: int = 300) -> dict[str, Any]:
        return service.get_conversation(source_session_id, limit=limit)

    @server.tool()
    def read_recall_tiers(target: WriteTarget = write_target) -> dict[str, str]:
        return service.read_tiers(write_target=target)

    @server.tool()
    def build_compaction_payload(
        limit: int = 20,
        pending_only: bool = True,
        entry_limit: int = 300,
        target: WriteTarget = write_target,
    ) -> dict[str, Any]:
        return service.build_payload(
            limit=limit,
            pending_only=pending_only,
            entry_limit=entry_limit,
            write_target=target,
        )

    @server.tool()
    def apply_compaction_notes(
        notes: list[dict[str, Any]],
        target: WriteTarget = write_target,
        dry_run: bool = False,
        mark_processed: bool = True,
    ) -> dict[str, Any]:
        return service.apply_notes_payload(
            {"notes": notes},
            write_target=target,
            dry_run=dry_run,
            mark_processed=mark_processed,
        )

    server.run()
