from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console


@dataclass(frozen=True)
class SyncRequest:
    compact: bool
    skip_embeddings: bool
    force_compact: bool
    since: datetime | None
    source: str | None
    selected_sources: list[str] | None
    sources: list[str] | None
    session_ids: list[str] | None
    max_sessions: int | None


@dataclass(frozen=True)
class StatusSnapshot:
    lines: list[str]


def build_sync_request(
    *,
    compact: bool,
    skip_embeddings: bool,
    force_compact: bool,
    source: str | None,
    since_days: int | None,
    session_id: list[str] | None,
    max_sessions: int | None,
    normalized_source: str | None,
    selected_sources: list[str] | None,
) -> SyncRequest:
    since = datetime.now(UTC) - timedelta(days=since_days) if since_days else None
    sources = [normalized_source] if normalized_source else selected_sources
    selected_session_ids = [item.strip() for item in (session_id or []) if item.strip()]
    return SyncRequest(
        compact=compact,
        skip_embeddings=skip_embeddings,
        force_compact=force_compact,
        since=since,
        source=source,
        selected_sources=selected_sources,
        sources=sources,
        session_ids=selected_session_ids or None,
        max_sessions=max_sessions,
    )


def run_sync(auto_sync, request: SyncRequest) -> dict[str, Any]:
    if request.compact:
        if request.skip_embeddings:
            return asyncio.run(
                auto_sync.sync_and_compact(
                    since=request.since,
                    sources=request.sources,
                    session_ids=request.session_ids,
                    max_sessions=request.max_sessions,
                    force_compact=request.force_compact,
                    skip_embeddings=True,
                )
            )
        return asyncio.run(
            auto_sync.sync_and_compact(
                since=request.since,
                sources=request.sources,
                session_ids=request.session_ids,
                max_sessions=request.max_sessions,
                force_compact=request.force_compact,
            )
        )

    return asyncio.run(
        auto_sync.sync(
            since=request.since,
            sources=request.sources,
            session_ids=request.session_ids,
            max_sessions=request.max_sessions,
        )
    )


def render_sync_summary(
    *,
    console: Console,
    results: dict[str, Any],
    request: SyncRequest,
    verbose: bool,
) -> None:
    lines = [""]
    if request.selected_sources and not request.source:
        lines.append(f"Using configured sources: {', '.join(request.selected_sources)}")
    if request.session_ids:
        lines.append(f"Requested session IDs: {len(request.session_ids)}")
    if request.max_sessions is not None:
        lines.append(f"Max sessions: {request.max_sessions}")
    lines.append(f"Sessions discovered: {results['sessions_discovered']}")
    lines.append(f"Sessions processed:  {results['sessions_processed']}")
    if int(results["sessions_skipped"]) > 0:
        lines.append(
            f"Sessions skipped:    {results['sessions_skipped']} "
            f"(already processed: {int(results.get('sessions_already_processed', 0))}, "
            f"empty: {int(results.get('empty_sessions', 0))})"
        )
    lines.append(f"Learnings extracted: {results['learnings_extracted']}")
    lines.append(f"LLM extraction requests: {results.get('llm_requests', 0)}")
    if request.session_ids and int(results["sessions_processed"]) == 0:
        already_processed = int(results.get("sessions_already_processed", 0))
        if already_processed == int(results["sessions_discovered"]) and already_processed > 0:
            lines.append(
                "[warning]Selected sessions were already processed; ingestion did not "
                "rerun.[/warning]"
            )
            lines.append(
                "[dim]Use `agent-recall sync reset --session-id <id>` to reprocess a session.[/dim]"
            )

    by_source = results.get("by_source", {})
    if by_source:
        lines.append("")
        lines.append("By source:")
        for source_name in sorted(by_source):
            values = by_source[source_name]
            lines.append(
                f"  {source_name}: {values.get('sessions_processed', 0)} sessions, "
                f"{values.get('learnings_extracted', 0)} learnings"
            )

    errors = results.get("errors", [])
    if errors:
        lines.append("")
        lines.append(f"[warning]Warnings: {len(errors)}[/warning]")
        for error in errors[:10]:
            lines.append(f"[dim]- {error}[/dim]")
        if len(errors) > 10:
            lines.append(f"[dim]... and {len(errors) - 10} more[/dim]")

    if verbose:
        lines.append("")
        lines.append(f"Raw results: {results}")

    console.print("\n".join(lines))
