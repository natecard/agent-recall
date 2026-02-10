from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.context import ContextAssembler
from agent_recall.core.ingest import TranscriptIngestor
from agent_recall.core.log import LogWriter
from agent_recall.core.retrieve import Retriever
from agent_recall.core.session import SessionManager
from agent_recall.llm import create_llm_provider
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LLMConfig, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage

app = typer.Typer(help="Agent Memory System - Persistent knowledge for AI coding agents")
console = Console()

AGENT_DIR = Path(".agent")
DB_PATH = AGENT_DIR / "state.db"

INITIAL_GUARDRAILS = """# Guardrails

Rules and warnings for this codebase. Entries added automatically from agent sessions.
"""

INITIAL_STYLE = """# Style Guide

Coding patterns and preferences for this codebase. Entries added automatically from agent sessions.
"""

INITIAL_RECENT = """# Recent Sessions

Summaries of recent agent sessions.
"""

INITIAL_CONFIG = """# Agent Memory Configuration

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514

compaction:
  max_recent_tokens: 1500
  max_sessions_before_compact: 5
  promote_pattern_after_occurrences: 3
  archive_sessions_older_than_days: 30

retrieval:
  backend: fts5
  top_k: 5
"""


def get_storage() -> SQLiteStorage:
    if not AGENT_DIR.exists():
        console.print("[red]Not initialized. Run 'agent-recall init' first.[/red]")
        raise typer.Exit(1)
    return SQLiteStorage(DB_PATH)


def get_files() -> FileStorage:
    return FileStorage(AGENT_DIR)


@app.command()
def init(force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing")):
    """Initialize agent memory in the current repository."""
    if AGENT_DIR.exists() and not force:
        console.print("[yellow].agent/ already exists. Use --force to reinitialize.[/yellow]")
        raise typer.Exit(1)

    if AGENT_DIR.exists() and force:
        shutil.rmtree(AGENT_DIR)

    AGENT_DIR.mkdir(exist_ok=True)
    (AGENT_DIR / "logs").mkdir(exist_ok=True)
    (AGENT_DIR / "archive").mkdir(exist_ok=True)

    (AGENT_DIR / "GUARDRAILS.md").write_text(INITIAL_GUARDRAILS)
    (AGENT_DIR / "STYLE.md").write_text(INITIAL_STYLE)
    (AGENT_DIR / "RECENT.md").write_text(INITIAL_RECENT)
    (AGENT_DIR / "config.yaml").write_text(INITIAL_CONFIG)

    SQLiteStorage(DB_PATH)

    console.print(
        Panel.fit(
            "[green]✓ Initialized .agent/ directory[/green]\n\n"
            "Files created:\n"
            "  • config.yaml - Configuration\n"
            "  • GUARDRAILS.md - Hard rules and warnings\n"
            "  • STYLE.md - Patterns and preferences\n"
            "  • RECENT.md - Session summaries\n"
            "  • state.db - Session and log storage",
            title="agent-recall",
        )
    )


@app.command()
def start(task: str = typer.Argument(..., help="Description of what this session is working on")):
    """Start a new session and output context for the agent."""
    storage = get_storage()
    files = get_files()

    session_mgr = SessionManager(storage)
    context_asm = ContextAssembler(storage, files)

    session = session_mgr.start(task)
    context = context_asm.assemble(task=task)

    console.print(f"[dim]Session started: {session.id}[/dim]\n")
    console.print(context)


@app.command()
def log(
    content: str = typer.Argument(..., help="The observation or learning to log"),
    label: str = typer.Option(..., "--label", "-l", help="Semantic label"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Log an observation or learning from the current session."""
    storage = get_storage()
    session_mgr = SessionManager(storage)
    log_writer = LogWriter(storage)

    try:
        semantic_label = SemanticLabel(label)
    except ValueError:
        valid = ", ".join(item.value for item in SemanticLabel)
        console.print(f"[red]Invalid label '{label}'. Valid: {valid}[/red]")
        raise typer.Exit(1) from None

    active = session_mgr.get_active()
    session_id = active.id if active else None

    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

    log_writer.log(
        content=content,
        label=semantic_label,
        session_id=session_id,
        tags=tag_list,
    )

    session_note = f" (session: {session_id})" if session_id else " (no active session)"
    console.print(f"[green]✓ Logged [{label}]{session_note}[/green]")


@app.command()
def end(summary: str = typer.Argument(..., help="Summary of what was accomplished")):
    """End the current session."""
    storage = get_storage()
    session_mgr = SessionManager(storage)

    active = session_mgr.get_active()
    if not active:
        console.print("[yellow]No active session to end.[/yellow]")
        raise typer.Exit(1)

    session = session_mgr.end(active.id, summary)
    console.print(f"[green]✓ Session ended: {session.id}[/green]")
    console.print(f"[dim]Summary: {summary}[/dim]")
    console.print(f"[dim]Entries logged: {session.entry_count}[/dim]")


@app.command()
def context(
    task: str | None = typer.Option(None, "--task", "-t", help="Task for relevant retrieval"),
    output_format: str = typer.Option("md", "--format", "-f", help="Output format: md or json"),
):
    """Output current context (guardrails, style, recent, relevant)."""
    storage = get_storage()
    files = get_files()

    context_asm = ContextAssembler(storage, files)
    output = context_asm.assemble(task=task)

    if output_format == "md":
        console.print(output)
        return

    if output_format == "json":
        payload = {"task": task, "context": output}
        console.print(json.dumps(payload, indent=2))
        return

    console.print("[red]Invalid format. Use 'md' or 'json'.[/red]")
    raise typer.Exit(1)


@app.command()
def compact(force: bool = typer.Option(False, "--force", "-f", help="Force compaction")):
    """Run compaction to update knowledge tiers from logs."""
    storage = get_storage()
    files = get_files()

    config_dict = files.read_config()
    llm_config = LLMConfig(**config_dict.get("llm", {}))

    try:
        llm = create_llm_provider(llm_config)
    except ValueError as exc:
        console.print(f"[red]LLM configuration error: {exc}[/red]")
        raise typer.Exit(1) from None

    engine = CompactionEngine(storage, files, llm)

    console.print("[dim]Running compaction...[/dim]")
    results = asyncio.run(engine.compact(force=force))

    console.print(
        Panel.fit(
            f"[green]✓ Compaction complete[/green]\n\n"
            f"Guardrails updated: {results['guardrails_updated']}\n"
            f"Style updated: {results['style_updated']}\n"
            f"Recent updated: {results['recent_updated']}\n"
            f"Chunks indexed: {results['chunks_indexed']}",
            title="Results",
        )
    )


@app.command()
def status():
    """Show current agent memory status."""
    storage = get_storage()
    files = get_files()
    session_mgr = SessionManager(storage)

    active = session_mgr.get_active()

    guardrails = files.read_tier(KnowledgeTier.GUARDRAILS)
    style = files.read_tier(KnowledgeTier.STYLE)
    recent = files.read_tier(KnowledgeTier.RECENT)

    lines: list[str] = []

    if active:
        lines.append(f"[green]Active session:[/green] {active.task}")
        lines.append(f"  Started: {active.started_at}")
        lines.append(f"  Entries: {active.entry_count}")
    else:
        lines.append("[dim]No active session[/dim]")

    lines.append("")
    lines.append(f"GUARDRAILS.md: {len(guardrails)} chars")
    lines.append(f"STYLE.md: {len(style)} chars")
    lines.append(f"RECENT.md: {len(recent)} chars")
    lines.append(f"Log entries: {storage.count_log_entries()}")
    lines.append(f"Chunks: {storage.count_chunks()}")

    console.print(Panel.fit("\n".join(lines), title="Agent Memory Status"))


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Maximum results"),
):
    """Retrieve relevant memory chunks using FTS5 search."""
    storage = get_storage()
    retriever = Retriever(storage)

    chunks = retriever.search(query=query, top_k=top_k)
    if not chunks:
        console.print("[yellow]No matching chunks found.[/yellow]")
        raise typer.Exit(0)

    for idx, chunk in enumerate(chunks, 1):
        tags = f" [{', '.join(chunk.tags)}]" if chunk.tags else ""
        console.print(f"{idx}. ({chunk.label.value}) {chunk.content}{tags}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    source_session_id: str | None = typer.Option(
        None,
        "--source-session-id",
        help="Native session ID",
    ),
):
    """Ingest a JSONL transcript into log entries."""
    storage = get_storage()
    ingestor = TranscriptIngestor(storage)

    count = ingestor.ingest_jsonl(path=path, source_session_id=source_session_id)
    console.print(f"[green]✓ Ingested {count} transcript entries[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
