from __future__ import annotations

from agent_recall.llm.base import LLMProvider, Message
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel, SessionStatus
from agent_recall.storage.sqlite import SQLiteStorage

GUARDRAILS_PROMPT = """You are analyzing development session logs
to extract hard rules and warnings.

Current GUARDRAILS.md:
{current_guardrails}

New log entries:
{entries}

Extract any NEW guardrails not already in the current file. A guardrail is:
- A hard failure: "Never do X because Y"
- A gotcha: "Watch out for X, it causes Y"
- A correction: "Don't do X, do Y instead"

Output format (one per line):
- [FAILURE] Description with why
- [GOTCHA] Description with why
- [CORRECTION] Don't X, do Y instead

If no new guardrails, output exactly: NONE"""

STYLE_PROMPT = """You are analyzing development session logs
to extract coding patterns and preferences.

Current STYLE.md:
{current_style}

New log entries:
{entries}

Extract any NEW style guidelines not already in the current file:
- Preferences: "Prefer X over Y"
- Patterns: "Use X pattern for Y"

Output format (one per line):
- [PREFERENCE] Description
- [PATTERN] Description

If no new guidelines, output exactly: NONE"""

RECENT_PROMPT = """Summarize these development sessions for quick reference.

Sessions:
{sessions}

For each session, output ONE line in this format:
**YYYY-MM-DD**: 1-2 sentence summary of task and outcome

Be concise."""


class CompactionEngine:
    def __init__(self, storage: SQLiteStorage, files: FileStorage, llm: LLMProvider):
        self.storage = storage
        self.files = files
        self.llm = llm

    async def compact(self, force: bool = False) -> dict[str, bool | int]:
        """Run compaction and return summary details."""
        _ = force  # Reserved for threshold-based behavior in a later version.

        results: dict[str, bool | int] = {
            "guardrails_updated": False,
            "style_updated": False,
            "recent_updated": False,
            "chunks_indexed": 0,
        }

        guardrail_labels = [
            SemanticLabel.HARD_FAILURE,
            SemanticLabel.GOTCHA,
            SemanticLabel.CORRECTION,
        ]
        style_labels = [SemanticLabel.PREFERENCE, SemanticLabel.PATTERN]

        guardrail_entries = self.storage.get_entries_by_label(guardrail_labels)
        style_entries = self.storage.get_entries_by_label(style_labels)

        if guardrail_entries:
            current = self.files.read_tier(KnowledgeTier.GUARDRAILS)
            entries_text = "\n".join(f"- [{e.label.value}] {e.content}" for e in guardrail_entries)
            response = await self.llm.generate(
                [
                    Message(
                        role="user",
                        content=GUARDRAILS_PROMPT.format(
                            current_guardrails=current or "(empty)",
                            entries=entries_text,
                        ),
                    )
                ]
            )

            if response.content.strip() != "NONE":
                update = response.content.strip()
                new_content = (
                    f"{current.rstrip()}\n\n{update}"
                    if current
                    else update
                )
                self.files.write_tier(KnowledgeTier.GUARDRAILS, new_content)
                results["guardrails_updated"] = True

        if style_entries:
            current = self.files.read_tier(KnowledgeTier.STYLE)
            entries_text = "\n".join(f"- [{e.label.value}] {e.content}" for e in style_entries)
            response = await self.llm.generate(
                [
                    Message(
                        role="user",
                        content=STYLE_PROMPT.format(
                            current_style=current or "(empty)",
                            entries=entries_text,
                        ),
                    )
                ]
            )

            if response.content.strip() != "NONE":
                update = response.content.strip()
                new_content = (
                    f"{current.rstrip()}\n\n{update}"
                    if current
                    else update
                )
                self.files.write_tier(KnowledgeTier.STYLE, new_content)
                results["style_updated"] = True

        completed_sessions = self.storage.list_sessions(limit=20, status=SessionStatus.COMPLETED)
        if completed_sessions:
            session_lines = []
            for session in completed_sessions:
                date = session.ended_at.date().isoformat() if session.ended_at else "unknown-date"
                summary = session.summary or "No summary provided"
                session_lines.append(f"- {date}: task={session.task}; summary={summary}")

            response = await self.llm.generate(
                [
                    Message(
                        role="user",
                        content=RECENT_PROMPT.format(sessions="\n".join(session_lines)),
                    )
                ]
            )

            if response.content.strip():
                current_recent = self.files.read_tier(KnowledgeTier.RECENT)
                if current_recent.strip() != response.content.strip():
                    self.files.write_tier(KnowledgeTier.RECENT, response.content.strip())
                    results["recent_updated"] = True

        all_entries = guardrail_entries + style_entries
        for entry in all_entries:
            chunk = Chunk(
                source=ChunkSource.LOG_ENTRY,
                source_ids=[entry.id],
                content=entry.content,
                label=entry.label,
                tags=entry.tags,
            )
            self.storage.store_chunk(chunk)
            results["chunks_indexed"] = int(results["chunks_indexed"]) + 1

        return results
