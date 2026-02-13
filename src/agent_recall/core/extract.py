from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agent_recall.ingest.base import RawMessage, RawSession
from agent_recall.llm.base import LLMProvider, Message
from agent_recall.storage.models import CurationStatus, LogEntry, LogSource, SemanticLabel

EXTRACTION_SYSTEM_PROMPT = """You are analyzing a development session transcript
to extract learnings that will help future AI agents working on this codebase.

Your job is to identify valuable insights in these categories:

1. hard_failure: Things that definitively broke or didn't work
2. gotcha: Non-obvious issues or quirks discovered
3. correction: When the user corrected the agent's approach
4. preference: Implicit or explicit team/codebase preferences
5. pattern: Useful patterns that worked well
6. decision: Significant architectural or design decisions with rationale

RULES:
- Focus on knowledge that helps future agents avoid mistakes or work more effectively
- Be specific and actionable
- Include the "why" when possible
- Do not extract routine operations
- Do not extract workflow/process instructions (task status updates, ticket/plan handling)
- Never include chain-of-thought, reasoning traces, or commentary in output
- If there are no meaningful learnings, return an empty array

For confidence scoring:
- High (0.9): Explicit statement or direct feedback
- Medium (0.7): Inferred from behavior or context
- Low (0.5): Tentative observation"""

EXTRACTION_USER_PROMPT = """Analyze this development session transcript and extract learnings.

Session source: {source}
Project: {project_path}
Date: {date}
Duration: {duration}
Segment: {segment}

=== TRANSCRIPT START ===
{transcript}
=== TRANSCRIPT END ===

Output as a JSON array. Each item must have:
{{
  "label": "<hard_failure|gotcha|correction|preference|pattern|decision>",
  "content": "<concise, actionable description>",
  "tags": ["<relevant>", "<tags>"],
  "confidence": <0.5-1.0>,
  "evidence": "<brief quote or reference>"
}}

If there are no meaningful learnings, output exactly: []
Do not include markdown fences, prose, or thinking tags.
Return only raw JSON.

JSON array:"""


class TranscriptExtractor:
    """Extract semantic learnings from normalized session transcripts."""

    def __init__(self, llm: LLMProvider, messages_per_batch: int = 100):
        self.llm = llm
        self.messages_per_batch = max(1, int(messages_per_batch))

    def _format_transcript(self, session: RawSession, max_chars: int = 5_000) -> str:
        lines: list[str] = []

        for message in session.messages:
            ts = ""
            if message.timestamp:
                ts = f" [{message.timestamp.strftime('%H:%M')}]"

            role_display = "USER" if message.role == "user" else "ASSISTANT"
            lines.append(f"### {role_display}{ts}")
            lines.append("")
            lines.append(message.content)

            for tool_call in message.tool_calls:
                status = "OK" if tool_call.success else "ERR"
                lines.append(f"\n  -> Tool: {tool_call.tool} {status}")

                if tool_call.args:
                    args_str = json.dumps(tool_call.args)
                    if len(args_str) > 200:
                        args_str = f"{args_str[:200]}..."
                    lines.append(f"    Args: {args_str}")

                if tool_call.result:
                    result_str = str(tool_call.result)
                    if len(result_str) > 300:
                        result_str = f"{result_str[:300]}..."
                    lines.append(f"    Result: {result_str}")

            lines.append("")
            lines.append("---")
            lines.append("")

        transcript = "\n".join(lines)
        if len(transcript) <= max_chars:
            return transcript

        keep_each = (max_chars - 100) // 2
        return (
            transcript[:keep_each]
            + "\n\n[... middle of session truncated for length ...]\n\n"
            + transcript[-keep_each:]
        )

    def _parse_llm_response(self, response: str, session: RawSession) -> list[LogEntry]:
        cleaned = self._sanitize_llm_response(response)
        if cleaned in {"[]", "", "NONE", "None", "null"}:
            return []

        parsed = self._parse_json_array(cleaned)
        if not isinstance(parsed, list):
            return []

        entries: list[LogEntry] = []
        for learning in parsed:
            if not isinstance(learning, dict):
                continue
            entry = self._build_entry(learning, session)
            if entry:
                entries.append(entry)

        return entries

    def _parse_json_array(self, text: str) -> list[Any] | None:
        try:
            parsed = json.loads(text)
            return self._unwrap_learnings_payload(parsed)
        except json.JSONDecodeError:
            pass

        for candidate in self._extract_json_candidates(text):
            try:
                parsed = json.loads(candidate)
                learnings = self._unwrap_learnings_payload(parsed)
                if isinstance(learnings, list):
                    return learnings
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _unwrap_learnings_payload(payload: Any) -> list[Any] | None:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("learnings", "entries", "items", "results", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return None

    @staticmethod
    def _sanitize_llm_response(response: str) -> str:
        cleaned = response.strip()
        if not cleaned:
            return ""

        # Remove model reasoning tags commonly emitted by local reasoning models.
        cleaned = re.sub(
            r"<\s*(think|analysis|reasoning)[^>]*>[\s\S]*?<\s*/\s*(think|analysis|reasoning)\s*>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Unwrap fenced blocks while keeping their content.
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        return cleaned.strip()

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        candidates: list[str] = []
        for opening, closing in (("[", "]"), ("{", "}")):
            candidates.extend(
                TranscriptExtractor._extract_balanced_segments(text, opening, closing)
            )
        return candidates

    @staticmethod
    def _extract_balanced_segments(text: str, opening: str, closing: str) -> list[str]:
        segments: list[str] = []
        for start in range(len(text)):
            if text[start] != opening:
                continue

            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                    continue
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        segments.append(text[start : idx + 1])
                        break
        return segments

    def _build_entry(self, learning: dict[str, Any], session: RawSession) -> LogEntry | None:
        content = str(learning.get("content", "")).strip()
        label_str = str(learning.get("label", "")).strip().lower()

        if not content or not label_str:
            return None
        if not self._is_functional_learning(content):
            return None

        label = self._resolve_label(label_str)

        tags_raw = learning.get("tags", [])
        if isinstance(tags_raw, list):
            tags = [str(item).strip().lower() for item in tags_raw if str(item).strip()]
        elif tags_raw:
            tags = [str(tags_raw).strip().lower()]
        else:
            tags = []

        try:
            confidence = float(learning.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        evidence = str(learning.get("evidence", ""))

        return LogEntry(
            session_id=None,
            source=LogSource.EXTRACTED,
            source_session_id=session.session_id,
            content=content,
            label=label,
            tags=tags,
            confidence=confidence,
            curation_status=CurationStatus.PENDING,
            metadata={
                "evidence": evidence,
                "source_tool": session.source,
                "extracted_at": datetime.now(UTC).isoformat(),
            },
        )

    @staticmethod
    def _is_functional_learning(content: str) -> bool:
        lowered = content.lower()
        blocked_phrases = [
            "do not modify plan",
            "don't modify plan",
            "do not edit plan",
            "in_progress",
            "todo",
            "to-do",
            "ticket",
            "jira",
            "workflow",
            "process step",
            "project management",
        ]
        if any(phrase in lowered for phrase in blocked_phrases):
            return False
        return True

    @staticmethod
    def _resolve_label(label_str: str) -> SemanticLabel:
        try:
            return SemanticLabel(label_str)
        except ValueError:
            fallback = {
                "failure": SemanticLabel.HARD_FAILURE,
                "error": SemanticLabel.HARD_FAILURE,
                "warning": SemanticLabel.GOTCHA,
                "tip": SemanticLabel.PATTERN,
                "style": SemanticLabel.PREFERENCE,
            }
            return fallback.get(label_str, SemanticLabel.PATTERN)

    @staticmethod
    def _chunk_messages(messages: list[RawMessage], chunk_size: int) -> list[list[RawMessage]]:
        return [
            messages[start : start + chunk_size] for start in range(0, len(messages), chunk_size)
        ]

    @staticmethod
    def _build_duration(session: RawSession) -> str:
        duration = "unknown"
        if session.ended_at:
            delta = session.ended_at - session.started_at
            minutes = int(delta.total_seconds() / 60)
            if minutes < 60:
                duration = f"{minutes} minutes"
            else:
                duration = f"{minutes // 60}h {minutes % 60}m"
        return duration

    @staticmethod
    def _emit_progress(
        callback: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _deduplicate_entries(entries: list[LogEntry]) -> list[LogEntry]:
        deduped: list[LogEntry] = []
        seen_keys: set[tuple[str, str]] = set()
        for entry in entries:
            key = (entry.label.value, entry.content.strip().lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(entry)
        return deduped

    async def extract(
        self,
        session: RawSession,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[LogEntry]:
        if len(session.messages) < 2:
            return []

        batches = self._chunk_messages(session.messages, self.messages_per_batch)
        if not batches:
            return []

        total_messages = len(session.messages)
        duration = self._build_duration(session)
        combined_entries: list[LogEntry] = []
        messages_processed = 0

        for batch_index, batch_messages in enumerate(batches, start=1):
            messages_processed += len(batch_messages)
            batch_session = session.model_copy(update={"messages": batch_messages})

            transcript = self._format_transcript(batch_session)
            if len(transcript) < 200:
                continue

            response = await self.llm.generate(
                [
                    Message(role="system", content=EXTRACTION_SYSTEM_PROMPT),
                    Message(
                        role="user",
                        content=EXTRACTION_USER_PROMPT.format(
                            source=session.source,
                            project_path=session.project_path or "unknown",
                            date=session.started_at.strftime("%Y-%m-%d %H:%M"),
                            duration=duration,
                            segment=f"batch {batch_index}/{len(batches)}",
                            transcript=transcript,
                        ),
                    ),
                ],
                temperature=0.1,
                max_tokens=700,
            )

            batch_entries = self._parse_llm_response(response.content, batch_session)
            combined_entries.extend(batch_entries)
            self._emit_progress(
                progress_callback,
                {
                    "event": "extraction_batch_complete",
                    "source": session.source,
                    "session_id": session.session_id,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "batch_messages": len(batch_messages),
                    "messages_processed": messages_processed,
                    "messages_total": total_messages,
                    "batch_learnings": len(batch_entries),
                },
            )

        return self._deduplicate_entries(combined_entries)
