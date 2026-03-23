from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_recall.ingest.base import RawMessage, RawSession
from agent_recall.llm.base import LLMProvider, Message
from agent_recall.storage.metadata import AttributionMetadata, build_entry_metadata
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

STRICT FORMAT CONTRACT:
- Return exactly one JSON object and nothing else
- The top-level object must contain exactly one property: "learnings"
- "learnings" must be a JSON array
- Do not include markdown fences, commentary, or thinking tags
- The response must match this JSON Schema:
{schema}

If there are no meaningful learnings, output exactly: {{"learnings": []}}.

JSON object:"""

EXTRACTION_RECOVERY_USER_PROMPT = """The first extraction pass returned no learnings.
These snippets were filtered to keep only high-signal technical content.

Ignore boilerplate, repo instructions, environment setup, and routine status updates.
Extract only concrete technical learnings that would help a future agent avoid mistakes
or work more effectively in this codebase.

Session source: {source}
Project: {project_path}
Date: {date}
Duration: {duration}
Segment: {segment}

=== HIGH-SIGNAL SNIPPETS START ===
{transcript}
=== HIGH-SIGNAL SNIPPETS END ===

STRICT FORMAT CONTRACT:
- Return exactly one JSON object and nothing else
- The top-level object must contain exactly one property: "learnings"
- "learnings" must be a JSON array
- Do not include markdown fences, commentary, or thinking tags
- The response must match this JSON Schema:
{schema}

If there are still no meaningful learnings, output exactly: {{"learnings": []}}.

JSON object:"""

EXTRACTION_REPAIR_USER_PROMPT = """Reformat the previous model output into the exact
extraction format.

Return exactly one JSON object and nothing else.
The top-level object must contain exactly one property: "learnings".
The "learnings" value must be a JSON array.
Do not include markdown fences, commentary, or thinking tags.
If the previous output contains no meaningful learnings, return {{"learnings": []}}.

The response must match this JSON Schema exactly:
{schema}

Previous output:
=== RAW MODEL OUTPUT START ===
{response}
=== RAW MODEL OUTPUT END ===

JSON object:"""


class ExtractionLearningPayload(BaseModel):
    """Strict response contract for one extracted learning."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["hard_failure", "gotcha", "correction", "preference", "pattern", "decision"]
    content: str = Field(..., min_length=1, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: str = Field(default="", max_length=400)


class TranscriptExtractor:
    """Extract semantic learnings from normalized session transcripts."""

    _HIGH_SIGNAL_KEYWORDS = (
        "decision",
        "default",
        "fallback",
        "retry",
        "timeout",
        "warning",
        "error",
        "failed",
        "failure",
        "fix",
        "fixed",
        "bug",
        "issue",
        "gotcha",
        "pattern",
        "preference",
        "correction",
        "should",
        "must",
        "keep",
        "use ",
        "prefer",
        "implemented",
        "added",
        "changed",
        "migrat",
        "refactor",
        "playwright",
        "agent-browser",
        "sqlite",
        "vector",
        "embedding",
        "local",
        "remote",
    )

    def __init__(
        self,
        llm: LLMProvider,
        messages_per_batch: int = 50,
        extracted_entry_curation_status: CurationStatus = CurationStatus.APPROVED,
    ):
        self.llm = llm
        self.messages_per_batch = max(1, int(messages_per_batch))
        self.extracted_entry_curation_status = extracted_entry_curation_status

    def _format_transcript(self, session: RawSession, max_chars: int = 8_000) -> str:
        segments: list[str] = []

        for message in session.messages:
            formatted = self._format_message(message)
            if formatted:
                segments.append(formatted)

        return self._fit_segments_to_budget(segments, max_chars=max_chars)

    @staticmethod
    def _extraction_schema() -> str:
        return json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["learnings"],
                "properties": {
                    "learnings": {
                        "type": "array",
                        "items": ExtractionLearningPayload.model_json_schema(),
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )

    def _render_prompt(
        self,
        template: str,
        *,
        session: RawSession,
        duration: str,
        segment: str,
        transcript: str,
        response: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "source": session.source,
            "project_path": session.project_path or "unknown",
            "date": session.started_at.strftime("%Y-%m-%d %H:%M"),
            "duration": duration,
            "segment": segment,
            "transcript": transcript,
            "schema": self._extraction_schema(),
        }
        if response is not None:
            payload["response"] = self._clip_text(response.strip(), max_chars=4_000)
        return template.format(**payload)

    def _validate_learning_payload(
        self,
        learning: Any,
    ) -> ExtractionLearningPayload | None:
        if not isinstance(learning, dict):
            return None
        try:
            return ExtractionLearningPayload.model_validate(learning)
        except ValidationError:
            return None

    def _parse_llm_response(
        self,
        response: str,
        session: RawSession,
    ) -> tuple[list[LogEntry], Literal["ok", "empty", "invalid"]]:
        cleaned = self._sanitize_llm_response(response)
        if cleaned in {
            "[]",
            "",
            "NONE",
            "None",
            "null",
            '{"learnings":[]}',
            '{"learnings": []}',
        }:
            return [], "empty"

        parsed = self._parse_json_array(cleaned)
        if not isinstance(parsed, list):
            return [], "invalid"

        entries: list[LogEntry] = []
        valid_items = 0
        for learning in parsed:
            payload = self._validate_learning_payload(learning)
            if payload is None:
                continue
            valid_items += 1
            entry = self._build_entry(payload.model_dump(), session)
            if entry:
                entries.append(entry)

        if parsed and valid_items == 0:
            return [], "invalid"
        return entries, "ok"

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
        attribution = AttributionMetadata(
            agent_source=session.source,
            provider=self.llm.provider_name,
            model=self.llm.model_name,
        )

        return LogEntry(
            session_id=None,
            source=LogSource.EXTRACTED,
            source_session_id=session.session_id,
            content=content,
            label=label,
            tags=tags,
            confidence=confidence,
            curation_status=self.extracted_entry_curation_status,
            metadata=build_entry_metadata(
                attribution=attribution,
                evidence=evidence,
                source_tool=session.source,
                extracted_at=datetime.now(UTC).isoformat(),
            ),
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

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"[ \t]+", " ", text).strip()

    def _normalize_message_content(self, content: str, *, max_chars: int) -> str:
        cleaned = content.strip()
        if not cleaned:
            return ""

        cleaned = re.sub(
            r"<environment_context>[\s\S]*?</environment_context>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned or self._looks_like_instruction_boilerplate(cleaned):
            return ""

        normalized_lines = [self._normalize_whitespace(line) for line in cleaned.splitlines()]
        normalized_lines = [line for line in normalized_lines if line]
        if not normalized_lines:
            return ""

        normalized = "\n".join(normalized_lines)
        return self._clip_text(normalized, max_chars=max_chars)

    @staticmethod
    def _looks_like_instruction_boilerplate(content: str) -> bool:
        lowered = content.lower()
        if lowered.startswith("# agents.md instructions"):
            return True

        markers = (
            "### available skills",
            "trigger rules",
            "how to use a skill",
            "intermediary updates",
            "working with the user",
            "final answer instructions",
        )
        matches = sum(1 for marker in markers if marker in lowered)
        return matches >= 3 and len(lowered) > 600

    @staticmethod
    def _clip_text(text: str, *, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 120:
            return f"{text[: max_chars - 3].rstrip()}..."
        head = max_chars // 2 - 10
        tail = max_chars - head - 25
        return f"{text[:head].rstrip()} ... {text[-tail:].lstrip()}"

    @staticmethod
    def _preview_value(value: Any, *, max_chars: int) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value)
            except TypeError:
                text = str(value)
        return TranscriptExtractor._clip_text(
            TranscriptExtractor._normalize_whitespace(text),
            max_chars=max_chars,
        )

    def _format_tool_call_lines(self, message: RawMessage) -> list[str]:
        lines: list[str] = []
        visible_calls = message.tool_calls[:2]
        for tool_call in visible_calls:
            status = "OK" if tool_call.success else "ERR"
            lines.append(f"  -> Tool: {tool_call.tool} {status}")

            cmd = tool_call.args.get("cmd") if isinstance(tool_call.args, dict) else None
            if isinstance(cmd, str) and cmd.strip():
                lines.append(f"    Command: {self._preview_value(cmd, max_chars=140)}")
            elif tool_call.args:
                lines.append(f"    Args: {self._preview_value(tool_call.args, max_chars=140)}")

            if not tool_call.success and tool_call.result:
                lines.append(f"    Error: {self._preview_value(tool_call.result, max_chars=180)}")

        omitted = len(message.tool_calls) - len(visible_calls)
        if omitted > 0:
            suffix = "call" if omitted == 1 else "calls"
            lines.append(f"    ... {omitted} additional tool {suffix} omitted")
        return lines

    def _format_message(self, message: RawMessage) -> str:
        content_budget = 420 if message.role == "user" else 360
        content = self._normalize_message_content(message.content, max_chars=content_budget)
        tool_lines = self._format_tool_call_lines(message)
        if not content and not tool_lines:
            return ""

        ts = f" [{message.timestamp.strftime('%H:%M')}]" if message.timestamp else ""
        role_display = "USER" if message.role == "user" else "ASSISTANT"
        lines = [f"### {role_display}{ts}"]
        if content:
            lines.extend(["", content])
        if tool_lines:
            lines.extend(["", *tool_lines])
        return "\n".join(lines)

    def _fit_segments_to_budget(self, segments: list[str], *, max_chars: int) -> str:
        if not segments:
            return ""

        separator = "\n\n---\n\n"
        transcript = separator.join(segments)
        if len(transcript) <= max_chars:
            return transcript

        available = max_chars - len(separator) * max(0, len(segments) - 1)
        if available <= 0:
            return self._clip_text(transcript, max_chars=max_chars)

        per_segment_budget = max(90, available // len(segments))
        compacted = [self._clip_text(segment, max_chars=per_segment_budget) for segment in segments]
        transcript = separator.join(compacted)
        if len(transcript) <= max_chars:
            return transcript

        return self._clip_text(transcript, max_chars=max_chars)

    def _message_signal_score(self, message: RawMessage) -> int:
        content = self._normalize_message_content(message.content, max_chars=2_000)
        if not content:
            return 0

        lowered = content.lower()
        score = 1 if message.role == "assistant" else 0
        score += sum(2 for keyword in self._HIGH_SIGNAL_KEYWORDS if keyword in lowered)
        if len(content) >= 140:
            score += 1
        if any(not tool_call.success for tool_call in message.tool_calls):
            score += 3
        elif message.tool_calls:
            score += 1
        return score

    def _select_recovery_messages(
        self,
        session: RawSession,
        *,
        max_messages: int = 12,
    ) -> list[RawMessage]:
        scored: list[tuple[int, int]] = []
        for index, message in enumerate(session.messages):
            score = self._message_signal_score(message)
            if score > 0:
                scored.append((score, index))

        if not scored:
            return []

        selected_indices = {
            index
            for _, index in sorted(
                scored,
                key=lambda item: (-item[0], item[1]),
            )[:max_messages]
        }

        for index in range(len(session.messages) - 1, -1, -1):
            if self._message_signal_score(session.messages[index]) > 0:
                selected_indices.add(index)
                break

        return [session.messages[index] for index in sorted(selected_indices)]

    async def _repair_entries(
        self,
        *,
        session: RawSession,
        duration: str,
        segment: str,
        raw_response: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> list[LogEntry]:
        repair_prompt = self._render_prompt(
            EXTRACTION_REPAIR_USER_PROMPT,
            session=session,
            duration=duration,
            segment=segment,
            transcript="",
            response=raw_response,
        )
        repaired = await self.llm.generate(
            [
                Message(role="system", content=EXTRACTION_SYSTEM_PROMPT),
                Message(role="user", content=repair_prompt),
            ],
            temperature=0.0,
            max_tokens=900,
        )
        entries, status = self._parse_llm_response(repaired.content, session)
        normalized_entries = self._deduplicate_entries(entries)
        payload = {
            "event": "extraction_repair_complete",
            "source": session.source,
            "session_id": session.session_id,
            "segment": segment,
            "repair_success": status != "invalid",
            "batch_learnings": len(normalized_entries),
        }
        if event_payload:
            payload.update(event_payload)
        self._emit_progress(progress_callback, payload)
        return normalized_entries if status != "invalid" else []

    async def _generate_entries(
        self,
        *,
        session: RawSession,
        duration: str,
        segment: str,
        transcript: str,
        max_tokens: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        repair_event_payload: dict[str, Any] | None = None,
    ) -> list[LogEntry]:
        prompt = self._render_prompt(
            EXTRACTION_USER_PROMPT
            if "recovery" not in segment
            else EXTRACTION_RECOVERY_USER_PROMPT,
            session=session,
            duration=duration,
            segment=segment,
            transcript=transcript,
        )
        response = await self.llm.generate(
            [
                Message(role="system", content=EXTRACTION_SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        entries, status = self._parse_llm_response(response.content, session)
        normalized_entries = self._deduplicate_entries(entries)
        if status != "invalid":
            return normalized_entries
        return await self._repair_entries(
            session=session,
            duration=duration,
            segment=segment,
            raw_response=response.content,
            progress_callback=progress_callback,
            event_payload=repair_event_payload,
        )

    async def _run_recovery_pass(
        self,
        session: RawSession,
        *,
        duration: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[LogEntry], bool]:
        recovery_messages = self._select_recovery_messages(session)
        if not recovery_messages:
            return [], False

        recovery_session = session.model_copy(update={"messages": recovery_messages})
        transcript = self._format_transcript(recovery_session, max_chars=5_500)
        if len(transcript) < 200:
            return [], False

        entries = await self._generate_entries(
            session=session,
            duration=duration,
            segment="recovery pass",
            transcript=transcript,
            max_tokens=900,
            progress_callback=progress_callback,
            repair_event_payload={
                "messages_total": len(session.messages),
                "messages_considered": len(recovery_messages),
            },
        )
        self._emit_progress(
            progress_callback,
            {
                "event": "extraction_recovery_complete",
                "source": session.source,
                "session_id": session.session_id,
                "messages_total": len(session.messages),
                "messages_considered": len(recovery_messages),
                "batch_learnings": len(entries),
            },
        )
        return entries, True

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

            batch_entries = await self._generate_entries(
                session=batch_session,
                duration=duration,
                segment=f"batch {batch_index}/{len(batches)}",
                transcript=transcript,
                max_tokens=700,
                progress_callback=progress_callback,
                repair_event_payload={
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "messages_processed": messages_processed,
                    "messages_total": total_messages,
                },
            )
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

        deduplicated = self._deduplicate_entries(combined_entries)
        if deduplicated or total_messages < max(self.messages_per_batch, 40):
            return deduplicated

        recovery_entries, attempted = await self._run_recovery_pass(
            session,
            duration=duration,
            progress_callback=progress_callback,
        )
        if attempted and recovery_entries:
            return recovery_entries
        return deduplicated
