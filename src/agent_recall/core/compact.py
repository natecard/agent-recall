from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from agent_recall.core.embeddings import generate_embedding
from agent_recall.core.tier_format import is_ralph_entry_start
from agent_recall.llm.base import LLMProvider, Message
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import (
    Chunk,
    ChunkSource,
    CurationStatus,
    LogEntry,
    SemanticLabel,
    SessionStatus,
)

GUARDRAILS_PROMPT = """You are synthesizing guardrails from development learnings.

Current GUARDRAILS.md:
{current_guardrails}

Candidate learnings:
{entries}

Return ONLY JSON in this shape:
{{
  "items": [
    {{
      "type": "FAILURE|GOTCHA|CORRECTION",
      "rule": "<concise actionable rule>",
      "why": "<brief reason>"
    }}
  ]
}}

Rules:
- Include only NEW, durable rules not already in the current file.
- If no updates are needed, return: {{"items":[]}}
- No markdown fences and no prose.
"""

STYLE_PROMPT = """You are synthesizing coding style from development learnings.

Current STYLE.md:
{current_style}

Candidate learnings:
{entries}

Return ONLY JSON in this shape:
{{
  "items": [
    {{
      "type": "PREFERENCE|PATTERN",
      "guideline": "<concise guideline>"
    }}
  ]
}}

Rules:
- Include only NEW guidance not already in the current file.
- If no updates are needed, return: {{"items":[]}}
- No markdown fences and no prose.
"""

RECENT_PROMPT = """Summarize recent development activity for RECENT.md.

Current RECENT.md:
{current_recent}

Session evidence:
{sessions}

Return ONLY JSON in this shape:
{{
  "items": [
    {{
      "date": "YYYY-MM-DD",
      "summary": "1-2 sentence summary"
    }}
  ]
}}

Rules:
- Keep summaries concrete and outcome-focused.
- Include up to 12 items, newest first.
- If no update is needed, return: {{"items":[]}}
- No markdown fences and no prose.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?|```", flags=re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*-\s*\[(?P<kind>[A-Z_]+)\]\s*(?P<text>.+?)\s*$")
_RECENT_RE = re.compile(r"^\s*\*\*(?P<date>\d{4}-\d{2}-\d{2})\*\*:\s*(?P<summary>.+)\s*$")


class CompactionEngine:
    def __init__(self, storage: Storage, files: FileStorage, llm: LLMProvider):
        self.storage = storage
        self.files = files
        self.llm = llm

    async def compact(self, force: bool = False) -> dict[str, bool | int]:
        """Run compaction and return summary details."""
        results: dict[str, bool | int] = {
            "guardrails_updated": False,
            "style_updated": False,
            "recent_updated": False,
            "chunks_indexed": 0,
            "llm_requests": 0,
            "llm_responses": 0,
        }

        config = self.files.read_config()
        compaction_cfg = config.get("compaction") if isinstance(config, dict) else {}
        if not isinstance(compaction_cfg, dict):
            compaction_cfg = {}
        retrieval_cfg = config.get("retrieval") if isinstance(config, dict) else {}
        if not isinstance(retrieval_cfg, dict):
            retrieval_cfg = {}

        pattern_threshold = int(compaction_cfg.get("promote_pattern_after_occurrences", 3))
        effective_pattern_threshold = 1 if force else max(1, pattern_threshold)
        recent_token_budget = int(compaction_cfg.get("max_recent_tokens", 1500))
        embedding_enabled = bool(retrieval_cfg.get("embedding_enabled", False))
        embedding_dimensions = int(retrieval_cfg.get("embedding_dimensions", 64))
        if embedding_dimensions < 8:
            embedding_dimensions = 8

        guardrail_labels = [
            SemanticLabel.HARD_FAILURE,
            SemanticLabel.GOTCHA,
            SemanticLabel.CORRECTION,
        ]
        style_labels = [SemanticLabel.PREFERENCE, SemanticLabel.PATTERN]
        non_style_index_labels = self._resolve_non_style_index_labels(compaction_cfg)
        non_style_index_thresholds = self._resolve_non_style_index_thresholds(compaction_cfg)

        curation_status = self._resolve_curation_status(compaction_cfg)
        guardrail_entries = self.storage.get_entries_by_label(
            guardrail_labels,
            curation_status=curation_status,
        )
        style_entries = self.storage.get_entries_by_label(
            style_labels,
            curation_status=curation_status,
        )
        non_style_index_entries = self._filter_non_style_index_entries(
            self.storage.get_entries_by_label(
                non_style_index_labels,
                curation_status=curation_status,
            ),
            non_style_index_thresholds,
        )
        promoted_style_entries = self._promoted_style_entries(
            style_entries,
            effective_pattern_threshold,
        )

        if guardrail_entries:
            current = self.files.read_tier(KnowledgeTier.GUARDRAILS)
            entries_text = self._format_entries_for_prompt(guardrail_entries)
            results["llm_requests"] = int(results["llm_requests"]) + 1
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
            results["llm_responses"] = int(results["llm_responses"]) + 1
            synthesized = self._extract_guardrail_lines(response.content)
            if synthesized:
                changed = self._merge_and_write_tier(
                    tier=KnowledgeTier.GUARDRAILS,
                    current=current,
                    new_lines=synthesized,
                    matcher=_BULLET_RE,
                )
                if changed:
                    results["guardrails_updated"] = True

        if promoted_style_entries:
            current = self.files.read_tier(KnowledgeTier.STYLE)
            entries_text = self._format_entries_for_prompt(promoted_style_entries)
            results["llm_requests"] = int(results["llm_requests"]) + 1
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
            results["llm_responses"] = int(results["llm_responses"]) + 1
            synthesized = self._extract_style_lines(response.content)
            if synthesized:
                changed = self._merge_and_write_tier(
                    tier=KnowledgeTier.STYLE,
                    current=current,
                    new_lines=synthesized,
                    matcher=_BULLET_RE,
                )
                if changed:
                    results["style_updated"] = True

        recent_evidence = self._recent_evidence_lines()
        if recent_evidence:
            current_recent = self.files.read_tier(KnowledgeTier.RECENT)
            results["llm_requests"] = int(results["llm_requests"]) + 1
            response = await self.llm.generate(
                [
                    Message(
                        role="user",
                        content=RECENT_PROMPT.format(
                            current_recent=current_recent or "(empty)",
                            sessions="\n".join(recent_evidence),
                        ),
                    )
                ]
            )
            results["llm_responses"] = int(results["llm_responses"]) + 1
            recent_lines = self._extract_recent_lines(response.content)
            if recent_lines:
                recent_lines = self._trim_recent_lines(recent_lines, recent_token_budget)
                changed = self._write_recent_lines(current_recent, recent_lines)
                if changed:
                    results["recent_updated"] = True

        indexed_entry_ids: set[str] = set()
        for entry in [*guardrail_entries, *promoted_style_entries, *non_style_index_entries]:
            entry_id = str(entry.id)
            if entry_id in indexed_entry_ids:
                continue
            indexed_entry_ids.add(entry_id)
            if self.storage.has_chunk(entry.content, entry.label):
                continue
            embedding = (
                generate_embedding(entry.content, dimensions=embedding_dimensions)
                if embedding_enabled
                else None
            )
            self.storage.store_chunk(
                Chunk(
                    source=ChunkSource.LOG_ENTRY,
                    source_ids=[entry.id],
                    content=entry.content,
                    label=entry.label,
                    tags=entry.tags,
                    embedding=embedding,
                )
            )
            results["chunks_indexed"] = int(results["chunks_indexed"]) + 1

        return results

    @staticmethod
    def _format_entries_for_prompt(entries: list[LogEntry]) -> str:
        lines: list[str] = []
        for entry in entries:
            lines.append(f"- id={entry.id} [{entry.label.value}] {entry.content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_line(line: str) -> str:
        return " ".join(line.strip().lower().split())

    @staticmethod
    def _normalize_content(content: str) -> str:
        return " ".join(content.strip().lower().split())

    def _promoted_style_entries(
        self,
        style_entries: list[LogEntry],
        pattern_threshold: int,
    ) -> list[LogEntry]:
        pattern_counts = Counter(
            self._normalize_content(entry.content)
            for entry in style_entries
            if entry.label == SemanticLabel.PATTERN
        )
        promoted: list[LogEntry] = []
        for entry in style_entries:
            if entry.label == SemanticLabel.PREFERENCE:
                promoted.append(entry)
                continue
            if entry.label == SemanticLabel.PATTERN:
                normalized = self._normalize_content(entry.content)
                if pattern_counts[normalized] >= max(1, pattern_threshold):
                    promoted.append(entry)
        return promoted

    @staticmethod
    def _resolve_non_style_index_labels(compaction_cfg: dict[str, Any]) -> list[SemanticLabel]:
        policies = [
            (SemanticLabel.DECISION_RATIONALE, compaction_cfg.get("index_decision_entries", True)),
            (SemanticLabel.EXPLORATION, compaction_cfg.get("index_exploration_entries", True)),
            (SemanticLabel.NARRATIVE, compaction_cfg.get("index_narrative_entries", False)),
        ]
        return [label for label, enabled in policies if bool(enabled)]

    @staticmethod
    def _resolve_non_style_index_thresholds(
        compaction_cfg: dict[str, Any],
    ) -> dict[SemanticLabel, float]:
        defaults = {
            SemanticLabel.DECISION_RATIONALE: 0.7,
            SemanticLabel.EXPLORATION: 0.7,
            SemanticLabel.NARRATIVE: 0.8,
        }
        thresholds: dict[SemanticLabel, float] = {}
        raw_values = [
            (
                SemanticLabel.DECISION_RATIONALE,
                compaction_cfg.get(
                    "index_decision_min_confidence",
                    defaults[SemanticLabel.DECISION_RATIONALE],
                ),
            ),
            (
                SemanticLabel.EXPLORATION,
                compaction_cfg.get(
                    "index_exploration_min_confidence",
                    defaults[SemanticLabel.EXPLORATION],
                ),
            ),
            (
                SemanticLabel.NARRATIVE,
                compaction_cfg.get(
                    "index_narrative_min_confidence",
                    defaults[SemanticLabel.NARRATIVE],
                ),
            ),
        ]

        for label, raw_value in raw_values:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = defaults[label]
            thresholds[label] = max(0.0, min(1.0, value))

        return thresholds

    @staticmethod
    def _resolve_curation_status(compaction_cfg: dict[str, Any]) -> CurationStatus:
        raw_value = compaction_cfg.get("curation_status", CurationStatus.APPROVED.value)
        try:
            return CurationStatus(str(raw_value).strip().lower())
        except ValueError:
            return CurationStatus.APPROVED

    @staticmethod
    def _filter_non_style_index_entries(
        entries: list[LogEntry],
        confidence_thresholds: dict[SemanticLabel, float],
    ) -> list[LogEntry]:
        return [
            entry
            for entry in entries
            if entry.confidence >= confidence_thresholds.get(entry.label, 0.0)
        ]

    def _recent_evidence_lines(self) -> list[str]:
        completed_sessions = self.storage.list_sessions(limit=20, status=SessionStatus.COMPLETED)
        if completed_sessions:
            lines: list[str] = []
            for session in completed_sessions:
                date = session.ended_at.date().isoformat() if session.ended_at else "unknown-date"
                summary = session.summary or "No summary provided"
                lines.append(f"- {date}: task={session.task}; summary={summary}")
            return lines

        inferred = self.storage.list_recent_source_sessions(limit=20)
        lines = []
        for item in inferred:
            last = item.get("last_timestamp")
            if isinstance(last, datetime):
                date = (
                    last.replace(tzinfo=UTC).date().isoformat()
                    if last.tzinfo is None
                    else last.astimezone(UTC).date().isoformat()
                )
            else:
                date = "unknown-date"
            session_id = str(item.get("source_session_id", "unknown"))
            entry_count = int(item.get("entry_count", 0))
            highlights = item.get("highlights")
            snippets: list[str] = []
            if isinstance(highlights, list):
                for highlight in highlights[:3]:
                    if not isinstance(highlight, dict):
                        continue
                    label = str(highlight.get("label", "?"))
                    content = str(highlight.get("content", "")).strip()
                    if not content:
                        continue
                    snippets.append(f"[{label}] {content[:120]}")
            highlight_text = "; ".join(snippets) if snippets else "No highlights"
            lines.append(
                f"- {date}: source_session_id={session_id}; entries={entry_count}; "
                f"highlights={highlight_text}"
            )
        return lines

    @staticmethod
    def _clean_llm_response(content: str) -> str:
        cleaned = content.strip()
        cleaned = _JSON_FENCE_RE.sub("", cleaned).strip()
        return cleaned

    def _parse_json_payload(self, content: str) -> Any:
        cleaned = self._clean_llm_response(content)
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start == -1 or end == -1 or end <= start:
                continue
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _extract_guardrail_lines(self, content: str) -> list[str]:
        return self._extract_typed_lines(
            content=content,
            allowed_types={"FAILURE", "GOTCHA", "CORRECTION"},
            text_keys=("rule", "content", "guideline"),
            why_keys=("why", "reason"),
        )

    def _extract_style_lines(self, content: str) -> list[str]:
        return self._extract_typed_lines(
            content=content,
            allowed_types={"PREFERENCE", "PATTERN"},
            text_keys=("guideline", "content", "rule"),
            why_keys=(),
        )

    def _extract_typed_lines(
        self,
        *,
        content: str,
        allowed_types: set[str],
        text_keys: tuple[str, ...],
        why_keys: tuple[str, ...],
    ) -> list[str]:
        payload = self._parse_json_payload(content)
        typed_lines: list[str] = []
        if isinstance(payload, dict):
            raw_items = payload.get("items")
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = None

        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("type", "")).strip().upper()
                if kind not in allowed_types:
                    continue
                text = ""
                for key in text_keys:
                    candidate = str(item.get(key, "")).strip()
                    if candidate:
                        text = candidate
                        break
                if not text:
                    continue

                suffix = ""
                for key in why_keys:
                    reason = str(item.get(key, "")).strip()
                    if reason and reason.lower() not in text.lower():
                        suffix = f" ({reason})"
                        break
                typed_lines.append(f"- [{kind}] {text}{suffix}")

        if typed_lines:
            return typed_lines

        fallback: list[str] = []
        for line in self._clean_llm_response(content).splitlines():
            match = _BULLET_RE.match(line)
            if not match:
                continue
            kind = str(match.group("kind")).strip().upper()
            text = str(match.group("text")).strip()
            if kind in allowed_types and text:
                fallback.append(f"- [{kind}] {text}")
        return fallback

    def _extract_recent_lines(self, content: str) -> list[str]:
        payload = self._parse_json_payload(content)
        recent_lines: list[str] = []
        raw_items: Any = None
        if isinstance(payload, dict):
            raw_items = payload.get("items")
        elif isinstance(payload, list):
            raw_items = payload

        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                date = str(item.get("date", "")).strip()
                summary = str(item.get("summary", "")).strip()
                if date and summary:
                    recent_lines.append(f"**{date}**: {summary}")
        if recent_lines:
            return recent_lines

        fallback: list[str] = []
        for line in self._clean_llm_response(content).splitlines():
            if _RECENT_RE.match(line):
                fallback.append(line.strip())
        return fallback

    @staticmethod
    def _trim_recent_lines(lines: list[str], token_budget: int) -> list[str]:
        if token_budget <= 0:
            return lines
        # Approximate tokens ~= chars/4 to keep budget deterministic and cheap.
        budget_chars = token_budget * 4
        kept: list[str] = []
        total = 0
        for line in lines:
            next_total = total + len(line) + 1
            if kept and next_total > budget_chars:
                break
            kept.append(line)
            total = next_total
        return kept

    def _merge_and_write_tier(
        self,
        *,
        tier: KnowledgeTier,
        current: str,
        new_lines: list[str],
        matcher: re.Pattern[str],
    ) -> bool:
        preamble, existing_lines = self._split_preamble_and_lines(current, matcher)
        seen = {self._normalize_line(line) for line in existing_lines}
        additions: list[str] = []
        for line in new_lines:
            normalized = self._normalize_line(line)
            if normalized and normalized not in seen:
                additions.append(line.strip())
                seen.add(normalized)
        if not additions:
            return False

        updated_lines = [*existing_lines, *additions]
        updated = self._compose_tier_text(preamble, updated_lines)
        if updated.strip() == current.strip():
            return False
        self._archive_tier_snapshot(tier, current)
        self.files.write_tier(tier, updated)
        return True

    def _write_recent_lines(self, current: str, lines: list[str]) -> bool:
        preamble, _existing_recent = self._split_preamble_and_lines(current, _RECENT_RE)
        updated = self._compose_tier_text(preamble, lines)
        if updated.strip() == current.strip():
            return False
        self._archive_tier_snapshot(KnowledgeTier.RECENT, current)
        self.files.write_tier(KnowledgeTier.RECENT, updated)
        return True

    @staticmethod
    def _split_preamble_and_lines(
        content: str,
        matcher: re.Pattern[str],
    ) -> tuple[list[str], list[str]]:
        lines = content.splitlines()
        preamble: list[str] = []
        extracted: list[str] = []
        in_ralph_block = False
        current_ralph_empty_count = 0
        found_entry = False

        for line in lines:
            if is_ralph_entry_start(line):
                in_ralph_block = True
                current_ralph_empty_count = 0
                found_entry = True
                continue

            if in_ralph_block:
                if line.startswith("## ") and not is_ralph_entry_start(line):
                    in_ralph_block = False
                elif line.strip() == "":
                    current_ralph_empty_count += 1
                    if current_ralph_empty_count >= 2:
                        in_ralph_block = False
                    continue
                else:
                    current_ralph_empty_count = 0
                    continue

            if matcher.match(line):
                extracted.append(line.strip())
                found_entry = True
            elif not found_entry:
                preamble.append(line)

        return preamble, extracted

    @staticmethod
    def _compose_tier_text(preamble_lines: list[str], body_lines: list[str]) -> str:
        preamble = "\n".join(preamble_lines).rstrip()
        body = "\n".join(body_lines).strip()
        if preamble and body:
            return f"{preamble}\n\n{body}\n"
        if preamble:
            return f"{preamble}\n"
        if body:
            return f"{body}\n"
        return ""

    def _archive_tier_snapshot(self, tier: KnowledgeTier, previous_content: str) -> None:
        if not previous_content.strip():
            return
        archive_dir = self.files.agent_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        archive_path = archive_dir / f"{tier.value.lower()}-{timestamp}.md"
        archive_path.write_text(previous_content)
