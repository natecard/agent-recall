from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.core.guardrail_enforcement import (
    GuardrailSuppressionStore,
    evaluate_guardrail_text,
    is_guardrail_enforcement_enabled,
    parse_guardrail_rules,
)
from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.core.tier_writer import TIER_HEADERS
from agent_recall.external_compaction.models import (
    ExternalCompactionExportPayload,
    external_notes_json_schema,
    validate_external_notes_payload,
)
from agent_recall.external_compaction.write_guard import ExternalWriteScopeGuard, WriteTarget
from agent_recall.storage.base import Storage, UnsupportedStorageCapabilityError
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LogEntry, PipelineEventAction, PipelineStage

_SUPPORTED_TIERS = (
    KnowledgeTier.GUARDRAILS,
    KnowledgeTier.STYLE,
    KnowledgeTier.RECENT,
)
_QUEUE_STATES = {"pending", "approved", "rejected", "applied"}
_CONFLICT_POLICIES = {"prefer_newest", "queue_for_review"}
_NEGATIVE_POLARITY_TOKENS = {
    "avoid",
    "ban",
    "deny",
    "disable",
    "dont",
    "never",
    "no",
    "not",
    "without",
}
_POSITIVE_POLARITY_TOKENS = {
    "always",
    "enable",
    "ensure",
    "must",
    "prefer",
    "require",
    "should",
    "use",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_BULLET_PREFIX_RE = re.compile(r"^\s*-\s*\[[A-Z_]+\]\s*", re.IGNORECASE)
_RECENT_PREFIX_RE = re.compile(r"^\s*\*\*\d{4}-\d{2}-\d{2}\*\*:\s*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _semantic_tokens(text: str) -> list[str]:
    cleaned = _BULLET_PREFIX_RE.sub("", text.strip().lower())
    cleaned = _RECENT_PREFIX_RE.sub("", cleaned)
    return _TOKEN_RE.findall(cleaned)


def _semantic_key(text: str) -> str:
    return " ".join(_semantic_tokens(text))


def _topic_key(text: str) -> str:
    tokens = _semantic_tokens(text)
    reduced = [
        token
        for token in tokens
        if token not in _STOPWORDS
        and token not in _NEGATIVE_POLARITY_TOKENS
        and token not in _POSITIVE_POLARITY_TOKENS
    ]
    if not reduced:
        reduced = [token for token in tokens if token not in _STOPWORDS]
    return " ".join(reduced[:10])


def _polarity(text: str) -> str:
    tokens = set(_semantic_tokens(text))
    has_negative = bool(tokens.intersection(_NEGATIVE_POLARITY_TOKENS))
    has_positive = bool(tokens.intersection(_POSITIVE_POLARITY_TOKENS))
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return "neutral"


def _token_set(text: str) -> set[str]:
    return {token for token in _semantic_tokens(text) if token not in _STOPWORDS}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _is_candidate_line(line: str) -> bool:
    normalized = line.strip()
    return bool(normalized) and (normalized.startswith("- [") or normalized.startswith("**"))


def _to_iso(value: object) -> str | None:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class _CandidateNote:
    line: str
    source_session_ids: tuple[str, ...]


@dataclass
class _MergeOutcome:
    merged: str
    notes_applied: int
    applied_notes: list[_CandidateNote]
    conflicts: list[dict[str, str]]


class ExternalCompactionService:
    """Read imported conversations and apply external notes to tier markdown files."""

    def __init__(
        self,
        storage: Storage,
        files: FileStorage,
        *,
        agent_dir: Path,
        repo_root: Path,
        state_path: Path | None = None,
        write_guard: ExternalWriteScopeGuard | None = None,
    ) -> None:
        self.storage = storage
        self.files = files
        self.agent_dir = agent_dir
        self.repo_root = repo_root.resolve()
        self.state_path = state_path or (agent_dir / "external_compaction_state.json")
        config = self.files.read_config()
        self.write_guard = write_guard or ExternalWriteScopeGuard.from_config(
            repo_root=self.repo_root,
            config=config,
        )
        self.conflict_policy = self._resolve_conflict_policy(config)
        self.guardrail_enforcement_enabled = is_guardrail_enforcement_enabled(config)
        self.guardrail_suppressions = GuardrailSuppressionStore(
            self.agent_dir / "guardrail_suppressions.json"
        )
        self._backfill_state_to_storage()

    def list_imported_conversations(
        self,
        *,
        limit: int = 20,
        pending_only: bool = True,
    ) -> list[dict[str, Any]]:
        rows = self.storage.list_recent_source_sessions(limit=max(1, int(limit)))
        state = self._load_state()
        sessions_state = state.get("sessions", {})
        if not isinstance(sessions_state, dict):
            sessions_state = {}

        results: list[dict[str, Any]] = []
        for row in rows:
            source_session_id = str(row.get("source_session_id", "")).strip()
            if not source_session_id:
                continue

            source_hash = self._hash_source_session(source_session_id)
            prior = sessions_state.get(source_session_id, {})
            prior_hash = ""
            if isinstance(prior, dict):
                prior_hash = str(prior.get("source_hash", "")).strip()

            is_pending = source_hash != prior_hash
            if pending_only and not is_pending:
                continue

            highlights_raw = row.get("highlights")
            highlights: list[dict[str, str]] = []
            if isinstance(highlights_raw, list):
                for item in highlights_raw:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label", "")).strip()
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    highlights.append({"label": label, "content": content})

            results.append(
                {
                    "source_session_id": source_session_id,
                    "last_timestamp": _to_iso(row.get("last_timestamp")),
                    "entry_count": int(row.get("entry_count", 0)),
                    "pending": is_pending,
                    "source_hash": source_hash,
                    "highlights": highlights,
                }
            )

        return results

    def get_conversation(
        self,
        source_session_id: str,
        *,
        limit: int = 300,
    ) -> dict[str, Any]:
        normalized = source_session_id.strip()
        if not normalized:
            raise ValueError("source_session_id is required")

        entries = self.storage.get_entries_by_source_session(normalized, limit=max(1, int(limit)))
        return {
            "source_session_id": normalized,
            "entry_count": len(entries),
            "entries": [self._serialize_entry(entry) for entry in entries],
        }

    def read_tiers(self, *, write_target: WriteTarget = "runtime") -> dict[str, str]:
        target = self._normalize_write_target(write_target)
        return {
            tier.value: self._read_target_tier(tier, write_target=target)
            for tier in _SUPPORTED_TIERS
        }

    def build_payload(
        self,
        *,
        source_session_ids: list[str] | None = None,
        limit: int = 20,
        pending_only: bool = True,
        entry_limit: int = 300,
        write_target: WriteTarget = "runtime",
    ) -> dict[str, Any]:
        target = self._normalize_write_target(write_target)
        selected: list[str]
        if source_session_ids:
            selected = [item.strip() for item in source_session_ids if item and item.strip()]
        else:
            sessions = self.list_imported_conversations(limit=limit, pending_only=pending_only)
            selected = [str(item["source_session_id"]) for item in sessions]

        conversations = [
            self.get_conversation(source_session_id, limit=entry_limit)
            for source_session_id in selected
        ]

        payload = ExternalCompactionExportPayload.model_validate(
            {
                "generated_at": datetime.now(UTC),
                "write_target": target,
                "tiers": self.read_tiers(write_target=target),
                "conversations": conversations,
                "notes_schema": external_notes_json_schema(),
            }
        )
        return payload.model_dump(mode="json")

    def apply_notes_payload(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        write_target: WriteTarget = "runtime",
        dry_run: bool = False,
        mark_processed: bool = True,
    ) -> dict[str, Any]:
        telemetry = PipelineTelemetry.from_config(
            agent_dir=self.agent_dir,
            config=self.files.read_config(),
        )
        run_id = telemetry.create_run_id("external-apply")
        started = time.perf_counter()
        target = self._normalize_write_target(write_target)
        try:
            notes = self._parse_notes(payload)
            guardrail_warnings: list[dict[str, Any]] = []
            guardrail_blocks: list[dict[str, Any]] = []
            if self.guardrail_enforcement_enabled:
                rules = parse_guardrail_rules(self.files.read_tier(KnowledgeTier.GUARDRAILS))
                for note in notes:
                    note_text = str(note.get("line", "")).strip()
                    if not note_text:
                        continue
                    violations = evaluate_guardrail_text(
                        note_text,
                        rules,
                        suppression_store=self.guardrail_suppressions,
                    )
                    for violation in violations:
                        payload_item = {
                            "rule_id": violation.rule_id,
                            "severity": violation.severity,
                            "description": violation.description,
                            "pattern": violation.pattern,
                            "line": note_text,
                        }
                        if violation.severity == "block":
                            guardrail_blocks.append(payload_item)
                        else:
                            guardrail_warnings.append(payload_item)

            if guardrail_blocks:
                first = guardrail_blocks[0]
                raise ValueError(
                    "Guardrail blocked external note apply: "
                    f"{first['description']} (rule {first['rule_id']})"
                )
            grouped: dict[KnowledgeTier, list[_CandidateNote]] = {
                tier: [] for tier in _SUPPORTED_TIERS
            }

            for note in notes:
                tier = note["tier"]
                grouped[tier].append(
                    _CandidateNote(
                        line=note["line"],
                        source_session_ids=tuple(
                            sorted(
                                {
                                    str(item).strip()
                                    for item in note["source_session_ids"]
                                    if str(item).strip()
                                }
                            )
                        ),
                    )
                )

            changed: dict[str, int] = {}
            notes_applied = 0
            conflicts: list[dict[str, str]] = []
            applied_source_session_ids: set[str] = set()
            evidence_notes: list[dict[str, Any]] = []

            for tier in _SUPPORTED_TIERS:
                candidates = grouped[tier]
                if not candidates:
                    continue
                current = self._read_target_tier(tier, write_target=target)
                merge_outcome = self._merge_lines(current, candidates, tier=tier)
                conflicts.extend(merge_outcome.conflicts)
                applied_for_tier = merge_outcome.notes_applied
                if applied_for_tier <= 0:
                    continue
                changed[tier.value] = applied_for_tier
                notes_applied += applied_for_tier
                for applied_note in merge_outcome.applied_notes:
                    for source_id in applied_note.source_session_ids:
                        applied_source_session_ids.add(source_id)
                    evidence_notes.append(
                        {
                            "tier": tier.value,
                            "line": applied_note.line,
                            "source_session_ids": list(applied_note.source_session_ids),
                        }
                    )
                if not dry_run:
                    self._write_target_tier(tier, merge_outcome.merged, write_target=target)

            sessions_marked = 0
            if mark_processed and not dry_run and applied_source_session_ids:
                sessions_marked = self._mark_sessions_processed(sorted(applied_source_session_ids))

            evidence_backlinks_written = 0
            if not dry_run and evidence_notes:
                evidence_backlinks_written = self._persist_evidence_backlinks(evidence_notes)

            result = {
                "write_target": target,
                "dry_run": dry_run,
                "notes_received": len(notes),
                "notes_applied": notes_applied,
                "tiers_changed": changed,
                "sessions_marked": sessions_marked,
                "conflict_policy": self.conflict_policy,
                "conflicts": conflicts,
                "evidence_backlinks_written": evidence_backlinks_written,
                "applied_notes": evidence_notes,
                "guardrail_warnings": guardrail_warnings,
                "guardrail_blocks": guardrail_blocks,
            }
            telemetry.record_event(
                run_id=run_id,
                stage=PipelineStage.APPLY,
                action=PipelineEventAction.COMPLETE,
                success=True,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={
                    "write_target": target,
                    "dry_run": dry_run,
                    "notes_received": len(notes),
                    "notes_applied": notes_applied,
                    "tiers_changed": changed,
                    "sessions_marked": sessions_marked,
                    "conflict_policy": self.conflict_policy,
                    "conflicts": len(conflicts),
                    "evidence_backlinks_written": evidence_backlinks_written,
                    "guardrail_warnings": len(guardrail_warnings),
                    "guardrail_blocks": len(guardrail_blocks),
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001
            telemetry.record_event(
                run_id=run_id,
                stage=PipelineStage.APPLY,
                action=PipelineEventAction.ERROR,
                success=False,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={"write_target": target, "dry_run": dry_run, "error": str(exc)},
            )
            raise

    def cleanup_state(self) -> dict[str, int]:
        """Remove stale/invalid external compaction state entries."""
        state = self._load_state()
        sessions_raw = state.get("sessions", {})
        if not isinstance(sessions_raw, dict):
            sessions_raw = {}

        removed_invalid = 0
        removed_stale = 0
        stale_ids: list[str] = []
        invalid_ids: list[str] = []

        for source_session_id, payload in sessions_raw.items():
            source_id = str(source_session_id).strip()
            source_hash = ""
            if isinstance(payload, dict):
                source_hash = str(payload.get("source_hash", "")).strip()
            if not source_id or not source_hash:
                removed_invalid += 1
                invalid_ids.append(source_id)
                continue
            if not self.storage.get_entries_by_source_session(source_id, limit=1):
                removed_stale += 1
                stale_ids.append(source_id)

        removed = 0
        if self._uses_storage_state_backend():
            for source_id in [*stale_ids, *invalid_ids]:
                removed += int(self.storage.delete_external_compaction_state(source_id) or 0)
        else:
            remaining: dict[str, Any] = {}
            for source_session_id, payload in sessions_raw.items():
                source_id = str(source_session_id).strip()
                if source_id in stale_ids or source_id in invalid_ids:
                    continue
                remaining[source_id] = payload
            removed = max(0, len(sessions_raw) - len(remaining))
            self._save_state({"sessions": remaining})

        return {
            "removed": removed,
            "removed_invalid": removed_invalid,
            "removed_stale": removed_stale,
        }

    def queue_notes_payload(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        notes = self._parse_notes(payload)
        if not self._uses_storage_queue_backend():
            raise RuntimeError("Queue operations require a storage backend with queue support.")
        queued = self.storage.enqueue_external_compaction_queue(
            [
                {
                    "tier": note["tier"].value,
                    "line": note["line"],
                    "source_session_ids": note["source_session_ids"],
                }
                for note in notes
            ],
            actor=actor,
        )
        return {"queued": len(queued), "items": queued}

    def list_queue(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._uses_storage_queue_backend():
            raise RuntimeError("Queue operations require a storage backend with queue support.")
        normalized_states = [str(item).strip().lower() for item in (states or []) if item]
        if any(state not in _QUEUE_STATES for state in normalized_states):
            raise ValueError("states must be one or more of pending, approved, rejected, applied")
        return self.storage.list_external_compaction_queue(
            states=normalized_states or None, limit=limit
        )

    def approve_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]:
        return self._transition_queue(ids=ids, target_state="approved", actor=actor)

    def reject_queue(self, *, ids: list[int], actor: str = "system") -> dict[str, int]:
        return self._transition_queue(ids=ids, target_state="rejected", actor=actor)

    def patch_preview(
        self,
        *,
        queue_ids: list[int] | None = None,
        states: list[str] | None = None,
        write_target: WriteTarget = "runtime",
    ) -> dict[str, Any]:
        target = self._normalize_write_target(write_target)
        queue = self.list_queue(states=states or ["approved"], limit=500)
        if queue_ids:
            wanted = {int(item) for item in queue_ids if int(item) > 0}
            queue = [item for item in queue if int(item.get("id", 0)) in wanted]

        by_tier: dict[KnowledgeTier, list[_CandidateNote]] = {tier: [] for tier in _SUPPORTED_TIERS}
        for item in queue:
            try:
                tier = KnowledgeTier(str(item.get("tier", "")).strip().upper())
            except ValueError:
                continue
            if tier not in _SUPPORTED_TIERS:
                continue
            line = str(item.get("line", "")).strip()
            if line and all(existing.line != line for existing in by_tier[tier]):
                by_tier[tier].append(_CandidateNote(line=line, source_session_ids=tuple()))

        preview_by_tier: dict[str, str] = {}
        combined: list[str] = []
        notes_considered = 0
        for tier in _SUPPORTED_TIERS:
            candidates = by_tier[tier]
            notes_considered += len(candidates)
            if not candidates:
                continue
            current = self._read_target_tier(tier, write_target=target)
            merge_outcome = self._merge_lines(current, candidates, tier=tier)
            diff_lines = list(
                difflib.unified_diff(
                    current.splitlines(),
                    merge_outcome.merged.splitlines(),
                    fromfile=f"{tier.value}.before",
                    tofile=f"{tier.value}.after",
                    lineterm="",
                )
            )
            if not diff_lines:
                continue
            rendered = "\n".join(diff_lines)
            preview_by_tier[tier.value] = rendered
            combined.append(rendered)

        return {
            "write_target": target,
            "notes_considered": notes_considered,
            "queue_items_considered": len(queue),
            "diff_by_tier": preview_by_tier,
            "combined_diff": "\n\n".join(combined),
        }

    def apply_approved_queue(
        self,
        *,
        actor: str = "system",
        write_target: WriteTarget = "runtime",
        mark_processed: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        queue = self.list_queue(states=["approved"], limit=500)
        if not queue:
            return {
                "queue_items_considered": 0,
                "queue_items_applied": 0,
                "notes_applied": 0,
                "tiers_changed": {},
                "dry_run": dry_run,
            }

        payload = {
            "notes": [
                {
                    "tier": str(item.get("tier", "")),
                    "line": str(item.get("line", "")),
                    "source_session_ids": list(item.get("source_session_ids", [])),
                }
                for item in queue
            ]
        }
        applied = self.apply_notes_payload(
            payload,
            write_target=write_target,
            dry_run=dry_run,
            mark_processed=mark_processed,
        )
        queue_applied = 0
        if not dry_run:
            applied_notes = applied.get("applied_notes", [])
            applied_keys: set[tuple[str, str, tuple[str, ...]]] = set()
            if isinstance(applied_notes, list):
                for note in applied_notes:
                    if not isinstance(note, dict):
                        continue
                    tier = str(note.get("tier", "")).strip().upper()
                    line = str(note.get("line", "")).strip()
                    source_ids_raw = note.get("source_session_ids", [])
                    source_ids = (
                        sorted({str(item).strip() for item in source_ids_raw if str(item).strip()})
                        if isinstance(source_ids_raw, list)
                        else []
                    )
                    if tier and line:
                        applied_keys.add((tier, line, tuple(source_ids)))

            ids = []
            for item in queue:
                item_id = int(item.get("id", 0))
                tier = str(item.get("tier", "")).strip().upper()
                line = str(item.get("line", "")).strip()
                source_ids_raw = item.get("source_session_ids", [])
                source_ids = (
                    sorted(
                        {str(source).strip() for source in source_ids_raw if str(source).strip()}
                    )
                    if isinstance(source_ids_raw, list)
                    else []
                )
                if item_id <= 0 or not tier or not line:
                    continue
                key = (tier, line, tuple(source_ids))
                if key in applied_keys:
                    ids.append(item_id)

            if ids:
                transition = self._transition_queue(ids=ids, target_state="applied", actor=actor)
                queue_applied = transition["updated"]

        return {
            "queue_items_considered": len(queue),
            "queue_items_applied": queue_applied,
            "notes_applied": int(applied.get("notes_applied", 0)),
            "tiers_changed": dict(applied.get("tiers_changed", {})),
            "dry_run": dry_run,
            "write_target": applied.get("write_target"),
            "sessions_marked": int(applied.get("sessions_marked", 0)),
            "conflict_policy": str(applied.get("conflict_policy", self.conflict_policy)),
            "conflicts": list(applied.get("conflicts", [])),
            "evidence_backlinks_written": int(applied.get("evidence_backlinks_written", 0)),
        }

    def _transition_queue(
        self,
        *,
        ids: list[int],
        target_state: str,
        actor: str,
    ) -> dict[str, int]:
        if not self._uses_storage_queue_backend():
            raise RuntimeError("Queue operations require a storage backend with queue support.")
        return self.storage.update_external_compaction_queue_state(
            ids=ids,
            target_state=target_state,
            actor=actor,
        )

    def _parse_notes(self, payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        validated = validate_external_notes_payload(payload)
        parsed: list[dict[str, Any]] = []
        for note in validated.notes:
            tier = KnowledgeTier(note.tier)
            parsed.append(
                {
                    "tier": tier,
                    "line": note.line,
                    "source_session_ids": list(note.source_session_ids),
                }
            )

        return parsed

    def _merge_lines(
        self,
        current: str,
        new_notes: list[_CandidateNote],
        *,
        tier: KnowledgeTier,
    ) -> _MergeOutcome:
        existing = current if current.strip() else TIER_HEADERS[tier]
        existing_lines = existing.splitlines()
        semantic_seen = {
            _semantic_key(line)
            for line in existing_lines
            if line.strip() and _is_candidate_line(line)
        }
        token_sets = [
            _token_set(line) for line in existing_lines if line.strip() and _is_candidate_line(line)
        ]
        topic_polarity: dict[str, tuple[str, str]] = {}
        for line in existing_lines:
            if not line.strip() or not _is_candidate_line(line):
                continue
            topic = _topic_key(line)
            if not topic:
                continue
            topic_polarity[topic] = (_polarity(line), line)

        working_lines = list(existing_lines)
        additions: list[str] = []
        applied_notes: list[_CandidateNote] = []
        conflicts: list[dict[str, str]] = []

        for note in new_notes:
            line = note.line.strip()
            if not line:
                continue
            semantic = _semantic_key(line)
            if semantic and semantic in semantic_seen:
                continue
            candidate_tokens = _token_set(line)
            if any(_jaccard_similarity(candidate_tokens, tokens) >= 0.85 for tokens in token_sets):
                continue

            topic = _topic_key(line)
            polarity = _polarity(line)
            if topic and topic in topic_polarity:
                existing_polarity, existing_line = topic_polarity[topic]
                if (
                    polarity != "neutral"
                    and existing_polarity != "neutral"
                    and existing_polarity != polarity
                ):
                    conflict = {
                        "tier": tier.value,
                        "existing_line": existing_line,
                        "incoming_line": line,
                        "policy": self.conflict_policy,
                    }
                    if self.conflict_policy == "queue_for_review":
                        conflicts.append(conflict)
                        continue
                    conflict["resolution"] = "prefer_newest"
                    conflicts.append(conflict)
                    if existing_line in working_lines:
                        working_lines.remove(existing_line)
                    semantic_seen.discard(_semantic_key(existing_line))
                    existing_tokens = _token_set(existing_line)
                    token_sets = [tokens for tokens in token_sets if tokens != existing_tokens]

            semantic_seen.add(semantic)
            token_sets.append(candidate_tokens)
            if topic:
                topic_polarity[topic] = (polarity, line)
            additions.append(line)
            applied_notes.append(note)

        if not additions:
            return _MergeOutcome(
                merged=existing,
                notes_applied=0,
                applied_notes=[],
                conflicts=conflicts,
            )

        merged = self._insert_lines_for_tier(working_lines, additions, tier=tier)
        return _MergeOutcome(
            merged=merged,
            notes_applied=len(applied_notes),
            applied_notes=applied_notes,
            conflicts=conflicts,
        )

    def _insert_lines_for_tier(
        self,
        existing_lines: list[str],
        additions: list[str],
        *,
        tier: KnowledgeTier,
    ) -> str:
        ralph_start = next(
            (index for index, line in enumerate(existing_lines) if line.startswith("## ")),
            len(existing_lines),
        )
        before_ralph = existing_lines[:ralph_start]
        ralph_lines = existing_lines[ralph_start:]

        if tier == KnowledgeTier.RECENT:
            first_entry = next(
                (index for index, line in enumerate(before_ralph) if _is_candidate_line(line)),
                len(before_ralph),
            )
            preamble_lines = before_ralph[:first_entry]
            entry_lines = before_ralph[first_entry:]
            parts = [
                "\n".join(preamble_lines).strip(),
                "\n".join(additions).strip(),
                "\n".join(entry_lines).strip(),
                "\n".join(ralph_lines).strip(),
            ]
        else:
            parts = [
                "\n".join(before_ralph).strip(),
                "\n".join(additions).strip(),
                "\n".join(ralph_lines).strip(),
            ]

        rendered_parts = [part for part in parts if part]
        if not rendered_parts:
            return "\n"
        return "\n\n".join(rendered_parts).rstrip() + "\n"

    def _persist_evidence_backlinks(self, notes: list[dict[str, Any]]) -> int:
        if not self.storage.capabilities.external_compaction_evidence:
            return 0
        try:
            return int(self.storage.record_external_compaction_evidence(notes))
        except UnsupportedStorageCapabilityError:
            return 0

    @staticmethod
    def _resolve_conflict_policy(config: dict[str, Any] | None) -> str:
        if not isinstance(config, dict):
            return "prefer_newest"
        compaction_cfg = config.get("compaction")
        if not isinstance(compaction_cfg, dict):
            return "prefer_newest"
        external_cfg = compaction_cfg.get("external")
        if not isinstance(external_cfg, dict):
            return "prefer_newest"
        raw_policy = str(external_cfg.get("conflict_policy", "prefer_newest")).strip().lower()
        if raw_policy in _CONFLICT_POLICIES:
            return raw_policy
        return "prefer_newest"

    def _template_path_for_tier(self, tier: KnowledgeTier) -> Path:
        return self.write_guard.template_path_for_tier(tier)

    def _read_target_tier(self, tier: KnowledgeTier, *, write_target: WriteTarget) -> str:
        if write_target == "runtime":
            return self.files.read_tier(tier)
        path = self._template_path_for_tier(tier)
        if path.exists():
            return path.read_text()
        return TIER_HEADERS[tier]

    def _write_target_tier(
        self,
        tier: KnowledgeTier,
        content: str,
        *,
        write_target: WriteTarget,
    ) -> None:
        if write_target == "runtime":
            self.files.write_tier(tier, content)
            return
        path = self._template_path_for_tier(tier)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @staticmethod
    def _serialize_entry(entry: LogEntry) -> dict[str, Any]:
        return {
            "id": str(entry.id),
            "timestamp": entry.timestamp.isoformat(),
            "label": entry.label.value,
            "confidence": entry.confidence,
            "content": entry.content,
            "tags": list(entry.tags),
            "source_session_id": entry.source_session_id,
        }

    def _hash_source_session(self, source_session_id: str) -> str:
        entries = self.storage.get_entries_by_source_session(source_session_id, limit=2000)
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(entry.timestamp.isoformat().encode("utf-8"))
            digest.update(b"|")
            digest.update(entry.label.value.encode("utf-8"))
            digest.update(b"|")
            digest.update(entry.content.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()[:24]

    def _load_state(self) -> dict[str, Any]:
        if self._uses_storage_state_backend():
            rows = self.storage.list_external_compaction_states()
            sessions: dict[str, dict[str, str]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                source_session_id = str(row.get("source_session_id", "")).strip()
                source_hash = str(row.get("source_hash", "")).strip()
                processed_at = str(row.get("processed_at", "")).strip()
                if not source_session_id:
                    continue
                sessions[source_session_id] = {
                    "source_hash": source_hash,
                    "processed_at": processed_at,
                }
            return {"sessions": sessions}

        if not self.state_path.exists():
            return {"sessions": {}}
        try:
            loaded = json.loads(self.state_path.read_text())
        except json.JSONDecodeError:
            return {"sessions": {}}
        if not isinstance(loaded, dict):
            return {"sessions": {}}
        sessions = loaded.get("sessions")
        if not isinstance(sessions, dict):
            loaded["sessions"] = {}
        return loaded

    def _save_state(self, state: dict[str, Any]) -> None:
        if self._uses_storage_state_backend():
            sessions = state.get("sessions", {})
            if not isinstance(sessions, dict):
                return
            for source_session_id, payload in sessions.items():
                if not isinstance(payload, dict):
                    continue
                source_id = str(source_session_id).strip()
                source_hash = str(payload.get("source_hash", "")).strip()
                processed_at = str(payload.get("processed_at", "")).strip()
                if not source_id or not source_hash or not processed_at:
                    continue
                self.storage.upsert_external_compaction_state(
                    source_id,
                    source_hash=source_hash,
                    processed_at=processed_at,
                )
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True))

    def _mark_sessions_processed(self, source_session_ids: list[str]) -> int:
        if not source_session_ids:
            return 0
        state = self._load_state()
        sessions = state.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
            state["sessions"] = sessions
        now = datetime.now(UTC).isoformat()
        marked = 0
        for source_session_id in source_session_ids:
            cleaned = source_session_id.strip()
            if not cleaned:
                continue
            source_hash = self._hash_source_session(cleaned)
            sessions[cleaned] = {"processed_at": now, "source_hash": source_hash}
            marked += 1
        self._save_state(state)
        return marked

    def _uses_storage_state_backend(self) -> bool:
        return bool(self.storage.capabilities.external_compaction_state)

    def _uses_storage_queue_backend(self) -> bool:
        return bool(self.storage.capabilities.external_compaction_queue)

    def _backfill_state_to_storage(self) -> None:
        if not self._uses_storage_state_backend() or not self.state_path.exists():
            return
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(loaded, dict):
            return
        sessions = loaded.get("sessions")
        if not isinstance(sessions, dict):
            return

        existing = {
            str(item.get("source_session_id", "")).strip()
            for item in self.storage.list_external_compaction_states()
            if isinstance(item, dict)
        }
        for source_session_id, payload in sessions.items():
            if not isinstance(payload, dict):
                continue
            source_id = str(source_session_id).strip()
            source_hash = str(payload.get("source_hash", "")).strip()
            processed_at = str(payload.get("processed_at", "")).strip()
            if not source_id or not source_hash:
                continue
            if not processed_at:
                processed_at = datetime.now(UTC).isoformat()
            if source_id in existing:
                continue
            self.storage.upsert_external_compaction_state(
                source_id,
                source_hash=source_hash,
                processed_at=processed_at,
            )

    def _normalize_write_target(self, write_target: WriteTarget | str) -> WriteTarget:
        return self.write_guard.resolve_target(str(write_target))
