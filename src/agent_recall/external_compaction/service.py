from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.core.tier_writer import TIER_HEADERS
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import LogEntry, PipelineEventAction, PipelineStage

WriteTarget = Literal["runtime", "templates"]

_SUPPORTED_TIERS = (
    KnowledgeTier.GUARDRAILS,
    KnowledgeTier.STYLE,
    KnowledgeTier.RECENT,
)

_TEMPLATE_FILE_BY_TIER: dict[KnowledgeTier, str] = {
    KnowledgeTier.GUARDRAILS: "GUARDRAILS.md",
    KnowledgeTier.STYLE: "STYLE.md",
    KnowledgeTier.RECENT: "RECENT.md",
}


def _normalize_line(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _to_iso(value: object) -> str | None:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    ) -> None:
        self.storage = storage
        self.files = files
        self.agent_dir = agent_dir
        self.repo_root = repo_root.resolve()
        self.state_path = state_path or (agent_dir / "external_compaction_state.json")
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

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "write_target": target,
            "tiers": self.read_tiers(write_target=target),
            "conversations": conversations,
            "notes_schema": {
                "notes": [
                    {
                        "tier": "GUARDRAILS|STYLE|RECENT",
                        "line": "- [TYPE] concise note (or RECENT date summary line)",
                        "source_session_ids": ["source-session-id"],
                    }
                ]
            },
        }

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
            grouped: dict[KnowledgeTier, list[str]] = {tier: [] for tier in _SUPPORTED_TIERS}
            source_session_ids: set[str] = set()

            for note in notes:
                tier = note["tier"]
                line = note["line"]
                if line not in grouped[tier]:
                    grouped[tier].append(line)
                for source_id in note["source_session_ids"]:
                    source_session_ids.add(source_id)

            changed: dict[str, int] = {}
            notes_applied = 0

            for tier in _SUPPORTED_TIERS:
                candidate_lines = grouped[tier]
                if not candidate_lines:
                    continue
                current = self._read_target_tier(tier, write_target=target)
                merged, applied_for_tier = self._merge_lines(current, candidate_lines, tier=tier)
                if applied_for_tier <= 0:
                    continue
                changed[tier.value] = applied_for_tier
                notes_applied += applied_for_tier
                if not dry_run:
                    self._write_target_tier(tier, merged, write_target=target)

            sessions_marked = 0
            if mark_processed and not dry_run and source_session_ids:
                sessions_marked = self._mark_sessions_processed(sorted(source_session_ids))

            result = {
                "write_target": target,
                "dry_run": dry_run,
                "notes_received": len(notes),
                "notes_applied": notes_applied,
                "tiers_changed": changed,
                "sessions_marked": sessions_marked,
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
            delete_fn = getattr(self.storage, "delete_external_compaction_state", None)
            if callable(delete_fn):
                for source_id in [*stale_ids, *invalid_ids]:
                    removed += int(delete_fn(source_id) or 0)
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

    def _parse_notes(self, payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            raw_notes = payload
        elif isinstance(payload, dict):
            container = payload.get("notes")
            if isinstance(container, list):
                raw_notes = container
            else:
                raw_notes = []
        else:
            raw_notes = []

        parsed: list[dict[str, Any]] = []
        for item in raw_notes:
            if not isinstance(item, dict):
                continue

            raw_tier = str(item.get("tier", "")).strip().upper()
            if not raw_tier:
                continue
            try:
                tier = KnowledgeTier(raw_tier)
            except ValueError:
                continue
            if tier not in _SUPPORTED_TIERS:
                continue

            line = str(item.get("line", "")).strip()
            if not line or line.startswith("#"):
                continue

            if tier in {KnowledgeTier.GUARDRAILS, KnowledgeTier.STYLE} and not line.startswith(
                "- "
            ):
                continue
            if tier == KnowledgeTier.RECENT and not (
                line.startswith("- ") or line.startswith("**")
            ):
                continue

            source_ids_raw = item.get("source_session_ids", [])
            source_ids: list[str] = []
            if isinstance(source_ids_raw, list):
                for source_id in source_ids_raw:
                    cleaned = str(source_id).strip()
                    if cleaned:
                        source_ids.append(cleaned)

            parsed.append(
                {
                    "tier": tier,
                    "line": line,
                    "source_session_ids": source_ids,
                }
            )

        return parsed

    def _merge_lines(
        self,
        current: str,
        new_lines: list[str],
        *,
        tier: KnowledgeTier,
    ) -> tuple[str, int]:
        existing = current if current.strip() else TIER_HEADERS[tier]
        existing_lines = existing.splitlines()
        seen = {_normalize_line(line) for line in existing_lines if line.strip()}

        additions: list[str] = []
        for line in new_lines:
            normalized = _normalize_line(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            additions.append(line)

        if not additions:
            return existing, 0

        merged = existing.rstrip() + "\n\n" + "\n".join(additions) + "\n"
        return merged, len(additions)

    def _template_path_for_tier(self, tier: KnowledgeTier) -> Path:
        relative = Path("src") / "agent_recall" / "templates" / _TEMPLATE_FILE_BY_TIER[tier]
        path = (self.repo_root / relative).resolve()
        if self.repo_root not in path.parents:
            raise ValueError(f"Template path escapes repo root: {path}")
        return path

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
            list_fn = getattr(self.storage, "list_external_compaction_states", None)
            if callable(list_fn):
                rows = list_fn()
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
            upsert_fn = getattr(self.storage, "upsert_external_compaction_state", None)
            if not callable(upsert_fn):
                return
            for source_session_id, payload in sessions.items():
                if not isinstance(payload, dict):
                    continue
                source_id = str(source_session_id).strip()
                source_hash = str(payload.get("source_hash", "")).strip()
                processed_at = str(payload.get("processed_at", "")).strip()
                if not source_id or not source_hash or not processed_at:
                    continue
                upsert_fn(source_id, source_hash=source_hash, processed_at=processed_at)
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
        return all(
            callable(getattr(self.storage, attr, None))
            for attr in (
                "list_external_compaction_states",
                "upsert_external_compaction_state",
                "delete_external_compaction_state",
            )
        )

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

        list_fn = getattr(self.storage, "list_external_compaction_states", None)
        upsert_fn = getattr(self.storage, "upsert_external_compaction_state", None)
        if not callable(list_fn) or not callable(upsert_fn):
            return

        existing = {
            str(item.get("source_session_id", "")).strip()
            for item in list_fn()
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
            upsert_fn(source_id, source_hash=source_hash, processed_at=processed_at)

    @staticmethod
    def _normalize_write_target(write_target: WriteTarget | str) -> WriteTarget:
        value = str(write_target).strip().lower()
        if value not in {"runtime", "templates"}:
            raise ValueError("write_target must be 'runtime' or 'templates'")
        return value  # type: ignore[return-value]
