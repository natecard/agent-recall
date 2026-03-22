from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.core.extract import TranscriptExtractor
from agent_recall.core.ordering import key_timestamp_desc_id
from agent_recall.core.telemetry import PipelineTelemetry
from agent_recall.ingest import SessionIngester, get_default_ingesters
from agent_recall.ingest.base import RawSession
from agent_recall.ingest.sources import normalize_source_name
from agent_recall.llm.base import LLMProvider, LLMRateLimitError
from agent_recall.storage.base import Storage
from agent_recall.storage.files import TIER_FILES, FileStorage, KnowledgeTier
from agent_recall.storage.models import PipelineEventAction, PipelineStage, SessionCheckpoint


@dataclass(frozen=True)
class _SessionCandidate:
    ingester: SessionIngester
    source_name: str
    session_path: Path
    session_id: str
    sort_timestamp: float


@dataclass(frozen=True)
class SyncDiscoverStage:
    active_ingesters: list[SessionIngester]
    candidates: list[_SessionCandidate]
    errors: list[str]


@dataclass(frozen=True)
class SyncFilterStage:
    candidates: list[_SessionCandidate]
    missing_session_ids: list[str]


@dataclass(frozen=True)
class SessionFilterStage:
    status: Literal["process", "skip_already_processed", "skip_empty", "failed_parse"]
    checkpoint: SessionCheckpoint | None
    is_fully_processed: bool
    raw_session: RawSession | None
    original_message_count: int | None
    message_count: int | None
    messages_filtered: bool
    content_hash: str | None
    error: str | None = None


@dataclass(frozen=True)
class SessionExtractStage:
    success: bool
    entries: list[Any]
    batch_events: list[dict[str, Any]]
    error: str | None
    duration_ms: float


@dataclass(frozen=True)
class SessionPersistStage:
    entries_written: int
    duration_ms: float


@dataclass(frozen=True)
class _CompactionDecision:
    should_compact: bool
    reason: str
    sessions_processed: int
    session_threshold: int
    recent_tokens: int
    token_threshold: int
    hours_since_recent: float | None
    age_threshold_hours: float | None


class AutoSync:
    """Discover, extract, and persist learnings from native agent sessions."""

    def __init__(
        self,
        storage: Storage,
        files: FileStorage,
        llm: LLMProvider | None,
        ingesters: list[SessionIngester] | None = None,
        project_path: Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.storage = storage
        self.files = files
        self.llm = llm
        self.extractor = TranscriptExtractor(llm) if llm else None
        self.ingesters = ingesters or get_default_ingesters(project_path)
        self.progress_callback = progress_callback
        self.extract_timeout_seconds = 45
        self.extract_retry_attempts = 3
        self.extract_retry_backoff_seconds = 2.0
        self.extract_retry_max_backoff_seconds = 45.0
        self.extract_rate_limit_min_messages_per_batch = 20
        self.zero_learning_warning_min_messages = 50

    async def sync(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        session_ids: list[str] | None = None,
        max_sessions: int | None = None,
        reset_checkpoints: bool = False,
        reset_full: bool = False,
        telemetry_run_id: str | None = None,
    ) -> dict[str, Any]:
        if self.extractor is None:
            msg = "LLM provider is required for sync"
            raise RuntimeError(msg)

        telemetry = PipelineTelemetry.from_config(
            agent_dir=self.files.agent_dir,
            config=self.files.read_config(),
        )
        run_id = telemetry_run_id or telemetry.create_run_id("sync")
        results = self._initial_sync_results()

        self._apply_reset_stage(
            reset_full=reset_full,
            reset_checkpoints=reset_checkpoints,
            sources=sources,
            session_ids=session_ids,
        )
        discover_stage = self._run_discover_stage(since=since, sources=sources)
        filter_stage = self._run_filter_stage(
            discover_stage=discover_stage,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )
        self._report_discover_stage(
            results=results,
            discover_stage=discover_stage,
            filter_stage=filter_stage,
        )

        for candidate in filter_stage.candidates:
            source_results = results["by_source"][candidate.source_name]
            filter_result = self._run_session_filter_stage(
                candidate,
                reset_checkpoints=reset_checkpoints,
            )

            if filter_result.status == "skip_already_processed":
                self._report_skip_already_processed(
                    results=results,
                    source_results=source_results,
                    candidate=candidate,
                    message_count=filter_result.original_message_count,
                )
                continue

            if filter_result.status == "skip_empty":
                raw_session = filter_result.raw_session
                content_hash = filter_result.content_hash
                if raw_session is None or content_hash is None:
                    msg = "Empty-session stage requires parsed session and content hash."
                    raise RuntimeError(msg)
                self._persist_empty_session(
                    candidate.session_id,
                    raw_session=raw_session,
                    content_hash=content_hash,
                    is_fully_processed=filter_result.is_fully_processed,
                )
                self._report_skip_empty(
                    results=results,
                    source_results=source_results,
                    candidate=candidate,
                    message_count=int(filter_result.message_count or 0),
                    messages_filtered=filter_result.messages_filtered,
                )
                continue

            if filter_result.status == "failed_parse":
                error_message = filter_result.error or "parse failure"
                self._report_failed_parse(
                    results=results,
                    candidate=candidate,
                    telemetry=telemetry,
                    run_id=run_id,
                    error=error_message,
                )
                continue

            raw_session = filter_result.raw_session
            content_hash = filter_result.content_hash
            if raw_session is None or content_hash is None:
                msg = "Process stage requires parsed session and content hash."
                raise RuntimeError(msg)

            extract_stage = await self._run_extract_stage(
                candidate,
                raw_session=raw_session,
                telemetry=telemetry,
                run_id=run_id,
            )
            if not extract_stage.success:
                self._report_failed_extraction(
                    results=results,
                    source_results=source_results,
                    candidate=candidate,
                    message_count=int(filter_result.message_count or 0),
                    error=extract_stage.error,
                )
                continue

            try:
                persist_stage = self._run_persist_stage(
                    candidate,
                    raw_session=raw_session,
                    content_hash=content_hash,
                    is_fully_processed=filter_result.is_fully_processed,
                    entries=extract_stage.entries,
                )
            except Exception as exc:  # noqa: BLE001
                self._report_failed_parse(
                    results=results,
                    candidate=candidate,
                    telemetry=telemetry,
                    run_id=run_id,
                    error=str(exc),
                )
                continue

            telemetry.record_event(
                run_id=run_id,
                stage=PipelineStage.INGEST,
                action=PipelineEventAction.COMPLETE,
                success=True,
                duration_ms=persist_stage.duration_ms,
                metadata={
                    "source": candidate.source_name,
                    "session_id": candidate.session_id,
                    "entries_written": persist_stage.entries_written,
                },
            )
            self._report_processed(
                results=results,
                source_results=source_results,
                candidate=candidate,
                filter_stage=filter_result,
                extract_stage=extract_stage,
            )

        return results

    @staticmethod
    def _initial_sync_results() -> dict[str, Any]:
        return {
            "sessions_discovered": 0,
            "sessions_processed": 0,
            "sessions_skipped": 0,
            "sessions_already_processed": 0,
            "sessions_incremental": 0,
            "empty_sessions": 0,
            "learnings_extracted": 0,
            "llm_requests": 0,
            "by_source": {},
            "session_diagnostics": [],
            "errors": [],
        }

    def _apply_reset_stage(
        self,
        *,
        reset_full: bool,
        reset_checkpoints: bool,
        sources: list[str] | None,
        session_ids: list[str] | None,
    ) -> None:
        if reset_full:
            self.storage.clear_processed_sessions()
            self.storage.clear_session_checkpoints()
            return
        if not reset_checkpoints:
            return
        if session_ids:
            for session_id in session_ids:
                self.storage.clear_session_checkpoints(source_session_id=session_id)
            return
        if sources:
            for source in sources:
                self.storage.clear_session_checkpoints(source=source)
            return
        self.storage.clear_session_checkpoints()

    def _run_discover_stage(
        self,
        *,
        since: datetime | None,
        sources: list[str] | None,
    ) -> SyncDiscoverStage:
        active_ingesters, candidates, errors = self._discover_candidates(
            since=since,
            sources=sources,
        )
        return SyncDiscoverStage(
            active_ingesters=active_ingesters,
            candidates=candidates,
            errors=errors,
        )

    def _run_filter_stage(
        self,
        *,
        discover_stage: SyncDiscoverStage,
        session_ids: list[str] | None,
        max_sessions: int | None,
    ) -> SyncFilterStage:
        candidates, missing_session_ids = self._apply_candidate_filters(
            candidates=discover_stage.candidates,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )
        return SyncFilterStage(candidates=candidates, missing_session_ids=missing_session_ids)

    def _report_discover_stage(
        self,
        *,
        results: dict[str, Any],
        discover_stage: SyncDiscoverStage,
        filter_stage: SyncFilterStage,
    ) -> None:
        results["errors"].extend(discover_stage.errors)
        self._seed_sync_source_results(results, discover_stage.active_ingesters)
        if filter_stage.missing_session_ids:
            missing = ", ".join(filter_stage.missing_session_ids)
            results["errors"].append(f"Requested session IDs not found: {missing}")
        results["sessions_discovered"] = len(filter_stage.candidates)
        self._populate_source_discovery_counts(results, filter_stage.candidates)

    def _run_session_filter_stage(
        self,
        candidate: _SessionCandidate,
        *,
        reset_checkpoints: bool,
    ) -> SessionFilterStage:
        checkpoint = self.storage.get_session_checkpoint(candidate.session_id)
        is_fully_processed = self.storage.is_session_processed(candidate.session_id)

        if is_fully_processed and checkpoint is None and not reset_checkpoints:
            return SessionFilterStage(
                status="skip_already_processed",
                checkpoint=checkpoint,
                is_fully_processed=is_fully_processed,
                raw_session=None,
                original_message_count=None,
                message_count=None,
                messages_filtered=False,
                content_hash=None,
            )

        try:
            raw_session = candidate.ingester.parse_session(candidate.session_path)
        except Exception as exc:  # noqa: BLE001
            return SessionFilterStage(
                status="failed_parse",
                checkpoint=checkpoint,
                is_fully_processed=is_fully_processed,
                raw_session=None,
                original_message_count=None,
                message_count=None,
                messages_filtered=False,
                content_hash=None,
                error=str(exc),
            )

        original_message_count = len(raw_session.messages)
        content_hash = self._compute_session_hash(raw_session)
        if checkpoint and checkpoint.content_hash == content_hash:
            return SessionFilterStage(
                status="skip_already_processed",
                checkpoint=checkpoint,
                is_fully_processed=is_fully_processed,
                raw_session=raw_session,
                original_message_count=original_message_count,
                message_count=original_message_count,
                messages_filtered=False,
                content_hash=content_hash,
            )

        filtered_session, messages_filtered = self._filter_messages_from_checkpoint(
            raw_session,
            checkpoint,
        )
        message_count = len(filtered_session.messages)
        if message_count < 2:
            return SessionFilterStage(
                status="skip_empty",
                checkpoint=checkpoint,
                is_fully_processed=is_fully_processed,
                raw_session=filtered_session,
                original_message_count=original_message_count,
                message_count=message_count,
                messages_filtered=messages_filtered,
                content_hash=content_hash,
            )

        return SessionFilterStage(
            status="process",
            checkpoint=checkpoint,
            is_fully_processed=is_fully_processed,
            raw_session=filtered_session,
            original_message_count=original_message_count,
            message_count=message_count,
            messages_filtered=messages_filtered,
            content_hash=content_hash,
        )

    def _persist_empty_session(
        self,
        session_id: str,
        *,
        raw_session: RawSession,
        content_hash: str,
        is_fully_processed: bool,
    ) -> None:
        self._update_checkpoint(session_id, raw_session, content_hash)
        if not is_fully_processed:
            self.storage.mark_session_processed(session_id)

    async def _run_extract_stage(
        self,
        candidate: _SessionCandidate,
        *,
        raw_session: RawSession,
        telemetry: PipelineTelemetry,
        run_id: str,
    ) -> SessionExtractStage:
        extractor = self.extractor
        if extractor is None:
            msg = "LLM provider is required for sync extraction stage"
            raise RuntimeError(msg)

        batch_events: list[dict[str, Any]] = []
        entries: list[Any] = []
        extraction_error: str | None = None
        extraction_started = time.perf_counter()
        message_count = len(raw_session.messages)
        initial_messages_per_batch = extractor.messages_per_batch
        current_attempt = 0
        self._emit_progress(
            {
                "event": "extraction_session_started",
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "messages_total": message_count,
                "messages_per_batch": initial_messages_per_batch,
            }
        )

        def _on_extract_progress(event: dict[str, Any]) -> None:
            event_payload = dict(event)
            if current_attempt > 0:
                event_payload.setdefault("attempt", current_attempt)
                event_payload.setdefault("max_attempts", self.extract_retry_attempts)
            if event_payload.get("event") == "extraction_batch_complete":
                batch_events.append(event_payload)
            self._emit_progress(event_payload)

        try:
            for attempt in range(1, self.extract_retry_attempts + 1):
                current_attempt = attempt
                try:
                    batch_events.clear()
                    entries = await asyncio.wait_for(
                        extractor.extract(
                            raw_session,
                            progress_callback=_on_extract_progress,
                        ),
                        timeout=self.extract_timeout_seconds,
                    )
                    extraction_error = None
                    break
                except TimeoutError:
                    extraction_error = (
                        f"{candidate.source_name}:{candidate.session_path.name}: "
                        f"extraction timed out after {self.extract_timeout_seconds}s "
                        f"(attempt {attempt}/{self.extract_retry_attempts})"
                    )
                    if attempt < self.extract_retry_attempts:
                        delay = self._compute_extract_retry_delay_seconds(
                            attempt=attempt,
                            retry_after_seconds=None,
                        )
                        self._emit_progress(
                            {
                                "event": "extraction_retry_scheduled",
                                "source": candidate.source_name,
                                "session_id": candidate.session_id,
                                "reason": "timeout",
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "max_attempts": self.extract_retry_attempts,
                                "delay_seconds": delay,
                                "messages_per_batch": extractor.messages_per_batch,
                            }
                        )
                        await asyncio.sleep(delay)
                except LLMRateLimitError as exc:
                    adjusted = self._maybe_reduce_extract_batch_size(extractor)
                    if adjusted is not None:
                        old_size, new_size = adjusted
                        self._emit_progress(
                            {
                                "event": "extraction_batch_size_adjusted",
                                "source": candidate.source_name,
                                "session_id": candidate.session_id,
                                "attempt": attempt,
                                "max_attempts": self.extract_retry_attempts,
                                "old_messages_per_batch": old_size,
                                "new_messages_per_batch": new_size,
                            }
                        )

                    retry_after = exc.retry_after_seconds
                    retry_after_text = (
                        f" (retry-after {retry_after:.1f}s)"
                        if isinstance(retry_after, float)
                        else ""
                    )
                    extraction_error = (
                        f"{candidate.source_name}:{candidate.session_path.name}: "
                        f"extraction rate-limited: {exc}{retry_after_text} "
                        f"(attempt {attempt}/{self.extract_retry_attempts})"
                    )
                    if attempt < self.extract_retry_attempts:
                        delay = self._compute_extract_retry_delay_seconds(
                            attempt=attempt,
                            retry_after_seconds=retry_after,
                        )
                        self._emit_progress(
                            {
                                "event": "extraction_retry_scheduled",
                                "source": candidate.source_name,
                                "session_id": candidate.session_id,
                                "reason": "rate_limit",
                                "attempt": attempt,
                                "next_attempt": attempt + 1,
                                "max_attempts": self.extract_retry_attempts,
                                "delay_seconds": delay,
                                "retry_after_seconds": retry_after,
                                "messages_per_batch": extractor.messages_per_batch,
                            }
                        )
                        await asyncio.sleep(delay)
                except Exception as exc:  # noqa: BLE001
                    extraction_error = (
                        f"{candidate.source_name}:{candidate.session_path.name}: "
                        f"extraction failed: {exc}"
                    )
                    break
        finally:
            extractor.messages_per_batch = initial_messages_per_batch

        extraction_duration_ms = (time.perf_counter() - extraction_started) * 1000.0
        if extraction_error:
            telemetry.record_event(
                run_id=run_id,
                stage=PipelineStage.EXTRACT,
                action=PipelineEventAction.ERROR,
                success=False,
                duration_ms=extraction_duration_ms,
                metadata={
                    "source": candidate.source_name,
                    "session_id": candidate.session_id,
                    "error": extraction_error,
                },
            )
            return SessionExtractStage(
                success=False,
                entries=[],
                batch_events=[],
                error=extraction_error,
                duration_ms=extraction_duration_ms,
            )

        telemetry.record_event(
            run_id=run_id,
            stage=PipelineStage.EXTRACT,
            action=PipelineEventAction.COMPLETE,
            success=True,
            duration_ms=extraction_duration_ms,
            metadata={
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "entries_extracted": len(entries),
                "llm_batches": len(batch_events),
            },
        )
        return SessionExtractStage(
            success=True,
            entries=entries,
            batch_events=batch_events,
            error=None,
            duration_ms=extraction_duration_ms,
        )

    def _run_persist_stage(
        self,
        candidate: _SessionCandidate,
        *,
        raw_session: RawSession,
        content_hash: str,
        is_fully_processed: bool,
        entries: list[Any],
    ) -> SessionPersistStage:
        started = time.perf_counter()
        for entry in entries:
            self.storage.append_entry(entry)
        self._update_checkpoint(candidate.session_id, raw_session, content_hash)
        if not is_fully_processed:
            self.storage.mark_session_processed(candidate.session_id)
        return SessionPersistStage(
            entries_written=len(entries),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _report_skip_already_processed(
        *,
        results: dict[str, Any],
        source_results: dict[str, int],
        candidate: _SessionCandidate,
        message_count: int | None,
    ) -> None:
        source_results["skipped"] += 1
        source_results["already_processed"] += 1
        results["sessions_skipped"] += 1
        results["sessions_already_processed"] += 1
        results["session_diagnostics"].append(
            {
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "status": "skipped_already_processed",
                "message_count": message_count,
                "learnings_extracted": 0,
            }
        )

    @staticmethod
    def _report_skip_empty(
        *,
        results: dict[str, Any],
        source_results: dict[str, int],
        candidate: _SessionCandidate,
        message_count: int,
        messages_filtered: bool,
    ) -> None:
        source_results["skipped"] += 1
        source_results["empty"] += 1
        results["sessions_skipped"] += 1
        results["empty_sessions"] += 1
        results["session_diagnostics"].append(
            {
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "status": "skipped_empty",
                "message_count": message_count,
                "learnings_extracted": 0,
                "incremental": messages_filtered,
            }
        )

    @staticmethod
    def _report_failed_parse(
        *,
        results: dict[str, Any],
        candidate: _SessionCandidate,
        telemetry: PipelineTelemetry,
        run_id: str,
        error: str,
    ) -> None:
        results["errors"].append(f"{candidate.source_name}:{candidate.session_path.name}: {error}")
        telemetry.record_event(
            run_id=run_id,
            stage=PipelineStage.INGEST,
            action=PipelineEventAction.ERROR,
            success=False,
            metadata={
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "error": error,
            },
        )
        results["session_diagnostics"].append(
            {
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "status": "failed_parse",
                "message_count": None,
                "learnings_extracted": 0,
                "error": error,
            }
        )

    @staticmethod
    def _report_failed_extraction(
        *,
        results: dict[str, Any],
        source_results: dict[str, int],
        candidate: _SessionCandidate,
        message_count: int,
        error: str | None,
    ) -> None:
        error_text = error or "extraction failed"
        results["errors"].append(error_text)
        source_results["skipped"] += 1
        source_results["extraction_failed"] += 1
        results["sessions_skipped"] += 1
        results["session_diagnostics"].append(
            {
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "status": "failed_extraction",
                "message_count": message_count,
                "learnings_extracted": 0,
                "error": error_text,
            }
        )

    def _report_processed(
        self,
        *,
        results: dict[str, Any],
        source_results: dict[str, int],
        candidate: _SessionCandidate,
        filter_stage: SessionFilterStage,
        extract_stage: SessionExtractStage,
    ) -> None:
        source_results["processed"] += 1
        source_results["learnings"] += len(extract_stage.entries)
        source_results["llm_batches"] += len(extract_stage.batch_events)
        results["sessions_processed"] += 1
        if filter_stage.messages_filtered:
            results["sessions_incremental"] += 1
        results["learnings_extracted"] += len(extract_stage.entries)
        results["llm_requests"] += len(extract_stage.batch_events)

        message_count = int(filter_stage.message_count or 0)
        diagnostic: dict[str, Any] = {
            "source": candidate.source_name,
            "session_id": candidate.session_id,
            "status": "processed",
            "message_count": message_count,
            "original_message_count": filter_stage.original_message_count,
            "learnings_extracted": len(extract_stage.entries),
            "incremental": filter_stage.messages_filtered,
        }
        if extract_stage.batch_events:
            diagnostic["llm_batches"] = len(extract_stage.batch_events)
        if message_count >= self.zero_learning_warning_min_messages and not extract_stage.entries:
            warning = (
                f"{candidate.source_name}:{candidate.session_id} has "
                f"{message_count} messages but yielded 0 learnings"
            )
            results["errors"].append(warning)
            diagnostic["warning"] = warning
        results["session_diagnostics"].append(diagnostic)

    async def sync_and_compact(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        session_ids: list[str] | None = None,
        max_sessions: int | None = None,
        force_compact: bool = False,
        reset_checkpoints: bool = False,
        reset_full: bool = False,
        skip_embeddings: bool = False,
    ) -> dict[str, Any]:
        telemetry = PipelineTelemetry.from_config(
            agent_dir=self.files.agent_dir,
            config=self.files.read_config(),
        )
        run_id = telemetry.create_run_id("sync")
        sync_results = await self.sync(
            since=since,
            sources=sources,
            session_ids=session_ids,
            max_sessions=max_sessions,
            reset_checkpoints=reset_checkpoints,
            reset_full=reset_full,
            telemetry_run_id=run_id,
        )

        learnings_extracted = int(sync_results["learnings_extracted"])
        if learnings_extracted > 0 or force_compact:
            decision = self._resolve_compaction_decision(
                sync_results=sync_results,
                force_compact=force_compact,
            )
            backend = self._resolve_compaction_backend()
            compact_started = time.perf_counter()
            if not decision.should_compact:
                sync_results["compaction"] = {
                    "backend": backend,
                    "deferred": True,
                    "deferred_reason": decision.reason,
                    "sessions_processed": decision.sessions_processed,
                    "session_threshold": decision.session_threshold,
                    "recent_tokens": decision.recent_tokens,
                    "token_threshold": decision.token_threshold,
                    "hours_since_recent": decision.hours_since_recent,
                    "age_threshold_hours": decision.age_threshold_hours,
                    "guardrails_updated": False,
                    "style_updated": False,
                    "recent_updated": False,
                    "chunks_indexed": 0,
                    "llm_requests": 0,
                    "llm_responses": 0,
                }
                telemetry.record_event(
                    run_id=run_id,
                    stage=PipelineStage.COMPACT,
                    action=PipelineEventAction.COMPLETE,
                    success=True,
                    duration_ms=(time.perf_counter() - compact_started) * 1000.0,
                    metadata={
                        "backend": backend,
                        "deferred": True,
                        "reason": decision.reason,
                        "sessions_processed": decision.sessions_processed,
                        "session_threshold": decision.session_threshold,
                        "recent_tokens": decision.recent_tokens,
                        "token_threshold": decision.token_threshold,
                        "hours_since_recent": decision.hours_since_recent,
                        "age_threshold_hours": decision.age_threshold_hours,
                    },
                )
            elif backend == "mcp_external":
                pending_limit = self._resolve_external_pending_limit()
                pending_sessions = self.storage.list_recent_source_sessions(limit=pending_limit)
                sync_results["compaction"] = {
                    "backend": "mcp_external",
                    "external_required": True,
                    "pending_external_conversations": len(pending_sessions),
                    "guardrails_updated": False,
                    "style_updated": False,
                    "recent_updated": False,
                    "chunks_indexed": 0,
                    "llm_requests": 0,
                    "llm_responses": 0,
                }
                telemetry.record_event(
                    run_id=run_id,
                    stage=PipelineStage.COMPACT,
                    action=PipelineEventAction.COMPLETE,
                    success=True,
                    duration_ms=(time.perf_counter() - compact_started) * 1000.0,
                    metadata={
                        "backend": "mcp_external",
                        "external_required": True,
                        "pending_external_conversations": len(pending_sessions),
                    },
                )
            else:
                if self.llm is None:
                    msg = "LLM provider is required for compaction"
                    raise RuntimeError(msg)
                compact_engine = CompactionEngine(self.storage, self.files, self.llm)
                try:
                    sync_results["compaction"] = await compact_engine.compact(force=force_compact)
                except Exception as exc:  # noqa: BLE001
                    telemetry.record_event(
                        run_id=run_id,
                        stage=PipelineStage.COMPACT,
                        action=PipelineEventAction.ERROR,
                        success=False,
                        duration_ms=(time.perf_counter() - compact_started) * 1000.0,
                        metadata={"backend": backend, "error": str(exc)},
                    )
                    raise
                telemetry.record_event(
                    run_id=run_id,
                    stage=PipelineStage.COMPACT,
                    action=PipelineEventAction.COMPLETE,
                    success=True,
                    duration_ms=(time.perf_counter() - compact_started) * 1000.0,
                    metadata={
                        "backend": backend,
                        "guardrails_updated": bool(
                            sync_results["compaction"].get("guardrails_updated")
                        ),
                        "style_updated": bool(sync_results["compaction"].get("style_updated")),
                        "recent_updated": bool(sync_results["compaction"].get("recent_updated")),
                    },
                )

        if self._embedding_indexing_enabled() and not skip_embeddings:
            indexer = EmbeddingIndexer(self.storage)
            sync_results["embedding_indexing"] = indexer.index_missing_embeddings()

        return sync_results

    def _resolve_compaction_decision(
        self,
        *,
        sync_results: dict[str, Any],
        force_compact: bool,
    ) -> _CompactionDecision:
        sessions_processed = int(sync_results.get("sessions_processed", 0))
        recent_tokens = self._estimate_recent_tokens()
        hours_since_recent = self._hours_since_recent_tier_update()

        config = self.files.read_config()
        compaction_cfg = config.get("compaction") if isinstance(config, dict) else {}
        if not isinstance(compaction_cfg, dict):
            compaction_cfg = {}

        session_threshold = self._coerce_positive_int(
            compaction_cfg.get("max_sessions_before_compact"),
            default=5,
        )
        token_threshold = self._coerce_positive_int(
            compaction_cfg.get("max_recent_tokens"),
            default=1500,
        )
        age_threshold_hours = self._coerce_nonnegative_float_or_none(
            compaction_cfg.get("max_hours_before_compact"),
            default=24.0,
        )

        if force_compact:
            return _CompactionDecision(
                should_compact=True,
                reason="forced",
                sessions_processed=sessions_processed,
                session_threshold=session_threshold,
                recent_tokens=recent_tokens,
                token_threshold=token_threshold,
                hours_since_recent=hours_since_recent,
                age_threshold_hours=age_threshold_hours,
            )

        session_reached = sessions_processed >= session_threshold
        token_reached = recent_tokens >= token_threshold
        age_reached = False
        if age_threshold_hours is not None:
            if hours_since_recent is None:
                age_reached = True
            else:
                age_reached = hours_since_recent >= age_threshold_hours

        if session_reached:
            reason = "session_threshold"
        elif token_reached:
            reason = "token_threshold"
        elif age_reached:
            reason = "age_threshold"
        else:
            reason = "below_thresholds"

        return _CompactionDecision(
            should_compact=session_reached or token_reached or age_reached,
            reason=reason,
            sessions_processed=sessions_processed,
            session_threshold=session_threshold,
            recent_tokens=recent_tokens,
            token_threshold=token_threshold,
            hours_since_recent=hours_since_recent,
            age_threshold_hours=age_threshold_hours,
        )

    def _estimate_recent_tokens(self) -> int:
        content = self.files.read_tier(KnowledgeTier.RECENT).strip()
        if not content:
            return 0
        return max(1, (len(content) + 3) // 4)

    def _hours_since_recent_tier_update(self) -> float | None:
        path = self._resolve_recent_tier_path()
        if path is None:
            return None
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            return None
        elapsed = (datetime.now(UTC) - modified_at).total_seconds() / 3600.0
        return max(0.0, elapsed)

    def _resolve_recent_tier_path(self) -> Path | None:
        shared_dir = self.files.shared_tiers_dir
        if shared_dir is not None:
            shared_path = shared_dir / TIER_FILES[KnowledgeTier.RECENT]
            if shared_path.exists():
                return shared_path
        local_path = self.files.agent_dir / TIER_FILES[KnowledgeTier.RECENT]
        if local_path.exists():
            return local_path
        return None

    @staticmethod
    def _coerce_positive_int(raw_value: Any, *, default: int) -> int:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, parsed)

    @staticmethod
    def _coerce_nonnegative_float_or_none(raw_value: Any, *, default: float) -> float | None:
        if raw_value is None:
            return default
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            parsed = default
        if parsed < 0:
            return None
        return parsed

    def _embedding_indexing_enabled(self) -> bool:
        config = self.files.read_config()
        if not isinstance(config, dict):
            return False

        retrieval_cfg = config.get("retrieval")
        if not isinstance(retrieval_cfg, dict):
            return False

        return bool(retrieval_cfg.get("semantic_index_enabled", False))

    def _resolve_compaction_backend(self) -> str:
        config = self.files.read_config()
        if not isinstance(config, dict):
            return "llm"
        compaction_cfg = config.get("compaction")
        if not isinstance(compaction_cfg, dict):
            return "llm"
        backend = str(compaction_cfg.get("backend", "llm")).strip().lower()
        if backend in {"llm", "coding_cli", "mcp_external"}:
            return backend
        return "llm"

    def _resolve_external_pending_limit(self) -> int:
        config = self.files.read_config()
        if not isinstance(config, dict):
            return 20
        compaction_cfg = config.get("compaction")
        if not isinstance(compaction_cfg, dict):
            return 20
        external_cfg = compaction_cfg.get("external")
        if not isinstance(external_cfg, dict):
            return 20
        raw_limit = external_cfg.get("pending_limit", 20)
        try:
            parsed = int(raw_limit)
        except (TypeError, ValueError):
            parsed = 20
        return max(1, parsed)

    def list_sessions(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        session_ids: list[str] | None = None,
        max_sessions: int | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "sessions_discovered": 0,
            "by_source": {},
            "sessions": [],
            "errors": [],
        }

        discover_stage = self._run_discover_stage(since=since, sources=sources)
        results["errors"].extend(discover_stage.errors)

        for ingester in discover_stage.active_ingesters:
            results["by_source"][ingester.source_name] = {
                "discovered": 0,
                "listed": 0,
            }

        filter_stage = self._run_filter_stage(
            discover_stage=discover_stage,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )
        if filter_stage.missing_session_ids:
            missing = ", ".join(filter_stage.missing_session_ids)
            results["errors"].append(f"Requested session IDs not found: {missing}")

        results["sessions_discovered"] = len(filter_stage.candidates)
        for candidate in filter_stage.candidates:
            source_stats = results["by_source"].setdefault(
                candidate.source_name,
                {"discovered": 0, "listed": 0},
            )
            source_stats["discovered"] += 1

            row: dict[str, Any] = {
                "source": candidate.source_name,
                "session_id": candidate.session_id,
                "title": None,
                "started_at": None,
                "ended_at": None,
                "message_count": 0,
                "processed": self.storage.is_session_processed(candidate.session_id),
                "session_path": candidate.session_path,
            }

            try:
                raw_session = candidate.ingester.parse_session(candidate.session_path)
                row["title"] = raw_session.title
                row["started_at"] = raw_session.started_at
                row["ended_at"] = raw_session.ended_at
                row["message_count"] = len(raw_session.messages)
                row["project_path"] = raw_session.project_path
            except Exception as exc:  # noqa: BLE001
                row["parse_error"] = str(exc)
                results["errors"].append(
                    f"{candidate.source_name}:{candidate.session_path.name}: {exc}"
                )

            source_stats["listed"] += 1
            results["sessions"].append(row)

        return results

    def get_source_status(self) -> dict[str, dict[str, Any]]:
        status: dict[str, dict[str, Any]] = {}

        for ingester in self.ingesters:
            try:
                sessions = ingester.discover_sessions()
                status[ingester.source_name] = {
                    "available": True,
                    "sessions_found": len(sessions),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                status[ingester.source_name] = {
                    "available": False,
                    "sessions_found": 0,
                    "error": str(exc),
                }

        return status

    def _discover_candidates(
        self,
        since: datetime | None,
        sources: list[str] | None,
    ) -> tuple[list[SessionIngester], list[_SessionCandidate], list[str]]:
        active_ingesters = self._select_ingesters(sources)
        candidates: list[_SessionCandidate] = []
        errors: list[str] = []

        for ingester in active_ingesters:
            try:
                session_paths = ingester.discover_sessions(since=since)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ingester.source_name}: {exc}")
                continue

            for session_path in session_paths:
                try:
                    session_id = ingester.get_session_id(session_path)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{ingester.source_name}:{session_path.name}: {exc}")
                    continue

                candidates.append(
                    _SessionCandidate(
                        ingester=ingester,
                        source_name=ingester.source_name,
                        session_path=session_path,
                        session_id=session_id,
                        sort_timestamp=self._session_sort_timestamp(
                            session_path=session_path,
                            session_id=session_id,
                        ),
                    )
                )

        return active_ingesters, candidates, errors

    def _seed_sync_source_results(
        self,
        results: dict[str, Any],
        active_ingesters: list[SessionIngester],
    ) -> None:
        by_source = results["by_source"]
        for ingester in active_ingesters:
            by_source[ingester.source_name] = {
                "discovered": 0,
                "processed": 0,
                "skipped": 0,
                "already_processed": 0,
                "extraction_failed": 0,
                "empty": 0,
                "learnings": 0,
                "llm_batches": 0,
            }

    @staticmethod
    def _session_sort_timestamp(session_path: Path, session_id: str) -> float:
        try:
            if session_path.exists():
                return float(session_path.stat().st_mtime)
        except OSError:
            pass

        token = session_id.rsplit("-", 1)[-1]
        if token.isdigit():
            numeric = float(token)
            if numeric > 1e12:
                return numeric / 1000
            return numeric

        return 0.0

    @staticmethod
    def _apply_candidate_filters(
        candidates: list[_SessionCandidate],
        session_ids: list[str] | None,
        max_sessions: int | None,
    ) -> tuple[list[_SessionCandidate], list[str]]:
        selected = list(candidates)
        missing_ids: list[str] = []

        if session_ids:
            requested_ids = {session_id.strip() for session_id in session_ids if session_id.strip()}
            selected = [
                candidate for candidate in selected if candidate.session_id in requested_ids
            ]
            found_ids = {candidate.session_id for candidate in selected}
            missing_ids = sorted(requested_ids - found_ids)

        selected.sort(
            key=lambda candidate: key_timestamp_desc_id(
                candidate.sort_timestamp,
                candidate.session_id,
            )
        )

        if max_sessions is not None:
            selected = selected[:max_sessions]

        return selected, missing_ids

    def _populate_source_discovery_counts(
        self,
        results: dict[str, Any],
        candidates: list[_SessionCandidate],
    ) -> None:
        by_source = results["by_source"]
        for candidate in candidates:
            source_results = by_source.setdefault(
                candidate.source_name,
                {
                    "discovered": 0,
                    "processed": 0,
                    "skipped": 0,
                    "already_processed": 0,
                    "extraction_failed": 0,
                    "empty": 0,
                    "learnings": 0,
                    "llm_batches": 0,
                },
            )
            source_results["discovered"] += 1

    def _select_ingesters(self, sources: list[str] | None) -> list[SessionIngester]:
        if not sources:
            return self.ingesters

        wanted = {self._normalize_source(source) for source in sources}
        return [
            ingester
            for ingester in self.ingesters
            if self._normalize_source(ingester.source_name) in wanted
        ]

    @staticmethod
    def _normalize_source(source: str) -> str:
        return normalize_source_name(source)

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:  # noqa: BLE001
            return

    def _compute_extract_retry_delay_seconds(
        self,
        *,
        attempt: int,
        retry_after_seconds: float | None,
    ) -> float:
        base_backoff = max(0.0, float(self.extract_retry_backoff_seconds))
        exponential_backoff = base_backoff * float(2 ** max(0, attempt - 1))
        retry_after = max(0.0, float(retry_after_seconds or 0.0))
        delay = max(exponential_backoff, retry_after)
        max_backoff = max(0.0, float(self.extract_retry_max_backoff_seconds))
        if max_backoff > 0:
            delay = min(delay, max_backoff)
        return delay

    def _maybe_reduce_extract_batch_size(
        self,
        extractor: TranscriptExtractor,
    ) -> tuple[int, int] | None:
        current_size = max(1, int(extractor.messages_per_batch))
        min_size = max(1, int(self.extract_rate_limit_min_messages_per_batch))
        if current_size <= min_size:
            return None
        next_size = max(min_size, current_size // 2)
        if next_size >= current_size:
            return None
        extractor.messages_per_batch = next_size
        return current_size, next_size

    def _compute_session_hash(self, raw_session: RawSession) -> str:
        """Compute a hash of session content for change detection."""
        content_parts = []
        for msg in raw_session.messages:
            content_parts.append(f"{msg.role}:{msg.content}")
        content_str = "|".join(content_parts)
        return hashlib.sha256(content_str.encode()).hexdigest()[:32]

    def _filter_messages_from_checkpoint(
        self,
        raw_session: RawSession,
        checkpoint: SessionCheckpoint | None,
    ) -> tuple[RawSession, bool]:
        """Filter messages to only those after the checkpoint.

        Returns:
            Tuple of (filtered session, bool indicating if messages were filtered)
        """
        if checkpoint is None:
            return raw_session, False

        if checkpoint.last_message_index is not None:
            # Index-based filtering
            if checkpoint.last_message_index < len(raw_session.messages) - 1:
                filtered_messages = raw_session.messages[checkpoint.last_message_index + 1 :]
                return raw_session.model_copy(update={"messages": filtered_messages}), True
            return raw_session, False

        if checkpoint.last_message_timestamp is not None:
            # Timestamp-based filtering
            checkpoint_time = checkpoint.last_message_timestamp
            filtered_messages = [
                msg
                for msg in raw_session.messages
                if msg.timestamp is None or msg.timestamp > checkpoint_time
            ]
            if len(filtered_messages) < len(raw_session.messages):
                return raw_session.model_copy(update={"messages": filtered_messages}), True
            return raw_session, False

        return raw_session, False

    def _update_checkpoint(
        self,
        session_id: str,
        raw_session: RawSession,
        content_hash: str,
    ) -> None:
        """Update checkpoint after processing a session."""
        if not raw_session.messages:
            return

        last_message = raw_session.messages[-1]
        last_index = len(raw_session.messages) - 1
        last_timestamp = last_message.timestamp

        checkpoint = SessionCheckpoint(
            source_session_id=session_id,
            last_message_index=last_index,
            last_message_timestamp=last_timestamp,
            content_hash=content_hash,
        )
        self.storage.save_session_checkpoint(checkpoint)
