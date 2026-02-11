from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.extract import TranscriptExtractor
from agent_recall.ingest import SessionIngester, get_default_ingesters
from agent_recall.llm.base import LLMProvider, LLMRateLimitError
from agent_recall.storage.files import FileStorage
from agent_recall.storage.sqlite import SQLiteStorage


@dataclass(frozen=True)
class _SessionCandidate:
    ingester: SessionIngester
    source_name: str
    session_path: Path
    session_id: str
    sort_timestamp: float


class AutoSync:
    """Discover, extract, and persist learnings from native agent sessions."""

    def __init__(
        self,
        storage: SQLiteStorage,
        files: FileStorage,
        llm: LLMProvider | None,
        ingesters: list[SessionIngester] | None = None,
        project_path: Path | None = None,
    ):
        self.storage = storage
        self.files = files
        self.llm = llm
        self.extractor = TranscriptExtractor(llm) if llm else None
        self.ingesters = ingesters or get_default_ingesters(project_path)
        self.extract_timeout_seconds = 45
        self.extract_retry_attempts = 3
        self.extract_retry_backoff_seconds = 2.0
        self.zero_learning_warning_min_messages = 50

    async def sync(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        session_ids: list[str] | None = None,
        max_sessions: int | None = None,
    ) -> dict[str, Any]:
        if self.extractor is None:
            msg = "LLM provider is required for sync"
            raise RuntimeError(msg)

        extractor = self.extractor

        results: dict[str, Any] = {
            "sessions_discovered": 0,
            "sessions_processed": 0,
            "sessions_skipped": 0,
            "sessions_already_processed": 0,
            "empty_sessions": 0,
            "learnings_extracted": 0,
            "by_source": {},
            "session_diagnostics": [],
            "errors": [],
        }

        active_ingesters, candidates, discovery_errors = self._discover_candidates(
            since=since,
            sources=sources,
        )
        results["errors"].extend(discovery_errors)
        self._seed_sync_source_results(results, active_ingesters)

        selected_candidates, missing_ids = self._apply_candidate_filters(
            candidates=candidates,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )
        if missing_ids:
            results["errors"].append(
                f"Requested session IDs not found: {', '.join(missing_ids)}"
            )

        results["sessions_discovered"] = len(selected_candidates)
        self._populate_source_discovery_counts(results, selected_candidates)

        for candidate in selected_candidates:
            source_results = results["by_source"][candidate.source_name]

            if self.storage.is_session_processed(candidate.session_id):
                source_results["skipped"] += 1
                source_results["already_processed"] += 1
                results["sessions_skipped"] += 1
                results["sessions_already_processed"] += 1
                results["session_diagnostics"].append(
                    {
                        "source": candidate.source_name,
                        "session_id": candidate.session_id,
                        "status": "skipped_already_processed",
                        "message_count": None,
                        "learnings_extracted": 0,
                    }
                )
                continue

            try:
                raw_session = candidate.ingester.parse_session(candidate.session_path)
                message_count = len(raw_session.messages)
                if message_count < 2:
                    self.storage.mark_session_processed(candidate.session_id)
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
                        }
                    )
                    continue

                entries: list[Any] = []
                extraction_error: str | None = None
                for attempt in range(1, self.extract_retry_attempts + 1):
                    try:
                        entries = await asyncio.wait_for(
                            extractor.extract(raw_session),
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
                            await asyncio.sleep(self.extract_retry_backoff_seconds * attempt)
                    except LLMRateLimitError as exc:
                        extraction_error = (
                            f"{candidate.source_name}:{candidate.session_path.name}: "
                            f"extraction rate-limited: {exc} "
                            f"(attempt {attempt}/{self.extract_retry_attempts})"
                        )
                        if attempt < self.extract_retry_attempts:
                            await asyncio.sleep(self.extract_retry_backoff_seconds * attempt)
                    except Exception as exc:  # noqa: BLE001
                        extraction_error = (
                            f"{candidate.source_name}:{candidate.session_path.name}: "
                            f"extraction failed: {exc}"
                        )
                        break

                if extraction_error:
                    results["errors"].append(extraction_error)
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
                            "error": extraction_error,
                        }
                    )
                    continue

                for entry in entries:
                    self.storage.append_entry(entry)

                self.storage.mark_session_processed(candidate.session_id)

                source_results["processed"] += 1
                source_results["learnings"] += len(entries)
                results["sessions_processed"] += 1
                results["learnings_extracted"] += len(entries)
                diagnostic: dict[str, Any] = {
                    "source": candidate.source_name,
                    "session_id": candidate.session_id,
                    "status": "processed",
                    "message_count": message_count,
                    "learnings_extracted": len(entries),
                }
                if (
                    message_count >= self.zero_learning_warning_min_messages
                    and len(entries) == 0
                ):
                    warning = (
                        f"{candidate.source_name}:{candidate.session_id} has "
                        f"{message_count} messages but yielded 0 learnings"
                    )
                    results["errors"].append(warning)
                    diagnostic["warning"] = warning
                results["session_diagnostics"].append(diagnostic)
            except Exception as exc:  # noqa: BLE001
                results["errors"].append(
                    f"{candidate.source_name}:{candidate.session_path.name}: {exc}"
                )
                results["session_diagnostics"].append(
                    {
                        "source": candidate.source_name,
                        "session_id": candidate.session_id,
                        "status": "failed_parse",
                        "message_count": None,
                        "learnings_extracted": 0,
                        "error": str(exc),
                    }
                )

        return results

    async def sync_and_compact(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        session_ids: list[str] | None = None,
        max_sessions: int | None = None,
        force_compact: bool = False,
    ) -> dict[str, Any]:
        sync_results = await self.sync(
            since=since,
            sources=sources,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )

        if int(sync_results["learnings_extracted"]) > 0 or force_compact:
            if self.llm is None:
                msg = "LLM provider is required for compaction"
                raise RuntimeError(msg)
            compact_engine = CompactionEngine(self.storage, self.files, self.llm)
            sync_results["compaction"] = await compact_engine.compact(force=force_compact)

        return sync_results

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

        active_ingesters, candidates, discovery_errors = self._discover_candidates(
            since=since,
            sources=sources,
        )
        results["errors"].extend(discovery_errors)

        for ingester in active_ingesters:
            results["by_source"][ingester.source_name] = {
                "discovered": 0,
                "listed": 0,
            }

        selected_candidates, missing_ids = self._apply_candidate_filters(
            candidates=candidates,
            session_ids=session_ids,
            max_sessions=max_sessions,
        )
        if missing_ids:
            results["errors"].append(
                f"Requested session IDs not found: {', '.join(missing_ids)}"
            )

        results["sessions_discovered"] = len(selected_candidates)
        for candidate in selected_candidates:
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
            requested_ids = {
                session_id.strip() for session_id in session_ids if session_id.strip()
            }
            selected = [
                candidate for candidate in selected if candidate.session_id in requested_ids
            ]
            found_ids = {candidate.session_id for candidate in selected}
            missing_ids = sorted(requested_ids - found_ids)

        selected.sort(
            key=lambda candidate: (candidate.sort_timestamp, candidate.session_id),
            reverse=True,
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
        return source.strip().lower().replace("_", "-")
