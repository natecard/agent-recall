from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_recall.core.compact import CompactionEngine
from agent_recall.core.extract import TranscriptExtractor
from agent_recall.ingest import SessionIngester, get_default_ingesters
from agent_recall.llm.base import LLMProvider
from agent_recall.storage.files import FileStorage
from agent_recall.storage.sqlite import SQLiteStorage


class AutoSync:
    """Discover, extract, and persist learnings from native agent sessions."""

    def __init__(
        self,
        storage: SQLiteStorage,
        files: FileStorage,
        llm: LLMProvider,
        ingesters: list[SessionIngester] | None = None,
        project_path: Path | None = None,
    ):
        self.storage = storage
        self.files = files
        self.llm = llm
        self.extractor = TranscriptExtractor(llm)
        self.ingesters = ingesters or get_default_ingesters(project_path)
        self.extract_timeout_seconds = 45

    async def sync(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            "sessions_discovered": 0,
            "sessions_processed": 0,
            "sessions_skipped": 0,
            "empty_sessions": 0,
            "learnings_extracted": 0,
            "by_source": {},
            "errors": [],
        }

        active_ingesters = self._select_ingesters(sources)

        for ingester in active_ingesters:
            source_results = {
                "discovered": 0,
                "processed": 0,
                "skipped": 0,
                "empty": 0,
                "learnings": 0,
            }

            try:
                session_paths = ingester.discover_sessions(since=since)
                source_results["discovered"] = len(session_paths)
                results["sessions_discovered"] += len(session_paths)

                for session_path in session_paths:
                    session_id = ingester.get_session_id(session_path)

                    if self.storage.is_session_processed(session_id):
                        source_results["skipped"] += 1
                        results["sessions_skipped"] += 1
                        continue

                    try:
                        raw_session = ingester.parse_session(session_path)
                        if len(raw_session.messages) < 2:
                            self.storage.mark_session_processed(session_id)
                            source_results["skipped"] += 1
                            source_results["empty"] += 1
                            results["sessions_skipped"] += 1
                            results["empty_sessions"] += 1
                            continue

                        try:
                            entries = await asyncio.wait_for(
                                self.extractor.extract(raw_session),
                                timeout=self.extract_timeout_seconds,
                            )
                        except TimeoutError:
                            results["errors"].append(
                                f"{ingester.source_name}:{session_path.name}: "
                                f"extraction timed out after {self.extract_timeout_seconds}s"
                            )
                            entries = []
                        except Exception as exc:  # noqa: BLE001
                            results["errors"].append(
                                f"{ingester.source_name}:{session_path.name}: "
                                f"extraction failed: {exc}"
                            )
                            entries = []
                        for entry in entries:
                            self.storage.append_entry(entry)

                        self.storage.mark_session_processed(session_id)

                        source_results["processed"] += 1
                        source_results["learnings"] += len(entries)
                        results["sessions_processed"] += 1
                        results["learnings_extracted"] += len(entries)
                    except Exception as exc:  # noqa: BLE001
                        results["errors"].append(
                            f"{ingester.source_name}:{session_path.name}: {exc}"
                        )
            except Exception as exc:  # noqa: BLE001
                results["errors"].append(f"{ingester.source_name}: {exc}")

            results["by_source"][ingester.source_name] = source_results

        return results

    async def sync_and_compact(
        self,
        since: datetime | None = None,
        sources: list[str] | None = None,
        force_compact: bool = False,
    ) -> dict[str, Any]:
        sync_results = await self.sync(since=since, sources=sources)

        if int(sync_results["learnings_extracted"]) > 0 or force_compact:
            compact_engine = CompactionEngine(self.storage, self.files, self.llm)
            sync_results["compaction"] = await compact_engine.compact(force=force_compact)

        return sync_results

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
