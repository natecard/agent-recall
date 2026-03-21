from __future__ import annotations

import hashlib
import re
import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from agent_recall.storage.base import Storage
from agent_recall.storage.models import CurationStatus, SemanticLabel

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "about",
    "when",
    "were",
    "been",
    "have",
    "will",
    "should",
}


def _topic_tokens(text: str) -> list[str]:
    tokens = [match.group(0) for match in _TOKEN_RE.finditer(text.lower())]
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 2]


def _topic_key_for_chunk(chunk) -> str:
    for tag in chunk.tags:
        normalized = str(tag).strip().lower()
        if not normalized:
            continue
        segments = [segment for segment in re.split(r"[/.:_\-]+", normalized) if segment]
        for segment in segments:
            if segment not in _STOPWORDS and len(segment) > 2:
                return segment
    tokens = _topic_tokens(chunk.content)
    return tokens[0] if tokens else "misc"


def _entry_source_session_map(storage: Storage, limit_per_label: int = 2000) -> dict[UUID, str]:
    mapping: dict[UUID, str] = {}
    for label in SemanticLabel:
        entries = storage.get_entries_by_label(
            [label],
            limit=limit_per_label,
            curation_status=CurationStatus.APPROVED,
        )
        for entry in entries:
            if entry.source_session_id:
                mapping[entry.id] = entry.source_session_id
    return mapping


def build_topic_threads(
    storage: Storage,
    *,
    min_cluster_size: int = 2,
    max_threads: int = 25,
    max_links_per_thread: int = 80,
) -> list[dict[str, Any]]:
    chunks = storage.list_chunks()
    if not chunks:
        return []

    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[_topic_key_for_chunk(chunk)].append(chunk)

    source_session_by_entry = _entry_source_session_map(storage)
    now = datetime.now(UTC)
    min_size = max(1, int(min_cluster_size))
    link_limit = max(1, int(max_links_per_thread))
    generated: list[dict[str, Any]] = []
    for key, chunk_group in grouped.items():
        if len(chunk_group) < min_size:
            continue

        sorted_chunks = sorted(chunk_group, key=lambda chunk: chunk.created_at, reverse=True)
        newest = sorted_chunks[0].created_at
        age_hours = max(1.0, (now - newest).total_seconds() / 3600.0)
        score = float(len(sorted_chunks)) / (age_hours**0.15)
        title = key.replace("_", " ").strip().title()
        summary_lines = [
            f"- {textwrap.shorten(chunk.content.strip(), width=90, placeholder='...')}"
            for chunk in sorted_chunks[:3]
        ]
        summary = "\n".join(summary_lines)
        links: list[dict[str, str | None]] = []
        source_sessions: set[str] = set()
        for chunk in sorted_chunks:
            if len(links) >= link_limit:
                break
            if chunk.source_ids:
                for entry_id in chunk.source_ids:
                    source_session_id = source_session_by_entry.get(entry_id)
                    if source_session_id:
                        source_sessions.add(source_session_id)
                    links.append(
                        {
                            "entry_id": str(entry_id),
                            "chunk_id": str(chunk.id),
                            "source_session_id": source_session_id,
                            "created_at": chunk.created_at.isoformat(),
                        }
                    )
                    if len(links) >= link_limit:
                        break
            else:
                links.append(
                    {
                        "entry_id": None,
                        "chunk_id": str(chunk.id),
                        "source_session_id": None,
                        "created_at": chunk.created_at.isoformat(),
                    }
                )

        thread_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        generated.append(
            {
                "thread_id": thread_id,
                "title": title or "Misc",
                "summary": summary or "- No summary available",
                "score": score,
                "entry_count": len(links),
                "source_session_count": len(source_sessions),
                "last_seen_at": newest.isoformat(),
                "links": links,
            }
        )

    generated.sort(key=lambda item: (-float(item["score"]), str(item["thread_id"])))
    return generated[: max(1, int(max_threads))]
