from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage, KnowledgeTier
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel

PACK_FORMAT = "agent-recall-memory-pack"
PACK_VERSION = "1.0"
MergeStrategy = Literal["skip", "append", "overwrite"]


class MemoryPackChunk(BaseModel):
    id: str
    source: str
    source_ids: list[str] = Field(default_factory=list)
    content: str
    label: str
    tags: list[str] = Field(default_factory=list)
    created_at: str
    token_count: int | None = None
    embedding: list[float] | None = None
    embedding_version: int = 0


class MemoryPack(BaseModel):
    format: str = PACK_FORMAT
    version: str = PACK_VERSION
    created_at: str
    tiers: dict[str, str]
    chunks: list[MemoryPackChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_memory_pack(storage: Storage, files: FileStorage) -> MemoryPack:
    tiers = {
        KnowledgeTier.GUARDRAILS.value: files.read_tier(KnowledgeTier.GUARDRAILS),
        KnowledgeTier.STYLE.value: files.read_tier(KnowledgeTier.STYLE),
        KnowledgeTier.RECENT.value: files.read_tier(KnowledgeTier.RECENT),
    }
    chunks = [
        MemoryPackChunk(
            id=str(chunk.id),
            source=chunk.source.value,
            source_ids=[str(value) for value in chunk.source_ids],
            content=chunk.content,
            label=chunk.label.value,
            tags=list(chunk.tags),
            created_at=chunk.created_at.isoformat(),
            token_count=chunk.token_count,
            embedding=list(chunk.embedding) if chunk.embedding else None,
            embedding_version=chunk.embedding_version,
        )
        for chunk in storage.list_chunks()
    ]
    metadata = {
        "stats": storage.get_stats(),
        "chunk_count": len(chunks),
        "tier_chars": {name: len(content) for name, content in tiers.items()},
    }
    return MemoryPack(
        created_at=datetime.now(UTC).isoformat(),
        tiers=tiers,
        chunks=chunks,
        metadata=metadata,
    )


def write_memory_pack(path: Path, pack: MemoryPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack.model_dump(), indent=2), encoding="utf-8")


def read_memory_pack(path: Path) -> MemoryPack:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MemoryPack.model_validate(payload)


def validate_memory_pack(pack: MemoryPack) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if pack.format != PACK_FORMAT:
        errors.append(f"Unsupported format '{pack.format}'. Expected '{PACK_FORMAT}'.")
    if not str(pack.version).startswith("1."):
        errors.append(f"Unsupported major version '{pack.version}'.")
    for required_tier in (KnowledgeTier.GUARDRAILS.value, KnowledgeTier.STYLE.value):
        if required_tier not in pack.tiers:
            warnings.append(f"Tier '{required_tier}' is missing from pack payload.")
    if len(pack.chunks) == 0:
        warnings.append("No chunks included in memory pack.")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "chunk_count": len(pack.chunks),
    }


def import_memory_pack(
    storage: Storage,
    files: FileStorage,
    pack: MemoryPack,
    *,
    strategy: MergeStrategy = "append",
) -> dict[str, Any]:
    validation = validate_memory_pack(pack)
    if not validation["valid"]:
        raise ValueError("Invalid memory pack: " + "; ".join(validation["errors"]))

    tier_updates = 0
    written_chunks = 0
    skipped_chunks = 0
    tier_map = {
        KnowledgeTier.GUARDRAILS.value: KnowledgeTier.GUARDRAILS,
        KnowledgeTier.STYLE.value: KnowledgeTier.STYLE,
        KnowledgeTier.RECENT.value: KnowledgeTier.RECENT,
    }

    for tier_name, incoming_content in pack.tiers.items():
        tier = tier_map.get(tier_name)
        if tier is None:
            continue
        current = files.read_tier(tier)
        incoming = incoming_content or ""
        if strategy == "overwrite":
            if current != incoming:
                files.write_tier(tier, incoming)
                tier_updates += 1
            continue
        if strategy == "skip":
            if not current.strip() and incoming.strip():
                files.write_tier(tier, incoming)
                tier_updates += 1
            continue
        if incoming.strip() and incoming.strip() not in current:
            separator = "\n\n---\n\n" if current.strip() else ""
            merged = f"{current.rstrip()}{separator}{incoming.strip()}\n"
            files.write_tier(tier, merged)
            tier_updates += 1

    for chunk_row in pack.chunks:
        try:
            label = SemanticLabel(str(chunk_row.label).strip().lower())
        except ValueError:
            skipped_chunks += 1
            continue
        if storage.has_chunk(chunk_row.content, label):
            skipped_chunks += 1
            continue
        try:
            chunk_id = UUID(str(chunk_row.id))
        except ValueError:
            chunk_id = uuid4()
        source_ids: list[UUID] = []
        for source_id in chunk_row.source_ids:
            try:
                source_ids.append(UUID(str(source_id)))
            except ValueError:
                continue
        try:
            created_at = datetime.fromisoformat(str(chunk_row.created_at))
        except ValueError:
            created_at = datetime.now(UTC)
        try:
            source = ChunkSource(str(chunk_row.source))
        except ValueError:
            source = ChunkSource.IMPORT
        chunk = Chunk(
            id=chunk_id,
            source=source,
            source_ids=source_ids,
            content=chunk_row.content,
            label=label,
            tags=list(chunk_row.tags),
            created_at=created_at,
            token_count=chunk_row.token_count,
            embedding=chunk_row.embedding,
            embedding_version=int(chunk_row.embedding_version),
        )
        try:
            storage.store_chunk(chunk)
        except Exception:  # noqa: BLE001
            chunk = chunk.model_copy(update={"id": uuid4()})
            storage.store_chunk(chunk)
        written_chunks += 1

    return {
        "tier_updates": tier_updates,
        "chunks_written": written_chunks,
        "chunks_skipped": skipped_chunks,
        "warnings": validation["warnings"],
    }
