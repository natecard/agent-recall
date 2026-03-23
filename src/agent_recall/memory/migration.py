from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_recall.memory.embedding_provider import (
    EmbeddingProvider,
    ExternalEmbeddingProvider,
    LocalEmbeddingProvider,
)
from agent_recall.memory.policy import MemoryPolicy, normalize_memory_rows
from agent_recall.memory.vector_store import LocalVectorStore, TurboPufferVectorStore, VectorRecord
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage
from agent_recall.storage.models import CurationStatus, LogEntry


def learning_row_from_entry(entry: LogEntry) -> dict[str, Any]:
    metadata = dict(entry.metadata)
    source_session_id = entry.source_session_id or str(metadata.get("source_session_id") or "")
    row_tags = list(entry.tags)
    row_tags.extend(
        [
            entry.label.value,
            entry.source.value,
        ]
    )
    deduped_tags = [tag for tag in dict.fromkeys(tag for tag in row_tags if str(tag).strip())]
    return {
        "id": str(entry.id),
        "text": entry.content,
        "label": entry.label.value,
        "tags": deduped_tags,
        "metadata": {
            "entry_id": str(entry.id),
            "source": entry.source.value,
            "source_session_id": source_session_id or None,
            "session_id": str(entry.session_id) if entry.session_id else None,
            "timestamp": entry.timestamp.isoformat(),
            "confidence": float(entry.confidence),
            "curation_status": entry.curation_status.value,
            "entry_metadata": metadata,
        },
    }


def collect_learning_rows(storage: Storage) -> list[dict[str, Any]]:
    limit = max(100, int(storage.count_log_entries() or 0))
    entries = storage.list_entries_by_curation_status(CurationStatus.APPROVED, limit=limit)
    return [learning_row_from_entry(entry) for entry in entries if entry.content.strip()]


def build_embedding_provider_from_memory_config(
    memory_cfg: dict[str, Any],
    *,
    embedding_dimensions: int,
) -> tuple[str, EmbeddingProvider]:
    embedding_provider_name = str(memory_cfg.get("embedding_provider", "local")).strip().lower()
    cost_cfg = memory_cfg.get("cost", {})
    if not isinstance(cost_cfg, dict):
        cost_cfg = {}
    if embedding_provider_name == "external":
        base_url = str(memory_cfg.get("external_embedding_base_url") or "").strip()
        if not base_url:
            raise ValueError(
                "memory.external_embedding_base_url is required for external provider."
            )
        provider: EmbeddingProvider = ExternalEmbeddingProvider(
            base_url=base_url,
            api_key_env=str(memory_cfg.get("external_embedding_api_key_env", "OPENAI_API_KEY")),
            model=str(
                memory_cfg.get(
                    "external_embedding_model",
                    "text-embedding-3-small",
                )
            ),
            timeout_seconds=float(memory_cfg.get("external_embedding_timeout_seconds", 10.0)),
            max_cost_usd=float(cost_cfg.get("max_external_embedding_usd", 1.0)),
        )
        return embedding_provider_name, provider

    provider = LocalEmbeddingProvider(
        model_name=str(memory_cfg.get("local_model_name") or "all-MiniLM-L6-v2"),
        model_path=(
            str(memory_cfg.get("local_model_path")) if memory_cfg.get("local_model_path") else None
        ),
        cache_dir=(
            str(memory_cfg.get("local_model_cache_dir"))
            if memory_cfg.get("local_model_cache_dir")
            else None
        ),
        local_files_only=bool(memory_cfg.get("vector_enabled", False)),
        dimensions=embedding_dimensions,
        strict_local_model=bool(memory_cfg.get("vector_enabled", False)),
    )
    return "local", provider


def collect_memory_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tiers = snapshot.get("tiers")
    if isinstance(tiers, dict):
        for tier_name, content in tiers.items():
            if not isinstance(content, str):
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                text = stripped[2:].strip()
                if not text:
                    continue
                row_id = str(abs(hash(f"{tier_name}:{text}")))
                rows.append(
                    {
                        "id": f"tier-{row_id}",
                        "text": text,
                        "label": str(tier_name).lower(),
                        "tags": [str(tier_name).lower()],
                    }
                )

    chunks = snapshot.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("content", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "id": str(chunk.get("id", "")) or f"chunk-{abs(hash(text))}",
                    "text": text,
                    "label": str(chunk.get("label", "unknown")),
                    "tags": [str(tag) for tag in chunk.get("tags", []) if str(tag).strip()],
                }
            )
    return rows


@dataclass(frozen=True)
class VectorMigrationRequest:
    dry_run: bool = False
    max_records: int | None = None
    batch_size: int | None = None


class VectorMigrationService:
    def __init__(
        self,
        *,
        storage: Storage,
        files: FileStorage,
        memory_cfg: dict[str, Any],
        policy: MemoryPolicy,
        tenant_id: str,
        project_id: str,
        embedding_dimensions: int,
        vector_db_path: Path,
    ) -> None:
        self.storage = storage
        self.files = files
        self.memory_cfg = memory_cfg if isinstance(memory_cfg, dict) else {}
        self.policy = policy
        self.tenant_id = tenant_id
        self.project_id = project_id
        self.embedding_dimensions = int(embedding_dimensions)
        self.vector_db_path = vector_db_path

    def migrate(self, request: VectorMigrationRequest) -> dict[str, Any]:
        rows = self.collect_rows()
        normalized = normalize_memory_rows(
            rows,
            policy=self.policy,
            limit_override=request.max_records,
        )
        embedding_provider_name, provider = self.build_embedding_provider()
        vectors, estimated_tokens, estimated_cost_usd = self.embed_rows(
            normalized.rows,
            provider=provider,
            batch_size_override=request.batch_size,
        )
        vector_backend = self.resolve_backend()
        rows_written = self.persist_vectors(
            vectors,
            backend=vector_backend,
            dry_run=request.dry_run,
        )
        return {
            "rows_discovered": normalized.rows_discovered,
            "rows_normalized": normalized.rows_normalized,
            "rows_migrated": len(vectors),
            "rows_written": rows_written,
            "redacted_rows": normalized.redacted_rows,
            "rows_capped": normalized.rows_capped,
            "rows_deduplicated": normalized.rows_deduplicated,
            "embedding_provider": embedding_provider_name,
            "vector_backend": vector_backend,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "dry_run": request.dry_run,
        }

    def collect_rows(self) -> list[dict[str, Any]]:
        return collect_learning_rows(self.storage)

    def resolve_backend(self) -> str:
        return str(self.memory_cfg.get("vector_backend", "local")).strip().lower()

    def build_embedding_provider(self) -> tuple[str, EmbeddingProvider]:
        return build_embedding_provider_from_memory_config(
            self.memory_cfg,
            embedding_dimensions=self.embedding_dimensions,
        )

    def resolve_migration_batch_size(self, override: int | None = None) -> int:
        if override is not None:
            return max(1, int(override))
        return max(1, int(self.memory_cfg.get("migration_batch_size", 100)))

    def embed_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        provider: EmbeddingProvider,
        batch_size_override: int | None = None,
    ) -> tuple[list[VectorRecord], int, float]:
        batch_size = self.resolve_migration_batch_size(batch_size_override)
        vectors: list[VectorRecord] = []
        total_tokens = 0
        total_cost = 0.0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [str(item.get("text", "")) for item in batch]
            response = provider.embed_texts(texts)
            total_tokens += response.estimated_tokens
            total_cost += response.estimated_cost_usd
            for row, embedding in zip(batch, response.vectors, strict=False):
                metadata = row.get("metadata")
                vectors.append(
                    VectorRecord(
                        id=str(row.get("id", "")),
                        tenant_id=self.tenant_id,
                        project_id=self.project_id,
                        text=str(row.get("text", "")),
                        label=str(row.get("label", "unknown")),
                        tags=[str(tag) for tag in row.get("tags", []) if str(tag).strip()],
                        embedding=embedding,
                        metadata=metadata
                        if isinstance(metadata, dict)
                        else {"source": "migration"},
                        updated_at=datetime.now(UTC).isoformat(),
                    )
                )
        return vectors, total_tokens, total_cost

    def persist_vectors(
        self,
        vectors: list[VectorRecord],
        *,
        backend: str,
        dry_run: bool,
    ) -> int:
        if dry_run:
            return 0
        if not vectors:
            return 0
        if backend == "turbopuffer":
            turbo_cfg = self.memory_cfg.get("turbopuffer", {})
            if not isinstance(turbo_cfg, dict):
                turbo_cfg = {}
            base_url = str(turbo_cfg.get("base_url") or "").strip()
            if not base_url:
                raise ValueError("memory.turbopuffer.base_url is required.")
            vector_store = TurboPufferVectorStore(
                base_url=base_url,
                api_key_env=str(turbo_cfg.get("api_key_env", "TURBOPUFFER_API_KEY")),
                tenant_id=self.tenant_id,
                project_id=self.project_id,
                timeout_seconds=float(turbo_cfg.get("timeout_seconds", 10.0)),
                retry_attempts=int(turbo_cfg.get("retry_attempts", 2)),
            )
            return vector_store.upsert_records(vectors)
        vector_store = LocalVectorStore(
            self.vector_db_path,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        return vector_store.upsert_records(vectors)
