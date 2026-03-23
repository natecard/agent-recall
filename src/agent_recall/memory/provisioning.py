from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from agent_recall.core.config import load_config
from agent_recall.core.embedding_indexer import EmbeddingIndexer
from agent_recall.core.semantic_embedder import configure_from_memory_config
from agent_recall.memory.local_model import LocalEmbeddingModelManager, default_agent_recall_home
from agent_recall.memory.migration import VectorMigrationRequest, VectorMigrationService
from agent_recall.memory.policy import MemoryPolicy
from agent_recall.memory.vector_store import LocalVectorStore, resolve_local_vector_db_path
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path, 0o700)
        except OSError:
            return


def _normalize_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class VectorMemorySetupRequest:
    enabled: bool = True
    backend: Literal["local", "turbopuffer"] = "local"
    local_model_name: str | None = None
    local_model_path: str | None = None
    local_model_cache_dir: str | None = None
    local_model_auto_download: bool = True
    auto_sync_local_vectors: bool = True
    turbopuffer_base_url: str | None = None
    turbopuffer_api_key_env: str = "TURBOPUFFER_API_KEY"
    external_embedding_base_url: str | None = None
    external_embedding_api_key_env: str = "OPENAI_API_KEY"
    external_embedding_model: str = "text-embedding-3-small"
    external_embedding_timeout_seconds: float = 10.0


class VectorMemoryStatusStore:
    def __init__(self, home_dir: Path | None = None) -> None:
        self.home_dir = home_dir or default_agent_recall_home()
        self.base_dir = self.home_dir / "status" / "vector-memory"

    def load(self, repo_path: Path) -> dict[str, Any]:
        path = self._repo_path(repo_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, repo_path: Path, payload: dict[str, Any]) -> None:
        _ensure_private_dir(self.base_dir)
        path = self._repo_path(repo_path)
        merged = self.load(repo_path)
        merged.update(payload)
        merged["repository_path"] = str(repo_path)
        merged["updated_at"] = _now_iso()
        path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        if os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                return

    def _repo_path(self, repo_path: Path) -> Path:
        digest = hashlib.sha256(str(repo_path.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.base_dir / f"{digest}.json"


class VectorMemoryService:
    def __init__(
        self,
        *,
        storage: Storage,
        files: FileStorage,
        home_dir: Path | None = None,
    ) -> None:
        self.storage = storage
        self.files = files
        self.home_dir = home_dir or default_agent_recall_home()
        self.status_store = VectorMemoryStatusStore(home_dir=self.home_dir)

    def provision(
        self,
        request: VectorMemorySetupRequest,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        config_dict = self.files.read_config()
        self.apply_setup_config(config_dict, request)
        self.files.write_config(config_dict)

        payload: dict[str, Any] = {
            "enabled": request.enabled,
            "backend": request.backend,
            "configured_at": _now_iso(),
        }
        if not request.enabled:
            payload["status"] = "disabled"
            self.status_store.save(
                self.repo_path,
                {
                    "degraded": False,
                    "backend": request.backend,
                    "last_setup": payload,
                },
            )
            return payload

        try:
            parsed_config = load_config(self.files.agent_dir)
            memory_cfg = parsed_config.memory.model_dump(mode="python")
            model_status = LocalEmbeddingModelManager(
                memory_cfg,
                home_dir=self.home_dir,
            ).ensure_available()
            configure_from_memory_config(memory_cfg)
            cloud_status: dict[str, Any] | None = None
            if request.backend != "local":
                cloud_status = self.validate_cloud_config()

            embedding_indexing = EmbeddingIndexer(self.storage).index_missing_embeddings()
            vector_sync = self._sync_vectors_impl(
                VectorMigrationRequest(dry_run=False),
                trigger="setup",
                honor_feature_flag=False,
                persist_status=False,
                ensure_local_model=False,
            )
            payload.update(
                {
                    "status": "ready",
                    "model_status": model_status,
                    "embedding_indexing": embedding_indexing,
                    "vector_sync": vector_sync,
                }
            )
            if cloud_status is not None:
                payload["cloud_status"] = cloud_status
            self.status_store.save(
                self.repo_path,
                {
                    "degraded": False,
                    "backend": request.backend,
                    "last_setup": payload,
                    "model_status": model_status,
                    "cloud_status": cloud_status,
                    "last_sync": vector_sync,
                },
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            payload.update({"status": "degraded", "error": str(exc)})
            self.status_store.save(
                self.repo_path,
                {
                    "degraded": True,
                    "backend": request.backend,
                    "last_setup": payload,
                    "error": str(exc),
                },
            )
            if strict:
                raise
            return payload

    def sync_vectors(
        self,
        request: VectorMigrationRequest,
        *,
        trigger: str,
        honor_feature_flag: bool = True,
        persist_status: bool = True,
    ) -> dict[str, Any]:
        return self._sync_vectors_impl(
            request,
            trigger=trigger,
            honor_feature_flag=honor_feature_flag,
            persist_status=persist_status,
            ensure_local_model=True,
        )

    def _sync_vectors_impl(
        self,
        request: VectorMigrationRequest,
        *,
        trigger: str,
        honor_feature_flag: bool,
        persist_status: bool,
        ensure_local_model: bool,
    ) -> dict[str, Any]:
        config_dict = self.files.read_config()
        memory_cfg = _normalize_mapping(config_dict.get("memory"))
        vector_enabled = bool(memory_cfg.get("vector_enabled", False))
        if honor_feature_flag and not vector_enabled:
            payload = {
                "status": "skipped",
                "reason": "vector_memory_disabled",
                "trigger": trigger,
                "at": _now_iso(),
            }
            if persist_status:
                self.status_store.save(
                    self.repo_path,
                    {
                        "degraded": False,
                        "last_sync": payload,
                    },
                )
            return payload

        app_config = load_config(self.files.agent_dir)
        resolved_memory = app_config.memory.model_dump(mode="python")
        if (
            ensure_local_model
            and str(resolved_memory.get("embedding_provider", "local")).strip().lower() == "local"
            and int(app_config.retrieval.embedding_dimensions) == 384
        ):
            LocalEmbeddingModelManager(resolved_memory, home_dir=self.home_dir).ensure_available(
                verify=True
            )
        if (
            str(resolved_memory.get("embedding_provider", "local")).strip().lower() == "local"
            and int(app_config.retrieval.embedding_dimensions) == 384
        ):
            configure_from_memory_config(resolved_memory)

        service = VectorMigrationService(
            storage=self.storage,
            files=self.files,
            memory_cfg=resolved_memory,
            policy=MemoryPolicy.from_memory_config(resolved_memory),
            tenant_id=app_config.storage.shared.tenant_id,
            project_id=app_config.storage.shared.project_id,
            embedding_dimensions=app_config.retrieval.embedding_dimensions,
            vector_db_path=resolve_local_vector_db_path(self.files.agent_dir),
        )
        try:
            payload = service.migrate(request)
        except Exception as exc:  # noqa: BLE001
            error_payload = {
                "status": "degraded",
                "trigger": trigger,
                "error": str(exc),
                "at": _now_iso(),
            }
            if persist_status:
                self.status_store.save(
                    self.repo_path,
                    {
                        "degraded": True,
                        "error": str(exc),
                        "last_sync": error_payload,
                    },
                )
            raise

        enriched = {
            **payload,
            "status": "ready",
            "trigger": trigger,
            "at": _now_iso(),
        }
        if persist_status:
            self.status_store.save(
                self.repo_path,
                {
                    "degraded": False,
                    "last_sync": enriched,
                },
            )
        return enriched

    def status(self) -> dict[str, Any]:
        parsed_config = load_config(self.files.agent_dir)
        memory_cfg = parsed_config.memory.model_dump(mode="python")
        stored = self.status_store.load(self.repo_path)
        vector_enabled = bool(memory_cfg.get("vector_enabled", False))
        backend = str(memory_cfg.get("vector_backend", "local")).strip().lower()
        payload: dict[str, Any] = {
            "enabled": vector_enabled,
            "backend": backend,
            "embedding_provider": str(memory_cfg.get("embedding_provider", "local")),
            "mode": str(memory_cfg.get("mode", "markdown")),
            "semantic_index_enabled": bool(parsed_config.retrieval.semantic_index_enabled),
            "embedding_dimensions": int(parsed_config.retrieval.embedding_dimensions),
            "chunk_embeddings": EmbeddingIndexer(self.storage).get_indexing_stats(),
            "degraded": bool(stored.get("degraded", False)),
            "last_setup": stored.get("last_setup"),
            "last_sync": stored.get("last_sync"),
        }
        payload["model_status"] = LocalEmbeddingModelManager(
            memory_cfg, home_dir=self.home_dir
        ).inspect()
        if backend == "local":
            vector_db_path = resolve_local_vector_db_path(self.files.agent_dir)
            payload["vector_store"] = {
                "backend": "local",
                "path": str(vector_db_path),
                "record_count": LocalVectorStore(
                    vector_db_path,
                    tenant_id=parsed_config.storage.shared.tenant_id,
                    project_id=parsed_config.storage.shared.project_id,
                ).count_records(),
            }
        else:
            payload["cloud_status"] = self.validate_cloud_config(raise_on_error=False)
            payload["vector_store"] = {
                "backend": "turbopuffer",
                "namespace": (
                    f"{parsed_config.storage.shared.tenant_id}:"
                    f"{parsed_config.storage.shared.project_id}"
                ),
            }
        return payload

    def validate_cloud_config(self, *, raise_on_error: bool = True) -> dict[str, Any]:
        parsed_config = load_config(self.files.agent_dir)
        memory_cfg = parsed_config.memory.model_dump(mode="python")
        turbopuffer_cfg = _normalize_mapping(memory_cfg.get("turbopuffer"))
        missing: list[str] = []

        base_url = str(turbopuffer_cfg.get("base_url") or "").strip()
        turbo_env = str(turbopuffer_cfg.get("api_key_env") or "TURBOPUFFER_API_KEY").strip()
        embedding_base_url = str(memory_cfg.get("external_embedding_base_url") or "").strip()
        embedding_env = str(
            memory_cfg.get("external_embedding_api_key_env") or "OPENAI_API_KEY"
        ).strip()
        embedding_model = str(
            memory_cfg.get("external_embedding_model") or "text-embedding-3-small"
        ).strip()

        if not base_url:
            missing.append("memory.turbopuffer.base_url")
        if not turbo_env:
            missing.append("memory.turbopuffer.api_key_env")
        elif not os.environ.get(turbo_env, "").strip():
            missing.append(f"env:{turbo_env}")
        if not embedding_base_url:
            missing.append("memory.external_embedding_base_url")
        if not embedding_env:
            missing.append("memory.external_embedding_api_key_env")
        elif not os.environ.get(embedding_env, "").strip():
            missing.append(f"env:{embedding_env}")
        if not embedding_model:
            missing.append("memory.external_embedding_model")

        payload = {
            "backend": "turbopuffer",
            "base_url": base_url or None,
            "api_key_env": turbo_env,
            "external_embedding_base_url": embedding_base_url or None,
            "external_embedding_api_key_env": embedding_env,
            "external_embedding_model": embedding_model or None,
            "state": "ready" if not missing else "missing_configuration",
            "missing": missing,
        }
        if missing and raise_on_error:
            raise RuntimeError(
                "Cloud vector memory is missing required configuration: " + ", ".join(missing)
            )
        return payload

    def apply_setup_config(
        self,
        config_dict: dict[str, Any],
        request: VectorMemorySetupRequest,
    ) -> dict[str, Any]:
        memory_cfg = _normalize_mapping(config_dict.get("memory"))
        retrieval_cfg = _normalize_mapping(config_dict.get("retrieval"))

        memory_cfg["vector_enabled"] = bool(request.enabled)
        if request.enabled:
            memory_cfg["mode"] = "hybrid"
            memory_cfg["vector_backend"] = request.backend
            memory_cfg["auto_sync_local_vectors"] = bool(request.auto_sync_local_vectors)
            memory_cfg["local_model_name"] = (request.local_model_name or "").strip() or str(
                memory_cfg.get("local_model_name") or "all-MiniLM-L6-v2"
            )
            memory_cfg["local_model_path"] = (
                (request.local_model_path or "").strip()
                or memory_cfg.get("local_model_path")
                or None
            )
            memory_cfg["local_model_cache_dir"] = (
                (request.local_model_cache_dir or "").strip()
                or memory_cfg.get("local_model_cache_dir")
                or None
            )
            memory_cfg["local_model_auto_download"] = bool(request.local_model_auto_download)
            retrieval_cfg["semantic_index_enabled"] = True
            retrieval_cfg["embedding_dimensions"] = 384

            if request.backend == "local":
                memory_cfg["embedding_provider"] = "local"
            else:
                turbopuffer_cfg = _normalize_mapping(memory_cfg.get("turbopuffer"))
                memory_cfg["embedding_provider"] = "external"
                memory_cfg["external_embedding_base_url"] = (
                    request.external_embedding_base_url or ""
                ).strip() or None
                memory_cfg["external_embedding_api_key_env"] = (
                    request.external_embedding_api_key_env or ""
                ).strip() or "OPENAI_API_KEY"
                memory_cfg["external_embedding_model"] = (
                    request.external_embedding_model or ""
                ).strip() or "text-embedding-3-small"
                memory_cfg["external_embedding_timeout_seconds"] = float(
                    request.external_embedding_timeout_seconds
                )
                turbopuffer_cfg["base_url"] = (request.turbopuffer_base_url or "").strip() or None
                turbopuffer_cfg["api_key_env"] = (
                    request.turbopuffer_api_key_env or ""
                ).strip() or "TURBOPUFFER_API_KEY"
                memory_cfg["turbopuffer"] = turbopuffer_cfg

        config_dict["memory"] = memory_cfg
        config_dict["retrieval"] = retrieval_cfg
        return config_dict

    @property
    def repo_path(self) -> Path:
        return self.files.agent_dir.parent.resolve()
