from __future__ import annotations

import io
import os
import platform
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_recall.core.semantic_embedder import (
    MODEL_NAME,
    _prepare_model_loading_environment,
    embed_single,
    get_embedding_dimension,
    prime_loaded_model,
)

_PROBE_TEXT = "agent recall local semantic memory probe"


def default_agent_recall_home() -> Path:
    override = os.environ.get("AGENT_RECALL_HOME")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/agent-recall"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData/Roaming"
        return base / "agent-recall"

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "agent-recall"
    return Path.home() / ".config" / "agent-recall"


@dataclass(frozen=True)
class LocalModelResolution:
    backend: str
    source: str
    model_name: str
    model_path: str | None
    cache_dir: str
    auto_download: bool


class LocalEmbeddingModelManager:
    def __init__(self, memory_cfg: object, *, home_dir: Path | None = None) -> None:
        self.memory_cfg: dict[str, Any] = memory_cfg if isinstance(memory_cfg, dict) else {}
        self.home_dir = home_dir or default_agent_recall_home()

    def resolve(self) -> LocalModelResolution:
        model_path = (
            str(self.memory_cfg.get("local_model_path")).strip()
            if self.memory_cfg.get("local_model_path")
            else None
        )
        model_name = (
            str(self.memory_cfg.get("local_model_name") or MODEL_NAME).strip() or MODEL_NAME
        )
        cache_dir = (
            str(self.memory_cfg.get("local_model_cache_dir")).strip()
            if self.memory_cfg.get("local_model_cache_dir")
            else ""
        )
        resolved_cache_dir = cache_dir or str(
            (self.home_dir / "models" / "sentence-transformers").expanduser()
        )
        return LocalModelResolution(
            backend="local",
            source="path" if model_path else "managed",
            model_name=model_name,
            model_path=model_path,
            cache_dir=resolved_cache_dir,
            auto_download=bool(self.memory_cfg.get("local_model_auto_download", True)),
        )

    def inspect(self) -> dict[str, Any]:
        resolution = self.resolve()
        if resolution.model_path:
            path = Path(resolution.model_path).expanduser()
            return self._status_payload(
                resolution,
                available=path.exists(),
                downloaded=False,
                state="ready" if path.exists() else "missing",
                model_path=str(path),
            )

        try:
            self._load_model(resolution, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            return self._status_payload(
                resolution,
                available=False,
                downloaded=False,
                state="missing",
                error=str(exc),
            )

        return self._status_payload(
            resolution,
            available=True,
            downloaded=False,
            state="ready",
        )

    def ensure_available(self, *, verify: bool = True) -> dict[str, Any]:
        resolution = self.resolve()
        downloaded = False
        missing_error: Exception | None = None
        loaded_model: object | None = None

        if resolution.model_path:
            path = Path(resolution.model_path).expanduser()
            if not path.exists():
                raise RuntimeError(f"Configured local model path does not exist: {path}")
            if verify:
                loaded_model = self._load_model(resolution, local_files_only=True)
        else:
            try:
                loaded_model = self._load_model(resolution, local_files_only=True)
            except Exception as exc:  # noqa: BLE001
                missing_error = exc
                if not resolution.auto_download:
                    raise RuntimeError(
                        "Local embedding model is not cached and auto-download is disabled."
                    ) from exc
                loaded_model = self._load_model(resolution, local_files_only=False)
                downloaded = True

        if loaded_model is not None:
            prime_loaded_model(
                loaded_model,
                model_name=resolution.model_name,
                model_path=resolution.model_path,
                cache_dir=resolution.cache_dir,
                # Once setup has verified the model locally, keep later loads on the
                # local cache/path so provisioning does not re-enter download paths.
                local_files_only=True,
            )

        if verify:
            vector = embed_single(_PROBE_TEXT).tolist()
            if len(vector) != get_embedding_dimension():
                raise RuntimeError(
                    "Local embedding model returned an unexpected embedding dimension."
                )

        payload = self._status_payload(
            resolution,
            available=True,
            downloaded=downloaded,
            state="ready",
        )
        if missing_error is not None:
            payload["recovered_from"] = str(missing_error)
        return payload

    @staticmethod
    def _status_payload(
        resolution: LocalModelResolution,
        *,
        available: bool,
        downloaded: bool,
        state: str,
        model_path: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "backend": resolution.backend,
            "source": resolution.source,
            "model_name": resolution.model_name,
            "model_path": model_path,
            "cache_dir": resolution.cache_dir,
            "auto_download": resolution.auto_download,
            "available": available,
            "downloaded": downloaded,
            "state": state,
        }
        if error:
            payload["error"] = error
        return payload

    @staticmethod
    def _load_model(
        resolution: LocalModelResolution,
        *,
        local_files_only: bool,
    ) -> object:
        _prepare_model_loading_environment()
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for local semantic memory."
            ) from exc

        model_source = resolution.model_path or resolution.model_name
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return SentenceTransformer(
                model_source,
                cache_folder=resolution.cache_dir,
                local_files_only=(local_files_only or bool(resolution.model_path)),
            )
