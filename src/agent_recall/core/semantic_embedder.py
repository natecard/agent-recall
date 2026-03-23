from __future__ import annotations

import io
import os
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_embedding_lock = threading.Lock()
_cached_model: SentenceTransformer | None = None
_configured_model_name = MODEL_NAME
_configured_model_path: str | None = None
_configured_cache_dir: str | None = None
_configured_local_files_only = False


@dataclass(frozen=True)
class SemanticEmbedderConfig:
    model_name: str = MODEL_NAME
    model_path: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False


def get_model_config() -> SemanticEmbedderConfig:
    return SemanticEmbedderConfig(
        model_name=_configured_model_name,
        model_path=_configured_model_path,
        cache_dir=_configured_cache_dir,
        local_files_only=_configured_local_files_only,
    )


def configure_model(
    *,
    model_name: str | None = None,
    model_path: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> SemanticEmbedderConfig:
    global _configured_model_name, _configured_model_path, _configured_cache_dir
    global _configured_local_files_only

    resolved_name = (model_name or MODEL_NAME).strip() or MODEL_NAME
    resolved_path = (model_path or "").strip() or None
    resolved_cache_dir = (cache_dir or "").strip() or None
    resolved_local_files_only = bool(local_files_only or resolved_path)
    next_config = SemanticEmbedderConfig(
        model_name=resolved_name,
        model_path=resolved_path,
        cache_dir=resolved_cache_dir,
        local_files_only=resolved_local_files_only,
    )
    if next_config == get_model_config():
        return next_config
    with _embedding_lock:
        _configured_model_name = resolved_name
        _configured_model_path = resolved_path
        _configured_cache_dir = resolved_cache_dir
        _configured_local_files_only = resolved_local_files_only
        global _cached_model
        _cached_model = None
    return next_config


def prime_loaded_model(
    model: Any,
    *,
    model_name: str | None = None,
    model_path: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> SemanticEmbedderConfig:
    global _configured_model_name, _configured_model_path, _configured_cache_dir
    global _configured_local_files_only, _cached_model

    resolved_name = (model_name or MODEL_NAME).strip() or MODEL_NAME
    resolved_path = (model_path or "").strip() or None
    resolved_cache_dir = (cache_dir or "").strip() or None
    resolved_local_files_only = bool(local_files_only or resolved_path)
    next_config = SemanticEmbedderConfig(
        model_name=resolved_name,
        model_path=resolved_path,
        cache_dir=resolved_cache_dir,
        local_files_only=resolved_local_files_only,
    )
    with _embedding_lock:
        _configured_model_name = resolved_name
        _configured_model_path = resolved_path
        _configured_cache_dir = resolved_cache_dir
        _configured_local_files_only = resolved_local_files_only
        _cached_model = model
    return next_config


def configure_from_memory_config(memory_cfg: object) -> SemanticEmbedderConfig:
    cfg: dict[str, Any] = cast(dict[str, Any], memory_cfg) if isinstance(memory_cfg, dict) else {}
    embedding_provider = str(cfg.get("embedding_provider", "local")).strip().lower()
    local_files_only = bool(cfg.get("local_model_path")) or (
        bool(cfg.get("vector_enabled", False)) and embedding_provider == "local"
    )
    return configure_model(
        model_name=str(cfg.get("local_model_name") or MODEL_NAME),
        model_path=(
            str(cfg.get("local_model_path")).strip() if cfg.get("local_model_path") else None
        ),
        cache_dir=(
            str(cfg.get("local_model_cache_dir")).strip()
            if cfg.get("local_model_cache_dir")
            else None
        ),
        local_files_only=local_files_only,
    )


def _prepare_model_loading_environment() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _load_model() -> SentenceTransformer:
    _prepare_model_loading_environment()
    from sentence_transformers import SentenceTransformer

    config = get_model_config()
    model_source = config.model_path or config.model_name
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return SentenceTransformer(
            model_source,
            cache_folder=config.cache_dir,
            local_files_only=config.local_files_only,
        )


def _get_model() -> SentenceTransformer:
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    with _embedding_lock:
        if _cached_model is None:
            _cached_model = _load_model()
        return _cached_model


def embed_single(text: str) -> np.ndarray:
    values = _get_model().encode([text], convert_to_numpy=True)
    return np.asarray(values[0], dtype=np.float32)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    values = _get_model().encode(texts, convert_to_numpy=True)
    return [np.asarray(row, dtype=np.float32) for row in values]


def embed_batch_to_lists(texts: list[str]) -> list[list[float]]:
    return [row.tolist() for row in embed_batch(texts)]


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


def reset_model() -> None:
    global _cached_model

    with _embedding_lock:
        _cached_model = None
