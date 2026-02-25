from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_embedding_lock = threading.Lock()
_cached_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


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
