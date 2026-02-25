"""Tests for the semantic embedder module.

This module contains comprehensive unit tests for the semantic embedding pipeline,
including tests for single text embedding, batch processing, model caching, and
dimension verification. All tests use mocked models to avoid downloading actual
SentenceTransformer models.
"""

from __future__ import annotations

import threading

import numpy as np

from agent_recall.core import semantic_embedder


class _FakeModel:
    """Fake model for testing that returns deterministic embeddings."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.calls.append(texts)
        rows = [
            [float(index)] * semantic_embedder.get_embedding_dimension()
            for index, _ in enumerate(texts)
        ]
        return np.array(rows, dtype=np.float32)


def test_embed_single_returns_384_dim_vector(monkeypatch) -> None:
    """Test that embed_single returns a 384-dimensional numpy array."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    embedding = semantic_embedder.embed_single("hello world")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_embed_batch_processes_multiple_texts(monkeypatch) -> None:
    """Test that embed_batch correctly processes multiple texts in a single call."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    texts = ["hello", "world", "test"]
    embeddings = semantic_embedder.embed_batch(texts)

    assert len(embeddings) == 3
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (384,) for emb in embeddings)
    assert model.calls[-1] == texts


def test_get_model_caches_instance(monkeypatch) -> None:
    """Test that _get_model returns the cached instance on subsequent calls."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    load_calls = {"count": 0}

    def _loader() -> _FakeModel:
        load_calls["count"] += 1
        return model

    monkeypatch.setattr(semantic_embedder, "_load_model", _loader)

    first = semantic_embedder._get_model()
    second = semantic_embedder._get_model()

    assert first is second
    assert load_calls["count"] == 1


def test_reset_model_clears_cache(monkeypatch) -> None:
    """Test that reset_model clears the cached model instance."""
    semantic_embedder.reset_model()
    first = _FakeModel()
    second = _FakeModel()
    load_calls = {"count": 0}

    def _loader() -> _FakeModel:
        load_calls["count"] += 1
        return first if load_calls["count"] == 1 else second

    monkeypatch.setattr(semantic_embedder, "_load_model", _loader)

    model_a = semantic_embedder._get_model()
    semantic_embedder.reset_model()
    model_b = semantic_embedder._get_model()

    assert model_a is first
    assert model_b is second
    assert load_calls["count"] == 2


def test_embedding_dimension_is_384() -> None:
    """Test that the embedding dimension constant is 384."""
    assert semantic_embedder.get_embedding_dimension() == 384
    assert semantic_embedder.EMBEDDING_DIMENSION == 384


def test_embed_batch_to_lists_converts_to_python_lists(monkeypatch) -> None:
    """Test that embed_batch_to_lists returns Python lists instead of numpy arrays."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    values = semantic_embedder.embed_batch_to_lists(["hello", "world"])

    assert isinstance(values, list)
    assert len(values) == 2
    assert all(isinstance(v, list) for v in values)
    assert all(len(v) == 384 for v in values)
    assert all(isinstance(x, float) for v in values for x in v)


def test_embed_single_thread_safety(monkeypatch) -> None:
    """Test that embed_single is thread-safe when multiple threads call it."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    embeddings: list[np.ndarray] = []

    def _worker() -> None:
        emb = semantic_embedder.embed_single("test text")
        embeddings.append(emb)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(embeddings) == 10
    assert all(isinstance(e, np.ndarray) for e in embeddings)


def test_embed_batch_empty_list(monkeypatch) -> None:
    """Test that embed_batch handles empty list input correctly."""
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    result = semantic_embedder.embed_batch([])

    assert result == []
    assert len(model.calls) == 1
