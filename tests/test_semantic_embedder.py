from __future__ import annotations

import threading

import numpy as np

from agent_recall.core import semantic_embedder


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.calls.append(texts)
        rows = [
            [float(index)] * semantic_embedder.get_embedding_dimension()
            for index, _ in enumerate(texts)
        ]
        return np.array(
            rows,
            dtype=np.float32,
        )


def test_embed_single_returns_numpy_array(monkeypatch) -> None:
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    embedding = semantic_embedder.embed_single("hello world")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


def test_embed_batch_returns_list_of_numpy_arrays(monkeypatch) -> None:
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    embeddings = semantic_embedder.embed_batch(["hello", "world"])

    assert len(embeddings) == 2
    assert all(isinstance(item, np.ndarray) for item in embeddings)
    assert all(item.shape == (384,) for item in embeddings)


def test_embed_batch_to_lists_returns_python_lists(monkeypatch) -> None:
    semantic_embedder.reset_model()
    model = _FakeModel()
    monkeypatch.setattr(semantic_embedder, "_load_model", lambda: model)

    values = semantic_embedder.embed_batch_to_lists(["hello"])

    assert isinstance(values, list)
    assert isinstance(values[0], list)
    assert len(values[0]) == 384


def test_get_model_caches_instance(monkeypatch) -> None:
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


def test_get_model_is_thread_safe(monkeypatch) -> None:
    semantic_embedder.reset_model()
    load_calls = {"count": 0}

    def _loader() -> _FakeModel:
        load_calls["count"] += 1
        return _FakeModel()

    monkeypatch.setattr(semantic_embedder, "_load_model", _loader)

    models: list[object] = []

    def _worker() -> None:
        models.append(semantic_embedder._get_model())

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(models) == 10
    assert len({id(model) for model in models}) == 1
    assert load_calls["count"] == 1


def test_embedding_dimension_constant() -> None:
    assert semantic_embedder.get_embedding_dimension() == 384
