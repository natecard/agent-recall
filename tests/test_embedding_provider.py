from __future__ import annotations

import pytest

from agent_recall.memory.embedding_provider import ExternalEmbeddingProvider, LocalEmbeddingProvider


def test_local_embedding_provider_caches_results(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_generate(text: str, dimensions: int = 64) -> list[float]:
        _ = text
        calls["count"] += 1
        return [float(dimensions), 1.0]

    monkeypatch.setattr("agent_recall.memory.embedding_provider.generate_embedding", _fake_generate)
    provider = LocalEmbeddingProvider(dimensions=64, cache_size=10)
    first = provider.embed_texts(["repeat text", "repeat text"])
    second = provider.embed_texts(["repeat text"])
    assert len(first.vectors) == 2
    assert len(second.vectors) == 1
    assert calls["count"] == 1


def test_external_embedding_provider_cost_guardrail(monkeypatch) -> None:
    monkeypatch.setenv("EMB_KEY", "secret")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.1, 0.2]}]}

    class _Client:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            _ = (exc_type, exc, tb)

        def post(self, url: str, json: dict[str, object], headers: dict[str, str]):
            _ = (url, json, headers)
            return _Response()

    monkeypatch.setattr("agent_recall.memory.embedding_provider.httpx.Client", _Client)

    provider = ExternalEmbeddingProvider(
        base_url="https://emb.example",
        api_key_env="EMB_KEY",
        model="emb-small",
        max_cost_usd=0.001,
        cost_per_1k_tokens_usd=0.001,
    )
    response = provider.embed_texts(["short text"])
    assert response.vectors

    with pytest.raises(RuntimeError, match="cost guardrail"):
        provider.embed_texts(["x" * 20_000])
