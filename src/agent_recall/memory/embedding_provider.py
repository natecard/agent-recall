from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import httpx

from agent_recall.core.embeddings import generate_embedding
from agent_recall.core.semantic_embedder import (
    configure_model,
    embed_single,
    get_embedding_dimension,
)


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    provider: str
    model: str
    estimated_tokens: int
    estimated_cost_usd: float


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Embed text list and return vectors + usage metadata."""
        ...


class LocalEmbeddingProvider:
    """Local embedding provider with optional model path and in-memory LRU cache."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_path: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        dimensions: int = 64,
        cache_size: int = 2_000,
        strict_local_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only or model_path)
        self.dimensions = max(8, int(dimensions))
        self.cache_size = max(100, int(cache_size))
        self.strict_local_model = strict_local_model
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        vectors: list[list[float]] = []
        estimated_tokens = 0
        for text in texts:
            normalized = text.strip()
            estimated_tokens += _estimate_tokens(normalized)
            cache_key = (
                f"{self.model_name or self.model_path or 'default'}::"
                f"{self.cache_dir or '-'}::{self.dimensions}::{normalized}"
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                vectors.append(list(cached))
                self._cache.move_to_end(cache_key)
                continue
            vector = self._embed_one(normalized)
            vectors.append(vector)
            self._cache[cache_key] = vector
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return EmbeddingResponse(
            vectors=vectors,
            provider="local",
            model=self.model_path or self.model_name or "deterministic",
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=0.0,
        )

    def _embed_one(self, text: str) -> list[float]:
        if self.dimensions == get_embedding_dimension():
            try:
                configure_model(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
                return embed_single(text).tolist()
            except Exception as exc:  # noqa: BLE001
                if self.strict_local_model:
                    raise RuntimeError("Local embedding model is unavailable.") from exc
        return generate_embedding(text, dimensions=self.dimensions)


class ExternalEmbeddingProvider:
    """External API embedding provider with timeout and cost guardrails."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_seconds: float = 10.0,
        max_cost_usd: float = 1.0,
        cost_per_1k_tokens_usd: float = 0.0005,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_cost_usd = max(0.0, float(max_cost_usd))
        self.cost_per_1k_tokens_usd = max(0.0, float(cost_per_1k_tokens_usd))
        self.spent_usd = 0.0

    def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        token_estimate = sum(_estimate_tokens(text) for text in texts)
        projected_cost = (token_estimate / 1000.0) * self.cost_per_1k_tokens_usd
        if self.spent_usd + projected_cost > self.max_cost_usd:
            raise RuntimeError(
                "Embedding request blocked by cost guardrail. "
                f"Budget={self.max_cost_usd:.4f} USD, "
                f"projected={self.spent_usd + projected_cost:.4f} USD."
            )

        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing embedding API key in env var {self.api_key_env}")
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {api_key}"}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
            response.raise_for_status()
        data = response.json()
        vectors = _parse_embedding_vectors(data)
        self.spent_usd += projected_cost
        return EmbeddingResponse(
            vectors=vectors,
            provider="external",
            model=self.model,
            estimated_tokens=token_estimate,
            estimated_cost_usd=projected_cost,
        )


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.strip()) / 4))


def _parse_embedding_vectors(payload: object) -> list[list[float]]:
    if isinstance(payload, dict):
        payload_dict = {str(key): value for key, value in payload.items()}
        raw_vectors = payload_dict.get("vectors")
        if isinstance(raw_vectors, list):
            vectors: list[list[float]] = []
            for item in raw_vectors:
                vector = _coerce_vector(item)
                if vector is not None:
                    vectors.append(vector)
            return vectors
        data = payload_dict.get("data")
        if isinstance(data, list):
            vectors: list[list[float]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                item_dict = {str(key): value for key, value in item.items()}
                vector = _coerce_vector(item_dict.get("embedding"))
                if vector is not None:
                    vectors.append(vector)
            return vectors
    raise RuntimeError("Embedding provider returned an unsupported response shape.")


def _coerce_vector(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    vector: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        vector.append(float(item))
    return vector if vector else None
