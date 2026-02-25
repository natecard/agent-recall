from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_recall.core.embeddings import cosine_similarity
from agent_recall.storage.base import Storage


class EmbeddingDiagnostics:
    def __init__(self, storage: Storage):
        self.storage = storage

    def get_coverage_stats(self) -> dict[str, int | float]:
        chunks = self.storage.list_chunks()
        total_chunks = len(chunks)
        embedded_chunks = sum(1 for chunk in chunks if chunk.embedding is not None)
        coverage = (100.0 * embedded_chunks / total_chunks) if total_chunks > 0 else 0.0
        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "coverage_percent": coverage,
            "pending": max(0, total_chunks - embedded_chunks),
        }

    def get_similarity_distribution(self, max_pairs: int = 10_000) -> dict[str, float]:
        vectors = [chunk.embedding for chunk in self.storage.list_chunks_with_embeddings()]
        vectors = [vector for vector in vectors if vector]
        if len(vectors) < 2:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        similarities: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(cosine_similarity(vectors[i], vectors[j]))
                if len(similarities) >= max_pairs:
                    break
            if len(similarities) >= max_pairs:
                break

        if not similarities:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values = sorted(similarities)
        n = len(values)
        mean = sum(values) / n
        if n % 2 == 0:
            median = (values[(n // 2) - 1] + values[n // 2]) / 2.0
        else:
            median = values[n // 2]
        variance = sum((value - mean) ** 2 for value in values) / n

        return {
            "mean": mean,
            "median": median,
            "std": variance**0.5,
            "min": values[0],
            "max": values[-1],
        }

    def check_stale_embeddings(self, threshold_days: int = 90) -> dict[str, int]:
        cutoff = datetime.now(UTC) - timedelta(days=max(0, int(threshold_days)))
        embedded_chunks = [
            chunk for chunk in self.storage.list_chunks() if chunk.embedding is not None
        ]
        stale_count = sum(1 for chunk in embedded_chunks if chunk.created_at < cutoff)
        return {
            "threshold_days": max(0, int(threshold_days)),
            "embedded_chunks": len(embedded_chunks),
            "stale_chunks": stale_count,
        }

    def estimate_embedding_size(self) -> dict[str, int | float]:
        embedded = [chunk for chunk in self.storage.list_chunks() if chunk.embedding is not None]
        total_bytes = sum(len(chunk.embedding or []) * 4 for chunk in embedded)
        per_chunk_kb = (total_bytes / max(1, len(embedded))) / 1024.0
        return {
            "total_bytes": total_bytes,
            "per_chunk_kb": per_chunk_kb,
        }
