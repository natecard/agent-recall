from __future__ import annotations

import logging

from agent_recall.core.semantic_embedder import embed_batch_to_lists
from agent_recall.storage.base import Storage

logger = logging.getLogger(__name__)


class EmbeddingIndexer:
    def __init__(self, storage: Storage, batch_size: int = 32) -> None:
        self.storage = storage
        self.batch_size = max(1, int(batch_size))

    def index_missing_embeddings(self, max_chunks: int = 0) -> dict[str, int]:
        chunks = [chunk for chunk in self.storage.list_chunks() if chunk.embedding is None]
        if max_chunks > 0:
            chunks = chunks[:max_chunks]

        if not chunks:
            return {"indexed": 0, "skipped": 0}

        logger.info("Found %d chunks without embeddings.", len(chunks))

        indexed = 0
        with _progress(total=len(chunks), desc="Embedding chunks") as progress:
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                texts = [chunk.content for chunk in batch]
                embeddings = embed_batch_to_lists(texts)
                for chunk, embedding in zip(batch, embeddings, strict=False):
                    self.storage.index_chunk_embedding(chunk.id, embedding)
                    indexed += 1
                    if progress is not None:
                        progress.update(1)

        return {"indexed": indexed, "skipped": 0}

    def get_indexing_stats(self) -> dict[str, int]:
        all_chunks = self.storage.list_chunks()
        total_chunks = len(all_chunks)
        embedded_chunks = sum(1 for chunk in all_chunks if chunk.embedding is not None)
        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "pending": max(0, total_chunks - embedded_chunks),
        }


class _NoopProgress:
    def __enter__(self) -> _NoopProgress:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        _ = (exc_type, exc_val, exc_tb)

    def update(self, amount: int = 1) -> None:
        _ = amount


def _progress(total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc)
    except Exception:  # noqa: BLE001
        return _NoopProgress()
