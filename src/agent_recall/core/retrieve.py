from __future__ import annotations

from uuid import UUID

from agent_recall.core.embeddings import cosine_similarity, generate_embedding
from agent_recall.storage.models import Chunk, SemanticLabel
from agent_recall.storage.sqlite import SQLiteStorage


class Retriever:
    def __init__(self, storage: SQLiteStorage, backend: str = "fts5", fusion_k: int = 60):
        self.storage = storage
        self.backend = backend
        self.fusion_k = max(1, fusion_k)

    def search(
        self,
        query: str,
        top_k: int = 5,
        labels: list[SemanticLabel] | None = None,
        backend: str | None = None,
    ) -> list[Chunk]:
        selected_backend = (backend or self.backend).strip().lower()

        if selected_backend == "hybrid":
            chunks = self._search_hybrid(query=query, top_k=top_k)
        else:
            chunks = self.storage.search_chunks_fts(query=query, top_k=top_k)

        if not labels:
            return chunks
        allowed = {label.value for label in labels}
        return [chunk for chunk in chunks if chunk.label.value in allowed]

    def _search_hybrid(self, query: str, top_k: int) -> list[Chunk]:
        candidate_k = max(top_k, top_k * 4)
        fts_chunks = self.storage.search_chunks_fts(query=query, top_k=candidate_k)
        vector_scored = self._rank_vector_candidates(query=query)

        if not fts_chunks and not vector_scored:
            return []

        vector_top = vector_scored[:candidate_k]
        vector_chunks = [chunk for chunk, _score in vector_top]
        vector_similarity = {chunk.id: score for chunk, score in vector_top}
        vector_rank = {chunk.id: index for index, chunk in enumerate(vector_chunks, start=1)}
        fts_rank = {chunk.id: index for index, chunk in enumerate(fts_chunks, start=1)}

        chunks_by_id: dict[UUID, Chunk] = {}
        for chunk in fts_chunks:
            chunks_by_id[chunk.id] = chunk
        for chunk in vector_chunks:
            chunks_by_id[chunk.id] = chunk

        scored: list[tuple[float, float, int, str, Chunk]] = []
        for chunk_id, chunk in chunks_by_id.items():
            score = 0.0
            rank_fts = fts_rank.get(chunk_id)
            rank_vector = vector_rank.get(chunk_id)
            if rank_fts is not None:
                score += 1.0 / (self.fusion_k + rank_fts)
            if rank_vector is not None:
                score += 1.0 / (self.fusion_k + rank_vector)
            similarity = vector_similarity.get(chunk_id, 0.0)
            scored.append(
                (
                    score,
                    similarity,
                    rank_fts if rank_fts is not None else 1_000_000,
                    str(chunk.id),
                    chunk,
                )
            )

        scored.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
        return [chunk for *_meta, chunk in scored[:top_k]]

    def _rank_vector_candidates(self, query: str) -> list[tuple[Chunk, float]]:
        chunks = self.storage.list_chunks_with_embeddings()
        if not chunks:
            return []

        dimensions = next((len(chunk.embedding) for chunk in chunks if chunk.embedding), 0)
        if dimensions <= 0:
            return []

        query_embedding = generate_embedding(query, dimensions=dimensions)
        scored: list[tuple[Chunk, float]] = []
        for chunk in chunks:
            if chunk.embedding is None or len(chunk.embedding) != dimensions:
                continue
            similarity = cosine_similarity(query_embedding, chunk.embedding)
            if similarity > 0.0:
                scored.append((chunk, similarity))

        scored.sort(key=lambda row: (-row[1], str(row[0].id)))
        return scored
