from __future__ import annotations

import re
from uuid import UUID

from agent_recall.core.embeddings import cosine_similarity, generate_embedding
from agent_recall.storage.base import Storage
from agent_recall.storage.models import Chunk, SemanticLabel

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Retriever:
    def __init__(
        self,
        storage: Storage,
        backend: str = "fts5",
        fusion_k: int = 60,
        rerank_enabled: bool = False,
        rerank_candidate_k: int = 20,
    ):
        self.storage = storage
        self.backend = backend
        self.fusion_k = max(1, fusion_k)
        self.rerank_enabled = rerank_enabled
        self.rerank_candidate_k = max(1, rerank_candidate_k)

    def search(
        self,
        query: str,
        top_k: int = 5,
        labels: list[SemanticLabel] | None = None,
        backend: str | None = None,
        rerank: bool | None = None,
        rerank_candidate_k: int | None = None,
    ) -> list[Chunk]:
        selected_backend = (backend or self.backend).strip().lower()
        selected_rerank = self.rerank_enabled if rerank is None else rerank
        candidate_k = (
            max(top_k, rerank_candidate_k or self.rerank_candidate_k) if selected_rerank else top_k
        )

        if selected_backend == "hybrid":
            chunks = self._search_hybrid(query=query, top_k=candidate_k)
        else:
            chunks = self.storage.search_chunks_fts(query=query, top_k=candidate_k)

        if labels:
            allowed = {label.value for label in labels}
            chunks = [chunk for chunk in chunks if chunk.label.value in allowed]

        if selected_rerank:
            return self._rerank_chunks(query=query, chunks=chunks, top_k=top_k)

        return chunks[:top_k]

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

    def _rerank_chunks(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []

        query_terms = self._tokenize(query)
        query_text = query.strip().lower()
        dimensions = next((len(chunk.embedding) for chunk in chunks if chunk.embedding), 0)
        query_embedding = (
            generate_embedding(query, dimensions=dimensions) if dimensions > 0 else None
        )

        scored: list[tuple[float, float, float, int, str, Chunk]] = []
        for rank, chunk in enumerate(chunks, start=1):
            lexical_score = self._lexical_overlap_score(
                query_terms=query_terms,
                query_text=query_text,
                chunk=chunk,
            )
            similarity = 0.0
            if (
                query_embedding is not None
                and chunk.embedding is not None
                and len(chunk.embedding) == dimensions
            ):
                similarity = max(0.0, cosine_similarity(query_embedding, chunk.embedding))

            score = lexical_score + (similarity * 0.75)
            scored.append((score, similarity, lexical_score, rank, str(chunk.id), chunk))

        scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))
        return [chunk for *_meta, chunk in scored[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {match.group(0) for match in _TOKEN_PATTERN.finditer(text.lower())}

    def _lexical_overlap_score(
        self,
        query_terms: set[str],
        query_text: str,
        chunk: Chunk,
    ) -> float:
        if not query_terms:
            return 0.0

        searchable = f"{chunk.content} {' '.join(chunk.tags)}"
        chunk_terms = self._tokenize(searchable)
        if not chunk_terms:
            return 0.0

        overlap = len(query_terms & chunk_terms) / float(len(query_terms))
        phrase_bonus = 0.15 if query_text and query_text in searchable.lower() else 0.0
        return overlap + phrase_bonus
