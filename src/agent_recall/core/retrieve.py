from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from uuid import UUID

from agent_recall.core.embeddings import cosine_similarity, generate_embedding
from agent_recall.core.semantic_embedder import embed_single, get_embedding_dimension
from agent_recall.storage.base import Storage
from agent_recall.storage.models import Chunk, SemanticLabel

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        storage: Storage,
        backend: str = "fts5",
        fusion_k: int = 60,
        rerank_enabled: bool = False,
        rerank_candidate_k: int = 20,
        fts_weight: float = 0.4,
        semantic_weight: float = 0.6,
        feedback_weight: float = 0.2,
    ):
        self.storage = storage
        self.backend = backend
        self.fusion_k = max(1, fusion_k)
        self.rerank_enabled = rerank_enabled
        self.rerank_candidate_k = max(1, rerank_candidate_k)
        self.fts_weight = max(0.0, float(fts_weight))
        self.semantic_weight = max(0.0, float(semantic_weight))
        self.feedback_weight = max(0.0, float(feedback_weight))

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
            chunks = self.search_hybrid(query=query, top_k=candidate_k)
        elif selected_backend == "vector_primary":
            chunks = self.search_vector_primary(query=query, top_k=candidate_k)
        else:
            chunks = self.storage.search_chunks_fts(query=query, top_k=candidate_k)

        if labels:
            allowed = {label.value for label in labels}
            chunks = [chunk for chunk in chunks if chunk.label.value in allowed]

        if selected_rerank:
            return self._rerank_chunks(query=query, chunks=chunks, top_k=top_k)

        return chunks[:top_k]

    def search_by_vector_similarity(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[Chunk]:
        limit = max(1, int(top_k))
        threshold = max(0.0, float(min_similarity))
        scored = self._rank_vector_candidates(query=query, min_similarity=threshold)
        logger.debug(
            "Vector search found %d chunks with similarity >= %.3f",
            len(scored),
            threshold,
        )
        return [chunk for chunk, _score in scored[:limit]]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        fts_weight: float | None = None,
        semantic_weight: float | None = None,
        fts_top_k: int = 20,
    ) -> list[Chunk]:
        limit = max(1, int(top_k))
        candidate_k = max(limit, int(fts_top_k))
        fts_chunks = self.storage.search_chunks_fts(query=query, top_k=candidate_k)
        semantic_chunks = self.search_by_vector_similarity(query=query, top_k=candidate_k)
        semantic_scored = self._rank_vector_candidates(query=query)[:candidate_k]

        fts_rank_scores = self._normalize_fts_scores(fts_chunks)
        semantic_scores = self._normalize_semantic_scores(semantic_scored)

        chunks_by_id: dict[UUID, Chunk] = {}
        for chunk in fts_chunks:
            chunks_by_id[chunk.id] = chunk
        for chunk in semantic_chunks:
            chunks_by_id[chunk.id] = chunk

        if not chunks_by_id:
            return []
        feedback_scores = self._feedback_scores(query=query, chunks=list(chunks_by_id.values()))

        weighted_fts = self.fts_weight if fts_weight is None else max(0.0, float(fts_weight))
        weighted_semantic = (
            self.semantic_weight if semantic_weight is None else max(0.0, float(semantic_weight))
        )
        weighted_results: list[tuple[float, float, float, float, str, Chunk]] = []
        for chunk_id, chunk in chunks_by_id.items():
            fts_score = fts_rank_scores.get(chunk_id, 0.0)
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            feedback_score = feedback_scores.get(chunk_id, 0.0)
            hybrid_score = (
                (weighted_fts * fts_score)
                + (weighted_semantic * semantic_score)
                + (feedback_score * self.feedback_weight)
            )
            weighted_results.append(
                (
                    hybrid_score,
                    feedback_score,
                    semantic_score,
                    fts_score,
                    str(chunk.id),
                    chunk,
                )
            )

        weighted_results.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4]))
        logger.debug(
            "Hybrid search: %d FTS + %d semantic = %d unique, top %d returned",
            len(fts_chunks),
            len(semantic_chunks),
            len(chunks_by_id),
            limit,
        )
        return [chunk for *_meta, chunk in weighted_results[:limit]]

    def _rank_vector_candidates(
        self,
        query: str,
        min_similarity: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        chunks = self.storage.list_chunks_with_embeddings()
        if not chunks:
            return []

        dimensions = next(
            (
                len(normalized)
                for chunk in chunks
                if (normalized := self._coerce_embedding(chunk.embedding)) is not None
            ),
            0,
        )
        if dimensions <= 0:
            return []

        query_embedding = self._build_query_embedding(query=query, dimensions=dimensions)
        scored: list[tuple[Chunk, float]] = []
        for chunk in chunks:
            normalized = self._coerce_embedding(chunk.embedding)
            if normalized is None:
                logger.warning("Skipping malformed embedding for chunk_id=%s", chunk.id)
                continue
            if len(normalized) != dimensions:
                continue
            similarity = cosine_similarity(query_embedding, normalized)
            if similarity >= min_similarity:
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
            self._build_query_embedding(query=query, dimensions=dimensions)
            if dimensions > 0
            else None
        )
        feedback_scores = self._feedback_scores(query=query, chunks=chunks)

        scored: list[tuple[float, float, float, float, int, str, Chunk]] = []
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

            feedback_score = feedback_scores.get(chunk.id, 0.0)
            score = lexical_score + (similarity * 0.75)
            score += feedback_score * (0.3 + self.feedback_weight)
            scored.append(
                (
                    score,
                    feedback_score,
                    similarity,
                    lexical_score,
                    rank,
                    str(chunk.id),
                    chunk,
                )
            )

        scored.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4], row[5]))
        return [chunk for *_meta, chunk in scored[:top_k]]

    def search_vector_primary(self, query: str, top_k: int = 5) -> list[Chunk]:
        limit = max(1, int(top_k))
        semantic = self.search_by_vector_similarity(query=query, top_k=limit)
        if len(semantic) >= limit:
            return semantic[:limit]

        seen = {chunk.id for chunk in semantic}
        lexical = self.storage.search_chunks_fts(query=query, top_k=max(limit, self.fusion_k))
        merged = list(semantic)
        for chunk in lexical:
            if chunk.id in seen:
                continue
            merged.append(chunk)
            seen.add(chunk.id)
            if len(merged) >= limit:
                break
        return merged[:limit]

    def _feedback_scores(self, query: str, chunks: list[Chunk]) -> dict[UUID, float]:
        if not chunks:
            return {}
        score_fn = getattr(self.storage, "get_retrieval_feedback_scores", None)
        if not callable(score_fn):
            return {}
        try:
            raw_scores = score_fn(query=query, chunk_ids=[chunk.id for chunk in chunks])
        except (NotImplementedError, ValueError, TypeError):
            return {}
        if not isinstance(raw_scores, dict):
            return {}

        normalized: dict[UUID, float] = {}
        for key, value in raw_scores.items():
            chunk_id: UUID | None
            if isinstance(key, UUID):
                chunk_id = key
            else:
                try:
                    chunk_id = UUID(str(key))
                except ValueError:
                    chunk_id = None
            if chunk_id is None:
                continue
            try:
                normalized[chunk_id] = max(-1.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return normalized

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

    @staticmethod
    def _build_query_embedding(query: str, dimensions: int) -> list[float]:
        if dimensions == get_embedding_dimension():
            try:
                return embed_single(query).tolist()
            except Exception:  # noqa: BLE001
                pass
        return generate_embedding(query, dimensions=dimensions)

    @staticmethod
    def _coerce_embedding(raw: object) -> list[float] | None:
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            return None

        values: list[float] = []
        for item in raw:
            if isinstance(item, int | float):
                values.append(float(item))
            else:
                return None

        return values if values else None

    @staticmethod
    def _normalize_fts_scores(chunks: list[Chunk]) -> dict[UUID, float]:
        if not chunks:
            return {}

        ranked = {chunk.id: (1.0 / float(index)) for index, chunk in enumerate(chunks, start=1)}
        max_score = max(ranked.values()) if ranked else 1.0
        if max_score <= 0.0:
            return {chunk_id: 0.0 for chunk_id in ranked}
        return {chunk_id: (score / max_score) for chunk_id, score in ranked.items()}

    @staticmethod
    def _normalize_semantic_scores(
        chunks: list[tuple[Chunk, float]],
    ) -> dict[UUID, float]:
        if not chunks:
            return {}

        normalized: dict[UUID, float] = {}
        for chunk, score in chunks:
            clamped = max(0.0, min(1.0, float(score)))
            normalized[chunk.id] = clamped
        return normalized
