from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from agent_recall.core.retrieve import Retriever
from agent_recall.storage.base import Storage
from agent_recall.storage.models import Chunk


@dataclass(frozen=True)
class FeedbackQueryEvaluation:
    query: str
    baseline_score: float
    feedback_score: float
    delta: float
    positive_labels: int
    negative_labels: int
    baseline_top_ids: list[str]
    feedback_top_ids: list[str]


@dataclass(frozen=True)
class FeedbackEvaluationReport:
    queries_evaluated: int
    mean_baseline_score: float
    mean_feedback_score: float
    mean_delta: float
    improved_queries: int
    regressed_queries: int
    unchanged_queries: int
    results: list[FeedbackQueryEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries_evaluated": self.queries_evaluated,
            "mean_baseline_score": self.mean_baseline_score,
            "mean_feedback_score": self.mean_feedback_score,
            "mean_delta": self.mean_delta,
            "improved_queries": self.improved_queries,
            "regressed_queries": self.regressed_queries,
            "unchanged_queries": self.unchanged_queries,
            "results": [
                {
                    "query": result.query,
                    "baseline_score": result.baseline_score,
                    "feedback_score": result.feedback_score,
                    "delta": result.delta,
                    "positive_labels": result.positive_labels,
                    "negative_labels": result.negative_labels,
                    "baseline_top_ids": result.baseline_top_ids,
                    "feedback_top_ids": result.feedback_top_ids,
                }
                for result in self.results
            ],
        }


def _ranking_relevance_score(chunks: list[Chunk], labels: dict[UUID, float]) -> float:
    score = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        label_score = labels.get(chunk.id)
        if label_score is None:
            continue
        score += label_score / float(rank)
    return score


def evaluate_feedback_impact(
    storage: Storage,
    *,
    top_k: int = 10,
    min_labels_per_query: int = 2,
    feedback_limit: int = 2_000,
) -> FeedbackEvaluationReport:
    raw_feedback = storage.list_retrieval_feedback(limit=max(1, int(feedback_limit)))
    grouped: dict[str, dict[UUID, list[int]]] = defaultdict(lambda: defaultdict(list))

    for item in raw_feedback:
        query = str(item.get("query_text", "")).strip()
        chunk_id_raw = item.get("chunk_id")
        score_raw = item.get("score")
        if not query:
            continue
        try:
            chunk_id = UUID(str(chunk_id_raw))
            if isinstance(score_raw, int):
                score = score_raw
            elif isinstance(score_raw, str):
                score = int(score_raw.strip())
            else:
                continue
        except (ValueError, TypeError):
            continue
        if score not in {-1, 1}:
            continue
        grouped[query][chunk_id].append(score)

    evaluated: list[FeedbackQueryEvaluation] = []
    for query, chunk_scores in grouped.items():
        averaged: dict[UUID, float] = {}
        positives = 0
        negatives = 0
        for chunk_id, values in chunk_scores.items():
            avg_score = sum(values) / float(len(values))
            averaged[chunk_id] = max(-1.0, min(1.0, avg_score))
            if avg_score > 0:
                positives += 1
            elif avg_score < 0:
                negatives += 1
        if positives + negatives < max(1, int(min_labels_per_query)):
            continue

        limit = max(1, int(top_k))
        baseline = storage.search_chunks_fts(query=query, top_k=limit)
        retriever = Retriever(storage, backend="hybrid", rerank_enabled=True)
        feedback_ranked = retriever.search(
            query=query,
            top_k=limit,
            backend="hybrid",
            rerank=True,
            rerank_candidate_k=max(limit, 20),
        )
        baseline_score = _ranking_relevance_score(baseline, averaged)
        feedback_score = _ranking_relevance_score(feedback_ranked, averaged)
        evaluated.append(
            FeedbackQueryEvaluation(
                query=query,
                baseline_score=baseline_score,
                feedback_score=feedback_score,
                delta=feedback_score - baseline_score,
                positive_labels=positives,
                negative_labels=negatives,
                baseline_top_ids=[str(chunk.id) for chunk in baseline],
                feedback_top_ids=[str(chunk.id) for chunk in feedback_ranked],
            )
        )

    evaluated.sort(key=lambda item: item.query)
    count = len(evaluated)
    if count == 0:
        return FeedbackEvaluationReport(
            queries_evaluated=0,
            mean_baseline_score=0.0,
            mean_feedback_score=0.0,
            mean_delta=0.0,
            improved_queries=0,
            regressed_queries=0,
            unchanged_queries=0,
            results=[],
        )

    baseline_mean = sum(item.baseline_score for item in evaluated) / float(count)
    feedback_mean = sum(item.feedback_score for item in evaluated) / float(count)
    improved = sum(1 for item in evaluated if item.delta > 0)
    regressed = sum(1 for item in evaluated if item.delta < 0)
    unchanged = count - improved - regressed
    return FeedbackEvaluationReport(
        queries_evaluated=count,
        mean_baseline_score=baseline_mean,
        mean_feedback_score=feedback_mean,
        mean_delta=feedback_mean - baseline_mean,
        improved_queries=improved,
        regressed_queries=regressed,
        unchanged_queries=unchanged,
        results=evaluated,
    )
