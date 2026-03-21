from __future__ import annotations

from uuid import UUID

from agent_recall.core.retrieval_feedback import evaluate_feedback_impact
from agent_recall.storage.models import Chunk, ChunkSource, SemanticLabel


def test_evaluate_feedback_impact_reports_delta(storage, monkeypatch) -> None:
    query = "bounded retries"
    first = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000041"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="bounded retries alpha",
        label=SemanticLabel.PATTERN,
    )
    second = Chunk(
        id=UUID("00000000-0000-0000-0000-000000000042"),
        source=ChunkSource.MANUAL,
        source_ids=[],
        content="bounded retries beta",
        label=SemanticLabel.PATTERN,
    )
    storage.store_chunk(first)
    storage.store_chunk(second)
    storage.record_retrieval_feedback(query=query, chunk_id=first.id, score=-1)
    storage.record_retrieval_feedback(query=query, chunk_id=second.id, score=1)

    monkeypatch.setattr(storage, "search_chunks_fts", lambda query, top_k: [first, second][:top_k])
    monkeypatch.setattr(
        "agent_recall.core.retrieval_feedback.Retriever.search",
        lambda self, query, top_k, backend, rerank, rerank_candidate_k: [second, first][:top_k],
    )

    report = evaluate_feedback_impact(storage, top_k=2, min_labels_per_query=2)
    assert report.queries_evaluated == 1
    assert report.improved_queries == 1
    assert report.mean_delta > 0


def test_evaluate_feedback_impact_returns_empty_when_no_labels(storage) -> None:
    report = evaluate_feedback_impact(storage, top_k=5)
    assert report.queries_evaluated == 0
    assert report.results == []
