from __future__ import annotations

from typing import Any


def key_score_desc_id(score: float, identifier: Any) -> tuple[float, str]:
    """Sort by descending score and deterministic identifier tie-break."""
    return (-float(score), str(identifier))


def key_component_score_desc(
    *,
    score: float,
    feedback: float,
    semantic: float,
    lexical: float,
    identifier: Any,
    rank_hint: int | None = None,
) -> tuple[Any, ...]:
    """Sort by weighted score components with optional prior-rank fallback."""
    base: tuple[Any, ...] = (
        -float(score),
        -float(feedback),
        -float(semantic),
        -float(lexical),
    )
    if rank_hint is None:
        return (*base, str(identifier))
    return (*base, int(rank_hint), str(identifier))


def key_timestamp_name(timestamp: float, name: str) -> tuple[float, str]:
    """Sort by timestamp then by name for deterministic ordering."""
    return (float(timestamp), str(name))


def key_optional_timestamp_name(
    timestamp: float | None,
    name: str,
    *,
    missing_last: bool,
) -> tuple[float, str]:
    """Sort by optional timestamp, using deterministic missing-value placement."""
    if timestamp is None:
        marker = float("inf") if missing_last else 0.0
    else:
        marker = float(timestamp)
    return (marker, str(name))


def key_timestamp_index(
    timestamp: float | None,
    index: int,
    *,
    missing_last: bool,
) -> tuple[float, int]:
    """Sort by optional timestamp then stable input index."""
    if timestamp is None:
        marker = float("inf") if missing_last else 0.0
    else:
        marker = float(timestamp)
    return (marker, int(index))


def key_timestamp_desc_id(timestamp: float, identifier: Any) -> tuple[float, str]:
    """Sort by descending timestamp with deterministic identifier tie-break."""
    return (-float(timestamp), str(identifier))
