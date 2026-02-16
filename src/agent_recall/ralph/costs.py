from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_recall.ralph.iteration_store import IterationReport


@dataclass(frozen=True)
class Pricing:
    input_per_1k: float
    output_per_1k: float
    total_per_1k: float | None = None


@dataclass(frozen=True)
class CostBreakdown:
    item_id: str
    item_title: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class CostSummary:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    items: list[CostBreakdown]


_MODEL_PRICING: dict[str, Pricing] = {
    "gpt-4o": Pricing(input_per_1k=0.005, output_per_1k=0.015),
    "o4-mini": Pricing(input_per_1k=0.0015, output_per_1k=0.006),
    "gpt-5-codex": Pricing(input_per_1k=0.008, output_per_1k=0.024),
    "gpt-5.3-codex": Pricing(input_per_1k=0.008, output_per_1k=0.024),
    "claude-sonnet-4-20250514": Pricing(input_per_1k=0.003, output_per_1k=0.015),
    "claude-opus-4-6": Pricing(input_per_1k=0.015, output_per_1k=0.075),
    "claude-haiku-4-5-20251001": Pricing(input_per_1k=0.0008, output_per_1k=0.004),
}

_PROVIDER_DEFAULTS: dict[str, Pricing] = {
    "openai": Pricing(input_per_1k=0.006, output_per_1k=0.018, total_per_1k=0.012),
    "anthropic": Pricing(input_per_1k=0.004, output_per_1k=0.02, total_per_1k=0.012),
    "google": Pricing(input_per_1k=0.001, output_per_1k=0.002, total_per_1k=0.0015),
    "unknown": Pricing(input_per_1k=0.002, output_per_1k=0.002, total_per_1k=0.002),
}


def infer_provider(model: str | None) -> str:
    if not model:
        return "unknown"
    lowered = model.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gpt") or lowered.startswith("o1") or lowered.startswith("o4"):
        return "openai"
    if "gemini" in lowered or lowered.startswith("google"):
        return "google"
    return "unknown"


def resolve_pricing(model: str | None) -> Pricing:
    if model:
        pricing = _MODEL_PRICING.get(model)
        if pricing is not None:
            return pricing
    provider = infer_provider(model)
    return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["unknown"])


def estimate_cost(usage: dict[str, int] | None, model: str | None) -> float:
    if not usage:
        return 0.0
    prompt_tokens, completion_tokens, total_tokens = _normalize_tokens(usage)
    pricing = resolve_pricing(model)
    cost = 0.0
    if prompt_tokens or completion_tokens:
        cost += (prompt_tokens / 1000.0) * pricing.input_per_1k
        cost += (completion_tokens / 1000.0) * pricing.output_per_1k
        return cost
    if total_tokens:
        per_1k = pricing.total_per_1k
        if per_1k is None:
            per_1k = (pricing.input_per_1k + pricing.output_per_1k) / 2
        cost += (total_tokens / 1000.0) * per_1k
    return cost


def summarize_costs(reports: list[IterationReport]) -> CostSummary:
    items: dict[str, dict[str, Any]] = {}
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_cost = 0.0

    for report in reports:
        usage = report.token_usage
        if not usage:
            continue
        prompt_tokens, completion_tokens, report_total = _normalize_tokens(usage)
        cost = estimate_cost(usage, report.token_model)
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_tokens += report_total
        total_cost += cost

        item_id = report.item_id or "unknown"
        item_title = report.item_title or ""
        bucket = items.setdefault(
            item_id,
            {
                "item_id": item_id,
                "item_title": item_title,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
        )
        if not bucket["item_title"] and item_title:
            bucket["item_title"] = item_title
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += report_total
        bucket["cost_usd"] += cost

    sorted_items = sorted(
        (CostBreakdown(**value) for value in items.values()),
        key=lambda entry: entry.cost_usd,
        reverse=True,
    )

    return CostSummary(
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        items=sorted_items,
    )


def format_usd(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"${value:,.4f}"
    if value < 1000:
        return f"${value:,.2f}"
    return f"${value:,.0f}"


def budget_exceeded(total_cost_usd: float, budget_usd: float | None) -> bool:
    if budget_usd is None:
        return False
    return total_cost_usd > budget_usd


def _normalize_tokens(usage: dict[str, int]) -> tuple[int, int, int]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens
