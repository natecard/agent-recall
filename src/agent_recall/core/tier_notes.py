from __future__ import annotations

import re

_BULLET_PREFIX_RE = re.compile(r"^\s*-\s*\[[A-Z_]+\]\s*", re.IGNORECASE)
_RECENT_PREFIX_RE = re.compile(r"^\s*\*\*\d{4}-\d{2}-\d{2}\*\*:\s*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

NEGATIVE_POLARITY_TOKENS = {
    "avoid",
    "ban",
    "deny",
    "disable",
    "dont",
    "never",
    "no",
    "not",
    "without",
}
POSITIVE_POLARITY_TOKENS = {
    "always",
    "enable",
    "ensure",
    "must",
    "prefer",
    "require",
    "should",
    "use",
}
SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def normalize_tier_line(line: str) -> str:
    return " ".join(line.strip().lower().split())


def normalize_tier_content(content: str) -> str:
    return " ".join(content.strip().lower().split())


def semantic_tokens(text: str) -> list[str]:
    cleaned = _BULLET_PREFIX_RE.sub("", text.strip().lower())
    cleaned = _RECENT_PREFIX_RE.sub("", cleaned)
    return _TOKEN_RE.findall(cleaned)


def semantic_key(text: str) -> str:
    return " ".join(semantic_tokens(text))


def topic_key(text: str) -> str:
    tokens = semantic_tokens(text)
    reduced = [
        token
        for token in tokens
        if token not in SEMANTIC_STOPWORDS
        and token not in NEGATIVE_POLARITY_TOKENS
        and token not in POSITIVE_POLARITY_TOKENS
    ]
    if not reduced:
        reduced = [token for token in tokens if token not in SEMANTIC_STOPWORDS]
    return " ".join(reduced[:10])


def polarity(text: str) -> str:
    tokens = set(semantic_tokens(text))
    has_negative = bool(tokens.intersection(NEGATIVE_POLARITY_TOKENS))
    has_positive = bool(tokens.intersection(POSITIVE_POLARITY_TOKENS))
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return "neutral"


def semantic_token_set(text: str) -> set[str]:
    return {token for token in semantic_tokens(text) if token not in SEMANTIC_STOPWORDS}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
