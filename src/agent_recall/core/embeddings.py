from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")


def generate_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Generate a deterministic embedding without external model calls."""
    dims = max(1, dimensions)
    vector = [0.0] * dims
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for vectors of matching dimensions."""
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=False))
    return dot / (left_norm * right_norm)
