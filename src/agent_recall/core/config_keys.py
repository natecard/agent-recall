from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_CONFIG_KEY_RENAMES: tuple[tuple[str, str], ...] = (
    ("onboarding.selected_agents", "onboarding.selected_sources"),
    ("adapters.token_budget", "adapters.default_token_budget"),
    ("retrieval.embedding_enabled", "retrieval.semantic_index_enabled"),
)


class LegacyConfigKeyError(ValueError):
    """Raised when config.yaml contains removed keys."""


def validate_no_legacy_config_keys(config: Mapping[str, Any]) -> None:
    found: list[tuple[str, str]] = []
    for old_path, new_path in LEGACY_CONFIG_KEY_RENAMES:
        if _contains_path(config, old_path):
            found.append((old_path, new_path))

    if not found:
        return

    mappings = ", ".join(f"{old} -> {new}" for old, new in found)
    raise LegacyConfigKeyError(
        f"Legacy config keys are no longer supported. Rename the following keys: {mappings}"
    )


def _contains_path(config: Mapping[str, Any], dotted_path: str) -> bool:
    current: Any = config
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return False
        current = current[segment]
    return True
