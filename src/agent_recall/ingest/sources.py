from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    display_name: str
    aliases: tuple[str, ...] = ()
    location_attr: str | None = None
    location_suffix: tuple[str, ...] = ()


SOURCE_DEFINITIONS: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        name="cursor",
        display_name="Cursor",
        aliases=("cursor",),
        location_attr="storage_dir",
    ),
    SourceDefinition(
        name="claude-code",
        display_name="Claude Code",
        aliases=("claude-code", "claude_code", "claudecode", "claude"),
        location_attr="claude_dir",
        location_suffix=("projects",),
    ),
    SourceDefinition(
        name="opencode",
        display_name="OpenCode",
        aliases=("opencode", "open-code", "open_code"),
        location_attr="opencode_dir",
        location_suffix=("storage", "session"),
    ),
    SourceDefinition(
        name="codex",
        display_name="OpenAI Codex",
        aliases=("codex", "openai-codex", "openai_codex"),
        location_attr="codex_dir",
        location_suffix=("sessions",),
    ),
)

SOURCE_BY_NAME: dict[str, SourceDefinition] = {item.name: item for item in SOURCE_DEFINITIONS}
VALID_SOURCE_NAMES: tuple[str, ...] = tuple(item.name for item in SOURCE_DEFINITIONS)


def _build_alias_lookup() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source in SOURCE_DEFINITIONS:
        for alias in source.aliases:
            normalized = alias.strip().lower().replace("_", "-")
            if normalized:
                mapping[normalized] = source.name
        mapping[source.name] = source.name
    return mapping


SOURCE_NAME_ALIASES = _build_alias_lookup()


def normalize_source_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    return SOURCE_NAME_ALIASES.get(normalized, normalized)


def resolve_source_location_hint(ingester: Any) -> Path | None:
    source_name = normalize_source_name(str(getattr(ingester, "source_name", "")))
    source_def = SOURCE_BY_NAME.get(source_name)
    if source_def is None or source_def.location_attr is None:
        return None

    location_value = getattr(ingester, source_def.location_attr, None)
    if location_value is None:
        return None

    try:
        location = Path(location_value)
        for segment in source_def.location_suffix:
            location = location / segment
        return location.expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None
