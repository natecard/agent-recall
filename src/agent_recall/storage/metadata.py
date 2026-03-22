from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_recall.storage.normalize import normalize_non_empty_text, parse_json_object


@dataclass(frozen=True)
class AttributionMetadata:
    agent_source: str | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> AttributionMetadata:
        raw = parse_json_object(value)
        known_keys = {"agent_source", "provider", "model"}
        return cls(
            agent_source=normalize_non_empty_text(raw.get("agent_source")),
            provider=normalize_non_empty_text(raw.get("provider")),
            model=normalize_non_empty_text(raw.get("model")),
            extra={key: raw[key] for key in raw.keys() - known_keys},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        if self.agent_source:
            payload["agent_source"] = self.agent_source
        if self.provider:
            payload["provider"] = self.provider
        if self.model:
            payload["model"] = self.model
        return payload


@dataclass(frozen=True)
class EntryMetadata:
    attribution: AttributionMetadata | None = None
    evidence: str | None = None
    source_tool: str | None = None
    extracted_at: str | None = None
    ingested_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> EntryMetadata:
        raw = parse_json_object(value)
        known_keys = {"attribution", "evidence", "source_tool", "extracted_at", "ingested_from"}
        attribution_raw = raw.get("attribution")
        attribution = (
            AttributionMetadata.from_value(attribution_raw) if attribution_raw is not None else None
        )
        return cls(
            attribution=attribution,
            evidence=normalize_non_empty_text(raw.get("evidence")),
            source_tool=normalize_non_empty_text(raw.get("source_tool")),
            extracted_at=normalize_non_empty_text(raw.get("extracted_at")),
            ingested_from=normalize_non_empty_text(raw.get("ingested_from")),
            extra={key: raw[key] for key in raw.keys() - known_keys},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        if self.attribution is not None:
            payload["attribution"] = self.attribution.to_dict()
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.source_tool:
            payload["source_tool"] = self.source_tool
        if self.extracted_at:
            payload["extracted_at"] = self.extracted_at
        if self.ingested_from:
            payload["ingested_from"] = self.ingested_from
        return payload


@dataclass(frozen=True)
class FeedbackMetadata:
    surface: str | None = None
    source_session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: object) -> FeedbackMetadata:
        raw = parse_json_object(value)
        known_keys = {"surface", "source_session_id"}
        return cls(
            surface=normalize_non_empty_text(raw.get("surface")),
            source_session_id=normalize_non_empty_text(raw.get("source_session_id")),
            extra={key: raw[key] for key in raw.keys() - known_keys},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        if self.surface:
            payload["surface"] = self.surface
        if self.source_session_id:
            payload["source_session_id"] = self.source_session_id
        return payload


def build_entry_metadata(
    *,
    attribution: AttributionMetadata | None = None,
    evidence: str | None = None,
    source_tool: str | None = None,
    extracted_at: str | None = None,
    ingested_from: str | None = None,
    base: object = None,
) -> dict[str, Any]:
    existing = EntryMetadata.from_value(base)
    resolved = EntryMetadata(
        attribution=attribution if attribution is not None else existing.attribution,
        evidence=evidence if evidence is not None else existing.evidence,
        source_tool=source_tool if source_tool is not None else existing.source_tool,
        extracted_at=extracted_at if extracted_at is not None else existing.extracted_at,
        ingested_from=ingested_from if ingested_from is not None else existing.ingested_from,
        extra=existing.extra,
    )
    return resolved.to_dict()


def attribution_fields(
    metadata: object,
    *,
    fallback_agent_source: str,
    fallback_provider: str = "unknown",
    fallback_model: str = "unknown",
) -> tuple[str, str, str]:
    entry_metadata = EntryMetadata.from_value(metadata)
    attribution = entry_metadata.attribution or AttributionMetadata()
    agent_source = attribution.agent_source or fallback_agent_source
    provider = attribution.provider or fallback_provider
    model = attribution.model or fallback_model
    return agent_source, provider, model
