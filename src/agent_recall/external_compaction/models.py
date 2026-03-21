from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

_SUPPORTED_TIERS = {"GUARDRAILS", "STYLE", "RECENT"}


class ExternalNotesValidationError(ValueError):
    """Validation error with actionable details for external notes payloads."""

    def __init__(self, errors: list[str], *, example: dict[str, Any]) -> None:
        self.errors = errors
        self.example = example
        message = (
            "Invalid external notes payload. "
            f"{len(errors)} validation issue(s) found. "
            "Use the provided example shape."
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "invalid_external_notes_payload",
            "message": str(self),
            "errors": list(self.errors),
            "example": self.example,
        }


class ExternalCompactionNote(BaseModel):
    tier: str
    line: str
    source_session_ids: list[str] = Field(default_factory=list)

    @field_validator("tier")
    @classmethod
    def _validate_tier(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized not in _SUPPORTED_TIERS:
            raise ValueError("tier must be one of GUARDRAILS, STYLE, RECENT")
        return normalized

    @field_validator("line")
    @classmethod
    def _validate_line(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("line must be non-empty")
        if cleaned.startswith("#"):
            raise ValueError("line must be a content line, not a markdown header")
        return cleaned

    @field_validator("source_session_ids")
    @classmethod
    def _normalize_source_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return sorted(set(normalized))

    @model_validator(mode="after")
    def _validate_line_format(self) -> ExternalCompactionNote:
        if self.tier in {"GUARDRAILS", "STYLE"} and not self.line.startswith("- "):
            raise ValueError("GUARDRAILS/STYLE lines must start with '- '")
        if self.tier == "RECENT" and not (self.line.startswith("- ") or self.line.startswith("**")):
            raise ValueError("RECENT lines must start with '- ' or '**'")
        return self


class ExternalCompactionNotesPayload(BaseModel):
    notes: list[ExternalCompactionNote] = Field(default_factory=list)


class ExternalCompactionConversationEntry(BaseModel):
    id: str
    timestamp: str
    label: str
    confidence: float
    content: str
    tags: list[str] = Field(default_factory=list)
    source_session_id: str | None = None


class ExternalCompactionConversation(BaseModel):
    source_session_id: str
    entry_count: int
    entries: list[ExternalCompactionConversationEntry] = Field(default_factory=list)


class ExternalCompactionExportPayload(BaseModel):
    generated_at: datetime
    write_target: str
    tiers: dict[str, str]
    conversations: list[ExternalCompactionConversation]
    notes_schema: dict[str, Any]

    @field_validator("generated_at")
    @classmethod
    def _ensure_aware(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)


def external_notes_example_payload() -> dict[str, Any]:
    return {
        "notes": [
            {
                "tier": "GUARDRAILS",
                "line": "- [GOTCHA] Keep migration ordering deterministic.",
                "source_session_ids": ["source-session-id"],
            },
            {
                "tier": "STYLE",
                "line": "- [PATTERN] Prefer explicit transaction blocks around writes.",
                "source_session_ids": ["source-session-id"],
            },
        ]
    }


def external_notes_json_schema() -> dict[str, Any]:
    return ExternalCompactionNotesPayload.model_json_schema()


def validate_external_notes_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> ExternalCompactionNotesPayload:
    container: dict[str, Any]
    if isinstance(payload, list):
        container = {"notes": payload}
    elif isinstance(payload, dict):
        container = payload
    else:
        container = {"notes": []}

    try:
        return ExternalCompactionNotesPayload.model_validate(container)
    except ValidationError as exc:
        issues: list[str] = []
        for issue in exc.errors():
            loc = ".".join(str(part) for part in issue.get("loc", ()))
            msg = str(issue.get("msg", "invalid value"))
            issues.append(f"{loc}: {msg}")
        raise ExternalNotesValidationError(
            issues,
            example=external_notes_example_payload(),
        ) from exc
