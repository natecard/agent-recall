from __future__ import annotations

import json
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_recall.core.config import load_config
from agent_recall.core.semantic_embedder import configure_from_memory_config
from agent_recall.memory.migration import (
    build_embedding_provider_from_memory_config,
    collect_learning_rows,
)
from agent_recall.memory.vector_store import (
    LocalVectorStore,
    TurboPufferVectorStore,
    VectorRecord,
    resolve_local_vector_db_path,
)
from agent_recall.storage.base import Storage
from agent_recall.storage.files import FileStorage

AGENT_MEMORY_BUNDLE_FILENAME = "agent-memory.json"
MEMORY_REQUEST_ENVELOPE_KEY = "agent_recall_memory_request"
MEMORY_RESPONSE_ENVELOPE_KEY = "agent_recall_memory_response"


class MemoryCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    label: str
    tags: list[str] = Field(default_factory=list)
    score: float | None = None
    source: str | None = None
    source_session_id: str | None = None
    timestamp: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "search",
        "fetch",
        "promote",
        "list_working_set",
        "clear_working_set",
        "trim_working_set",
    ]
    query: str | None = None
    ids: list[str] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=5, ge=1, le=20)
    limit: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _validate_shape(self) -> MemoryRequestPayload:
        if self.action == "search" and not (self.query or "").strip():
            raise ValueError("search requests require a non-empty query")
        if self.action in {"fetch", "promote"} and not self.ids:
            raise ValueError(f"{self.action} requests require one or more ids")
        if self.action == "trim_working_set" and self.limit is None:
            raise ValueError("trim_working_set requests require a limit")
        return self


class MemoryRequestEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_recall_memory_request: MemoryRequestPayload


def _normalize_metadata_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _optional_float(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    return float(value)


def _memory_card_from_row(row: dict[str, Any], *, score: float | None = None) -> MemoryCard:
    metadata_dict = _normalize_metadata_mapping(row.get("metadata"))
    return MemoryCard(
        id=str(row.get("id", "")),
        text=str(row.get("text", "")),
        label=str(row.get("label", "unknown")),
        tags=[str(tag) for tag in row.get("tags", []) if str(tag).strip()],
        score=score,
        source=(
            str(metadata_dict.get("source"))
            if isinstance(metadata_dict.get("source"), str)
            else None
        ),
        source_session_id=(
            str(metadata_dict.get("source_session_id"))
            if isinstance(metadata_dict.get("source_session_id"), str)
            else None
        ),
        timestamp=(
            str(metadata_dict.get("timestamp"))
            if isinstance(metadata_dict.get("timestamp"), str)
            else None
        ),
        confidence=_optional_float(metadata_dict.get("confidence")),
        metadata=metadata_dict,
    )


def _memory_card_from_vector_record(
    record: VectorRecord,
    *,
    score: float | None = None,
) -> MemoryCard:
    return _memory_card_from_row(record.model_dump(mode="python"), score=score)


def _memory_card_from_remote_match(match: dict[str, Any]) -> MemoryCard:
    metadata_dict = _normalize_metadata_mapping(match.get("metadata"))
    return MemoryCard(
        id=str(match.get("id", "")),
        text=str(match.get("text", "")),
        label=str(match.get("label", "unknown")),
        tags=[str(tag) for tag in match.get("tags", []) if str(tag).strip()],
        score=(
            float(match.get("score", 0.0)) if isinstance(match.get("score"), int | float) else 0.0
        ),
        source=(
            str(metadata_dict.get("source"))
            if isinstance(metadata_dict.get("source"), str)
            else None
        ),
        source_session_id=(
            str(metadata_dict.get("source_session_id"))
            if isinstance(metadata_dict.get("source_session_id"), str)
            else None
        ),
        timestamp=(
            str(metadata_dict.get("timestamp"))
            if isinstance(metadata_dict.get("timestamp"), str)
            else None
        ),
        confidence=_optional_float(metadata_dict.get("confidence")),
        metadata=metadata_dict,
    )


def write_agent_memory_bundle(output_dir: Path, bundle: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / AGENT_MEMORY_BUNDLE_FILENAME
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def load_agent_memory_bundle(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        return {}
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def memory_protocol_contract() -> dict[str, Any]:
    return {
        "request_key": MEMORY_REQUEST_ENVELOPE_KEY,
        "response_key": MEMORY_RESPONSE_ENVELOPE_KEY,
        "strict_json_only": True,
        "supported_actions": [
            "search",
            "fetch",
            "promote",
            "list_working_set",
            "clear_working_set",
            "trim_working_set",
        ],
        "request_schema": MemoryRequestEnvelope.model_json_schema(),
        "instructions": (
            "If you need more project memory, respond with exactly one JSON object whose top-level "
            f'key is "{MEMORY_REQUEST_ENVELOPE_KEY}". Do not include prose, markdown fences, '
            "or any other text in a memory request."
        ),
    }


def memory_request_instructions(bundle_path: Path | None = None) -> str:
    location = f"Starter bundle path: {bundle_path}" if bundle_path is not None else ""
    lines = [
        "If you need more project memory, reply with ONLY a strict JSON request object.\n",
        f'Top-level key: "{MEMORY_REQUEST_ENVELOPE_KEY}"\n',
        (
            "Supported actions: search, fetch, promote, list_working_set, "
            "clear_working_set, trim_working_set.\n"
        ),
        "If you are not making a memory request, continue normally.\n",
    ]
    if location:
        lines.append(location)
    return "".join(lines).strip()


def render_agent_memory_prompt(bundle: dict[str, Any]) -> str:
    long_term = bundle.get("long_term_memory")
    working_set = bundle.get("working_set")
    lines = [
        "## Agent Recall Structured Memory",
        "",
        f"Memory status: {bundle.get('memory_status', {}).get('status', 'unknown')}",
    ]
    if isinstance(working_set, list) and working_set:
        lines.extend(["", "### Task Working Set"])
        for item in working_set[:8]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "memory"))
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"- [{label}] {text}")
    if isinstance(long_term, list) and long_term:
        lines.extend(["", "### Long-Range Memory Hits"])
        for item in long_term[:8]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "memory"))
            text = str(item.get("text", "")).strip()
            score = item.get("score")
            suffix = f" (score={float(score):.3f})" if isinstance(score, int | float) else ""
            if not text:
                continue
            lines.append(f"- [{label}] {text}{suffix}")
    lines.extend(
        [
            "",
            "### Live Memory Protocol",
            memory_request_instructions(),
        ]
    )
    return "\n".join(lines)


class AgentMemoryIndex:
    def __init__(self, *, storage: Storage, files: FileStorage) -> None:
        self.storage = storage
        self.files = files

    def status(self) -> dict[str, Any]:
        config = load_config(self.files.agent_dir)
        memory_cfg = config.memory.model_dump(mode="python")
        return {
            "enabled": bool(memory_cfg.get("vector_enabled", False)),
            "backend": str(memory_cfg.get("vector_backend", "local")).strip().lower(),
            "embedding_provider": (
                str(memory_cfg.get("embedding_provider", "local")).strip().lower()
            ),
            "embedding_dimensions": int(config.retrieval.embedding_dimensions),
            "tenant_id": config.storage.shared.tenant_id,
            "project_id": config.storage.shared.project_id,
            "memory_cfg": memory_cfg,
        }

    def recent_memories(self, *, limit: int = 3) -> list[MemoryCard]:
        return [
            _memory_card_from_row(row)
            for row in collect_learning_rows(self.storage)[: max(1, int(limit))]
        ]

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryCard]:
        runtime = self.status()
        if not runtime["enabled"]:
            return []

        memory_cfg = runtime["memory_cfg"]
        if (
            str(runtime["embedding_provider"]) == "local"
            and int(runtime["embedding_dimensions"]) == 384
        ):
            configure_from_memory_config(memory_cfg)
        _provider_name, provider = build_embedding_provider_from_memory_config(
            memory_cfg,
            embedding_dimensions=int(runtime["embedding_dimensions"]),
        )
        query_embedding = provider.embed_texts([query]).vectors[0]
        backend = str(runtime["backend"])
        if backend == "turbopuffer":
            turbopuffer_cfg = memory_cfg.get("turbopuffer")
            config = turbopuffer_cfg if isinstance(turbopuffer_cfg, dict) else {}
            store = TurboPufferVectorStore(
                base_url=str(config.get("base_url") or "").strip(),
                api_key_env=str(config.get("api_key_env", "TURBOPUFFER_API_KEY")),
                tenant_id=str(runtime["tenant_id"]),
                project_id=str(runtime["project_id"]),
                timeout_seconds=float(config.get("timeout_seconds", 10.0)),
                retry_attempts=int(config.get("retry_attempts", 2)),
            )
            return [
                _memory_card_from_remote_match(match)
                for match in store.query(embedding=query_embedding, top_k=top_k)
            ]

        store = LocalVectorStore(
            resolve_local_vector_db_path(self.files.agent_dir),
            tenant_id=str(runtime["tenant_id"]),
            project_id=str(runtime["project_id"]),
        )
        return [
            _memory_card_from_vector_record(record, score=score)
            for record, score in store.query(embedding=query_embedding, top_k=top_k)
        ]


def build_agent_memory_bundle(
    *,
    storage: Storage,
    files: FileStorage,
    task: str | None,
    active_session_id: str | None,
    repo_path: Path,
    refreshed_at: datetime,
    top_k: int = 5,
) -> dict[str, Any]:
    index = AgentMemoryIndex(storage=storage, files=files)
    status = index.status()
    recent = [card.model_dump(mode="json") for card in index.recent_memories(limit=3)]
    long_term: list[dict[str, Any]] = []
    working_set = list(recent)
    memory_status = {
        "status": "disabled" if not status["enabled"] else "ready",
        "enabled": bool(status["enabled"]),
        "backend": status["backend"],
        "embedding_provider": status["embedding_provider"],
    }
    if task and status["enabled"]:
        try:
            long_term_cards = index.search(task, top_k=top_k)
            long_term = [card.model_dump(mode="json") for card in long_term_cards]
            seen_ids = {str(item.get("id", "")) for item in working_set if isinstance(item, dict)}
            for card in long_term[: min(3, len(long_term))]:
                card_id = str(card.get("id", ""))
                if card_id and card_id not in seen_ids:
                    working_set.append(card)
                    seen_ids.add(card_id)
        except Exception as exc:  # noqa: BLE001
            memory_status = {
                **memory_status,
                "status": "degraded",
                "error": str(exc),
            }
    return {
        "format_version": 1,
        "task": task,
        "active_session_id": active_session_id,
        "repo_path": str(repo_path),
        "refreshed_at": (
            refreshed_at.replace(tzinfo=UTC)
            if refreshed_at.tzinfo is None
            else refreshed_at.astimezone(UTC)
        ).isoformat(),
        "memory_status": memory_status,
        "long_term_memory": long_term,
        "working_set": working_set,
        "memory_protocol": memory_protocol_contract(),
    }


def try_parse_memory_request(raw: str) -> MemoryRequestPayload | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or MEMORY_REQUEST_ENVELOPE_KEY not in payload:
        return None
    try:
        parsed = MemoryRequestEnvelope.model_validate(payload)
    except ValidationError:
        return None
    return parsed.agent_recall_memory_request


class AgentMemoryBroker:
    def __init__(
        self,
        *,
        storage: Storage,
        files: FileStorage,
        bundle: dict[str, Any],
    ) -> None:
        self.storage = storage
        self.files = files
        self.bundle = bundle
        self.index = AgentMemoryIndex(storage=storage, files=files)
        self.known_cards: dict[str, MemoryCard] = {}
        self.working_set: OrderedDict[str, MemoryCard] = OrderedDict()
        for group_name in ("long_term_memory", "working_set"):
            group = bundle.get(group_name)
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                try:
                    card = MemoryCard.model_validate(item)
                except ValidationError:
                    continue
                self.known_cards[card.id] = card
                if group_name == "working_set":
                    self.working_set[card.id] = card

    def handle_request(self, request: MemoryRequestPayload) -> dict[str, Any]:
        try:
            if request.action == "search":
                results = self.index.search(str(request.query or ""), top_k=request.top_k)
                for result in results:
                    self.known_cards[result.id] = result
                return self._response(
                    action="search",
                    results=[item.model_dump(mode="json") for item in results],
                )
            if request.action == "fetch":
                found, missing = self._find_cards(request.ids)
                return self._response(
                    action="fetch",
                    results=[item.model_dump(mode="json") for item in found],
                    missing_ids=missing,
                )
            if request.action == "promote":
                found, missing = self._find_cards(request.ids)
                for item in found:
                    self.working_set[item.id] = item
                return self._response(
                    action="promote",
                    results=[item.model_dump(mode="json") for item in found],
                    missing_ids=missing,
                    working_set=self._working_set_payload(),
                )
            if request.action == "list_working_set":
                return self._response(
                    action="list_working_set",
                    working_set=self._working_set_payload(),
                )
            if request.action == "clear_working_set":
                self.working_set.clear()
                return self._response(action="clear_working_set", working_set=[])
            if request.action == "trim_working_set":
                limit = int(request.limit or 1)
                while len(self.working_set) > limit:
                    self.working_set.popitem(last=False)
                return self._response(
                    action="trim_working_set",
                    working_set=self._working_set_payload(),
                )
        except Exception as exc:  # noqa: BLE001
            return self._response(action=request.action, status="error", error=str(exc))
        return self._response(action=request.action, status="error", error="Unsupported action")

    def handle_raw_request(
        self,
        raw: str,
    ) -> tuple[MemoryRequestPayload | None, dict[str, Any] | None]:
        request = try_parse_memory_request(raw)
        if request is None:
            return None, None
        return request, self.handle_request(request)

    def response_json(self, response: dict[str, Any]) -> str:
        return json.dumps(response, indent=2)

    def _working_set_payload(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.working_set.values()]

    def _find_cards(self, ids: list[str]) -> tuple[list[MemoryCard], list[str]]:
        found: list[MemoryCard] = []
        missing: list[str] = []
        for value in ids:
            card = self.known_cards.get(str(value))
            if card is None:
                missing.append(str(value))
                continue
            found.append(card)
        return found, missing

    def _response(
        self,
        *,
        action: str,
        status: str = "ok",
        error: str | None = None,
        results: list[dict[str, Any]] | None = None,
        working_set: list[dict[str, Any]] | None = None,
        missing_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": action,
            "status": status,
            "working_set": working_set if working_set is not None else self._working_set_payload(),
        }
        if results is not None:
            payload["results"] = results
        if missing_ids:
            payload["missing_ids"] = missing_ids
        if error:
            payload["error"] = error
        return {MEMORY_RESPONSE_ENVELOPE_KEY: payload}
