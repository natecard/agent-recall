from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from agent_recall.storage.sqlite import SQLiteStorage

_JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]


def create_shared_backend_wsgi_app(
    db_path: Path,
    *,
    default_tenant_id: str = "default",
    default_project_id: str = "default",
    bearer_token: str | None = None,
    max_limit: int = 2000,
):
    """Build a minimal WSGI app for shared-backend HTTP storage endpoints.

    This server currently exposes:
    - GET /entries/by-source-session?source_session_id=<id>&limit=<n>
    """

    resolved_db_path = Path(db_path).expanduser()
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    safe_max_limit = max(1, int(max_limit))

    def _json_response(
        start_response,
        status: str,
        payload: dict[str, Any] | list[dict[str, Any]],
    ) -> list[bytes]:
        body = json.dumps(payload, indent=2).encode("utf-8")
        headers = list(_JSON_HEADERS)
        headers.append(("Content-Length", str(len(body))))
        start_response(status, headers)
        return [body]

    def _header(environ: dict[str, Any], name: str) -> str:
        key = f"HTTP_{name.upper().replace('-', '_')}"
        return str(environ.get(key, "")).strip()

    def _resolve_scope(environ: dict[str, Any]) -> tuple[str, str]:
        tenant_id = _header(environ, "X-Tenant-ID") or default_tenant_id
        project_id = _header(environ, "X-Project-ID") or default_project_id
        return tenant_id, project_id

    def _is_authorized(environ: dict[str, Any]) -> bool:
        if not bearer_token:
            return True
        authorization = _header(environ, "Authorization")
        return authorization == f"Bearer {bearer_token}"

    def app(environ: dict[str, Any], start_response):
        method = str(environ.get("REQUEST_METHOD", "")).strip().upper()
        path = str(environ.get("PATH_INFO", "")).strip()

        if method == "GET" and path == "/entries/by-source-session":
            if not _is_authorized(environ):
                return _json_response(
                    start_response,
                    "401 Unauthorized",
                    {
                        "error": "unauthorized",
                        "message": "Missing or invalid bearer token.",
                    },
                )

            query = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=True)
            source_session_id = str(query.get("source_session_id", [""])[0]).strip()
            if not source_session_id:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {
                        "error": "invalid_request",
                        "message": "Query parameter 'source_session_id' is required.",
                    },
                )

            raw_limit = str(query.get("limit", ["200"])[0]).strip()
            try:
                limit = int(raw_limit)
            except ValueError:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {
                        "error": "invalid_request",
                        "message": "Query parameter 'limit' must be an integer.",
                    },
                )
            limit = max(1, min(limit, safe_max_limit))

            tenant_id, project_id = _resolve_scope(environ)
            scoped_storage = SQLiteStorage(
                resolved_db_path,
                tenant_id=tenant_id,
                project_id=project_id,
                strict_namespace_validation=False,
            )
            entries = scoped_storage.get_entries_by_source_session(source_session_id, limit=limit)
            if not entries:
                return _json_response(
                    start_response,
                    "404 Not Found",
                    {
                        "error": "not_found",
                        "message": "No entries found for source_session_id in current scope.",
                    },
                )

            payload = [entry.model_dump(mode="json") for entry in entries]
            return _json_response(start_response, "200 OK", payload)

        return _json_response(
            start_response,
            "404 Not Found",
            {"error": "not_found", "message": "Endpoint not found."},
        )

    return app
