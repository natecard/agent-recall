# Shared Backend API Contract: `GET /entries/by-source-session`

Endpoint:
- `GET /entries/by-source-session`

Purpose:
- Return log entries for one imported `source_session_id` in the active tenant/project scope.

## Request

Query parameters:
- `source_session_id` (required): source session identifier.
- `limit` (optional, integer): max records to return. Defaults to `200`, clamped to a server max.

Headers:
- `X-Tenant-ID`: tenant namespace (required in shared deployments).
- `X-Project-ID`: project namespace (required in shared deployments).
- `Authorization: Bearer <token>`: required when server auth is enabled.

## Response

Success:
- `200 OK`
- Body: JSON array of `LogEntry` objects.

Not found:
- `404 Not Found` when no entries exist for the scoped `source_session_id`.

Validation/auth errors:
- `400 Bad Request` for missing/invalid query params.
- `401 Unauthorized` for missing/invalid bearer token when auth is enabled.

## Ordering and paging semantics

- Sort order: ascending by `timestamp` (oldest to newest).
- Paging: limit-based single-page retrieval (`limit`). No cursor/offset in this contract revision.

## Client normalization

The SDK/client normalizes:
- `404` to an empty list (`[]`).
- `204` to an empty list (`[]`) when returned by compatible backends.
