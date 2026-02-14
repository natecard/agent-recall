- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-4.log, agent_recall/ralph/.runtime/validate-4.log

## 2026-02-12T21:30:00Z Iteration 1 (AR-006)
- Use fcntl.flock with LOCK_EX | LOCK_NB for non-blocking exclusive file locks in Unix environments.
- Store background sync status in SQLite for persistence across process restarts.
- Use dataclasses for simple result objects (BackgroundSyncResult) and Pydantic models for storage entities.
- Keep CLI commands simple and focused; delegate complex logic to core modules.
- Use `os.getpid()` to track process ownership of locks for stale detection.

## 2026-02-12T20:40:06Z Iteration 2 (AR-006)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-2.log, agent_recall/ralph/.runtime/validate-2.log

## 2026-02-12T20:43:49Z Iteration 1 (AR-006)
- For context export commands, emit both markdown and JSON artifacts for human and tool consumers.
- Resolve refresh task in order: explicit `--task`, then active session task fallback.

## 2026-02-12T20:51:11Z Iteration 2 (AR-006)
- Model retry policy as explicit manager fields (`retry_attempts`, `retry_backoff_seconds`) with bounded minimums.
- Return structured retry diagnostics in result dataclasses so CLI can render actionable failure context.
- For CLI retry UX, keep success output compact and add retry-count lines only when retries actually occurred.

## 2026-02-12T20:53:59Z Iteration 1 (AR-003)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-12T20:59:22Z Iteration 1 (AR-003)
- Put reusable retrieval math in a focused core utility module (`core/embeddings.py`) rather than embedding logic directly in CLI/storage layers.
- Keep SQLite serialization/deserialization logic in small static helpers on `SQLiteStorage` for type-safe chunk mapping.
- Cover tracer-bullet retrieval changes with one generation-path test (`compact`) and one persistence round-trip test (`retrieve`).

## 2026-02-12T21:04:50Z Iteration 2 (AR-003)
- Implement hybrid retrieval as a small `Retriever` slice (`_search_hybrid` + `_rank_vector_candidates`) without widening CLI surface area in the same commit.
- Use reciprocal-rank fusion (`1 / (fusion_k + rank)`) to combine FTS and vector channels with deterministic sorting.
- Add retrieval tests that pin behavior with explicit UUIDs for tie cases and a vector-only match fixture.

## 2026-02-12T21:10:09Z Iteration 1 (AR-003)
- Add retrieval enhancements as optional `Retriever.search(...)` flags first, then wire CLI/config in a separate slice.
- Expand retrieval candidate depth only when reranking is enabled; keep default `top_k` behavior unchanged.
- Use deterministic rerank sort keys in order: combined score, similarity, lexical score, original rank, chunk id.

## 2026-02-12T21:18:29Z Iteration 2 (AR-003)
- Centralize retrieval runtime resolution in a helper (`_build_retriever`) so config defaults and CLI overrides stay aligned.
- Pass an explicit `Retriever` into `ContextAssembler` instead of hardcoding direct FTS calls in context assembly.
- Use monkeypatched `Retriever` objects in CLI tests to verify wiring (backend/top_k/fusion/rerank) deterministically.

## 2026-02-12T21:25:25Z Iteration 1 (AR-004)
- Add label-selection behavior via a small helper (`_resolve_non_style_index_labels`) and feed it into the existing indexing loop.
- Keep tracer-bullet tests focused: assert labels that must be indexed and labels that must remain excluded by default.
- Ship config-backed defaults in `templates/config.yaml` whenever compaction behavior changes.

## 2026-02-12T21:30:45Z Iteration 2 (AR-004)
- Add anti-noise compaction controls as narrow helpers (`_resolve_non_style_index_thresholds`, `_filter_non_style_index_entries`) rather than widening the main compaction loop.
- For threshold behavior, write one focused compaction test that asserts both promoted and filtered entries so chunk growth constraints stay deterministic.

## 2026-02-12T23:49:14Z Iteration 1 (AR-007)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-12T23:53:00Z Iteration 1 (AR-007)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:05:49Z Iteration 1 (AR-007)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:21:33Z Iteration 1 (AR-009)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:27:57Z Iteration 1 (AR-009)
- Define backend contracts with nested config models (`storage.shared`) so local/shared settings stay explicit and typed.
- Cover backend selection slices with focused factory tests: local backend success path + shared backend explicit-not-implemented path.
- In stub backends, route unsupported methods through one `NoReturn` helper to keep behavior and type checking consistent.

## 2026-02-13T00:34:08Z Iteration 2 (AR-009)
- For tracer-bullet backend implementations, prefer subclassing the stable storage backend (`SQLiteStorage`) over duplicating method-by-method adapter code.
- Resolve shared backend targets through a focused URL-to-path helper to keep backend selection logic testable.
- Add one focused factory-routing test and one cross-instance state test when changing storage backend wiring.

## 2026-02-13T00:39:15Z Iteration 3 (AR-009)
- Extract shared URL parsing into a reusable helper (`resolve_shared_db_path`) so storage factory and CLI tier-sync wiring use one resolution path.
- Extend `FileStorage` with an optional `shared_tiers_dir` mirror instead of changing config-file root behavior.
- Keep shared tier sync behavior deterministic: writes mirror to local+shared, reads prefer shared then fall back to local.

## 2026-02-13T01:13:29Z Iteration 1 (AR-009)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: error: Failed to get PID of child process
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T01:18:54Z Iteration 1 (AR-009)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T01:45:00Z Iteration 4 (AR-009)
- Use `e2e` tests to validate complex multi-repo workflows like shared storage, using `isolated_filesystem` and subdirectories for distinct "repositories".
- When mocking LLM for compact/context tests, ensure prompt matching covers all expected tiers (GUARDRAILS, STYLE, RECENT) to avoid empty updates.


## 2026-02-13T01:46:58Z Iteration 5 (AR-009)
- When modifying storage backends, centralize retry/fallback logic in a single `_execute` wrapper rather than duplicating it across all methods.
- Use `unittest.mock.patch` context managers for mocking dependencies in tests to ensure clean teardown.
- Avoid broad `except Exception` blocks in production code unless specifically implementing a "last resort" fallback.

## 2026-02-13T00:20:00Z Iteration 7 (AR-303)
- For bash-delegated CLI commands, surface script/prd resolution errors before running subprocess and always return the subprocess exit code.

## 2026-02-13T23:30:00Z Iteration 4 (AR-603)
- Use parse helpers (parse_tier_content) to preserve multi-line Ralph blocks in compaction rewrite paths.

## 2026-02-13T07:10:00Z Iteration 1 (AR-018)
- Define command contract data in a dedicated module and reuse it for help text, suggestions, and parity reporting.

## 2026-02-13T08:00:00Z Iteration 2 (AR-018)
- Extend parity helpers to report extra CLI commands alongside missing/extra TUI entries.

## 2026-02-13T09:12:00Z Iteration 3 (AR-018)
- Treat top-level CLI command groups (e.g. `config`, `theme`, `ralph`) as contract entries to avoid false extra-in-CLI drift.

## 2026-02-13T09:45:00Z Iteration 4 (AR-015)
- Keep CLI config updates verified by reading `.agent/config.yaml` in tests after invoking the command.

## 2026-02-13T02:30:00Z Iteration 6 (AR-009)
- Map `Storage` interface methods to flat, noun-based REST endpoints (e.g. `POST /sessions`, `GET /sessions/{id}`) for clarity.
- Use query parameters (`GET /entries?labels=...`) for optional filters instead of complex path structures.
- Rely on Pydantic's `model_dump_json()` and `model_validate()` for reliable serialization across the HTTP boundary.

## 2026-02-13T03:30:00Z Iteration 1 (AR-011)
- Add auth enforcement toggles as config fields with defaults and cover them with a focused HTTP client test that asserts the Authorization header.

## 2026-02-13T04:05:00Z Iteration 2 (AR-011)
- Keep RBAC checks as small HTTP client helpers (e.g. `_require_role`) rather than scattering conditional logic per call.
- Add focused allow/deny tests alongside existing HTTP client tests to pin RBAC behavior.
- Keep token budget trimming deterministic and byte-based (chars/4) to avoid external tokenizer drift.

## 2026-02-13T05:12:08Z Iteration 3 (AR-011)
- Model audit events as Pydantic payloads and reuse JSON serialization helpers for HTTP posts.

## 2026-02-13T21:45:00Z Iteration 6 (AR-206)
- Keep bash loop hook helpers small and side-effect isolated; capture output to .runtime and only print warnings on failure.

## 2026-02-13T23:59:00Z Iteration 1 (AR-501)
- Use StrEnum for user-facing enum values to keep CLI/JSON output stable and readable.

## 2026-02-13T23:59:00Z Iteration 2 (AR-502)
- Use dataclasses with default_factory lists for parsed tier content containers to avoid shared mutable defaults.

## 2026-02-13T02:05:00Z Iteration 1 (AR-602)
- Keep Ralph-block skipping logic isolated in the tier line splitter so merge/write paths stay unchanged.

## 2026-02-13T10:05:00Z Iteration 2 (AR-601)
- Prefer verifying existing code with direct grep/reads before editing; mark PRD passes when acceptance criteria are already satisfied.

## 2026-02-13T22:25:00Z Iteration 3 (AR-602)
- Confirm PRD acceptance by reading the code path end-to-end before marking passes.

## 2026-02-13T02:02:56Z Iteration 4 (AR-010)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-4.log, agent_recall/ralph/.runtime/validate-4.log

## 2026-02-13T02:04:25Z Iteration 1 (AR-010)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T02:04:38Z Iteration 2 (AR-010)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-2.log, agent_recall/ralph/.runtime/validate-2.log

## 2026-02-13T02:10:18Z Iteration 1 (AR-010)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T02:50:00Z Iteration 1 (AR-010)
- Define validation functions as pure helpers (e.g. `validate_shared_namespace()`) that raise specific exceptions; makes testing and reuse straightforward.
- Add `strict_namespace_validation` parameter to storage backends to toggle validation at runtime; keeps local mode permissive while enforcing shared mode constraints.
- Pass tenant/project from config through storage factory to backends; avoid hardcoding defaults in multiple places.
- For security-critical features like isolation, write negative test assertions that prove data does NOT cross boundaries (e.g., `assert tenant_b.get_session(id) is None`).

## 2026-02-13T02:46:21Z Iteration 1 (AR-016)
- Implement tier file operations as a dedicated module (`tier_writer`) with clear separation between schemas, policies, and operations.
- Use dataclasses for policy configuration (`WritePolicy`) to make behavior explicit and configurable per-write.
- Provide both high-level convenience methods (e.g., `write_guardrails_entry`) and low-level primitives (`_bounded_append`, `_replace_section`) for flexibility.
- Add CLI commands that mirror the programmatic API for scripting and Ralph loop integration.
- Include comprehensive statistics and linting commands for operational visibility into tier file health.

## 2026-02-13T03:00:00Z Iteration 1 (AR-017)
- Implement post-loop compaction as a separate hook (`TierCompactionHook`) from the main compaction logic to keep concerns separated.
- Use iteration+item_id as the deduplication key for tier entries rather than content hash to allow same-item updates across iterations.
- Provide both automatic (config-driven) and manual (CLI/TUI) compaction paths that produce identical results for parity.
- Include dry-run mode in CLI commands so users can preview changes before committing them.
- Test idempotency explicitly: running compaction twice on the same data should produce no changes the second time.

## 2026-02-13T03:09:57Z Iteration 3 (AR-011)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-3.log, agent_recall/ralph/.runtime/validate-3.log

## 2026-02-13T03:14:06Z Iteration 1 (AR-011)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T03:15:22Z Iteration 1 (AR-011)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T06:00:00Z Iteration 4 (AR-007)
- Document shared storage with a compact config snippet plus a migration checklist for first-time enablement.

## 2026-02-13T10:32:00Z Iteration 5 (AR-012)
- When extending SQLite schemas, add migration logic and update row-to-model mapping in the same change.

## 2026-02-13T04:18:42Z Iteration 5 (AR-015)
- Add CLI config commands as small helpers for reading/writing config sections so status and updates stay consistent across CLI/TUI.

## 2026-02-13T10:58:12Z Iteration 6 (AR-012)
- For curation workflows, reuse existing storage APIs and add narrow list/update methods to avoid spreading SQL across CLI.

## 2026-02-13T12:00:00Z Iteration 1 (AR-012)
- Resolve curation status from config via a small helper that normalizes and defaults on invalid values.

## 2026-02-13T12:30:00Z Iteration 2 (AR-012)
- Add curation lifecycle assertions in storage-backed tests to cover pending -> rejected -> approved transitions.

## 2026-02-13T20:40:00Z Iteration 5 (AR-013)
- Format adapter status lines via small helpers to keep CLI/status rendering consistent.

## 2026-02-13T18:15:00Z Iteration 3 (AR-013)
- Add adapter payload outputs under the existing context bundle directory to keep CLI/TUI expectations aligned.

## 2026-02-13T20:55:00Z Iteration 6 (AR-013)
- When testing adapter payload behavior, assert on the payload file count and avoid relying on file mtimes for idempotency checks.

## 2026-02-13T21:20:00Z Iteration 1 (AR-201)
- Use dataclasses for archive record models and keep datetime serialization ISO-8601 with UTC normalization.

## 2026-02-13T22:15:00Z Iteration 2 (AR-202)
- Keep PRD archive search encapsulated in PRDArchive (generate embeddings + cosine similarity) so CLI wiring stays thin.

## 2026-02-13T22:45:00Z Iteration 3 (AR-203)
- Format Ralph refresh task strings in one place (hook) to keep CLI/batch callers consistent.
- Keep CLI action routing consolidated in the existing command handler when adding Ralph sub-actions.
## 2026-02-13T23:55:00Z Iteration 5 (AR-204)
- Keep Ralph CLI outputs aligned with existing Rich patterns (Panel/Table/bullets) to match other CLI surfaces.

## 2026-02-13T18:31:00Z Iteration 5 (AR-301)
- Implement Ralph enable/disable using a small state JSON writer/reader to avoid adding unused heavy loop machinery.

## 2026-02-13T00:00:00Z Iteration 6 (AR-301)
- Prefer Typer subapps for command groups while keeping main CLI helper wrappers for test stability.

## 2026-02-13T23:59:59Z Iteration 8 (AR-302)
- For dual-mode CLI status views, split rendering into focused helpers and reuse shared truncation utilities.

## 2026-02-13T23:59:59Z Iteration 9 (AR-303)
- Keep progress callbacks in CLI narrowly focused on rendering while loop logic owns payload structure.

## 2026-02-14T04:37:49Z Iteration 1 (WM-001)
- Prefer one logical change per commit.
- Run local checks before commit when available.
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-14T05:10:00Z Iteration 1 (WM-001)
- Use ISO-8601 UTC serialization for datetime fields in report JSON models.

## 2026-02-14T04:47:22Z Iteration 2 (WM-002)
- Prefer one logical change per commit.
- Keep validation command green before committing: uv run pytest && uv run ruff check . && uv run ty check
- Start debugging from the first actionable validation line: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-2.log, agent_recall/ralph/.runtime/validate-2.log

## 2026-02-14T06:10:00Z Iteration 1 (WM-002)
- Use unittest.mock patching to keep subprocess-based helpers deterministic in tests.

## 2026-02-14T09:00:00Z Iteration 1 (WM-003)
- Keep heuristic reports ordered newest-first in storage and reverse for oldest-to-newest display.

## 2026-02-14T12:15:00Z Iteration 2 (WM-006)
- Gate uv-driven loop steps behind `command -v uv` and keep failures non-fatal with warnings.

## 2026-02-14T13:20:00Z Iteration 3 (WM-007)
- Keep prompt templates parameterized with item_id, item_title, description, and validation_command placeholders.
