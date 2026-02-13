## 2026-02-12T21:25:25Z Iteration 1 (AR-004)
- Define non-style indexing with explicit compaction config toggles (`index_decision_entries`, `index_exploration_entries`, `index_narrative_entries`) rather than implicit behavior.
- Keep decision/exploration/narrative additions scoped to chunk indexing; do not mix them into GUARDRAILS/STYLE synthesis prompts.
- Preserve duplicate protection (`indexed_entry_ids` + `has_chunk`) when expanding indexed label sets.

## 2026-02-12T21:30:45Z Iteration 2 (AR-004)
- Clamp non-style confidence thresholds to `[0.0, 1.0]` and fall back to defaults on invalid config values to avoid compaction crashes.
- Run lint before full validation when adding config-heavy helpers; long `dict.get(...)` lines can fail `ruff` even when tests pass.

## 2026-02-12T23:49:14Z HARD FAILURE Iteration 1 (AR-007)
- Item: Introduce optional shared team memory backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 1%]
  - tests/test_checkpoints.py ......... [ 6%]
  - tests/test_cli.py .............F.F.F.........FFFFF.F
  - _____________________ test_cli_sync_session_filters_wiring _____________________
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-12T23:53:00Z HARD FAILURE Iteration 1 (AR-007)
- Item: Introduce optional shared team memory backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - Traceback (most recent call last):
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/bin/pytest", line 10, in <module>
  - sys.exit(console_main())
  - ~~~~~~~~~~~~^^
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 223, in console_main
  - code = main()
- Primary actionable signal: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:05:49Z HARD FAILURE Iteration 1 (AR-007)
- Item: Introduce optional shared team memory backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 1%]
  - tests/test_checkpoints.py ......... [ 6%]
  - tests/test_cli.py ....F.F..FFF.
  - ____________ test_cli_retrieve_accepts_backend_and_tuning_overrides ____________
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:21:33Z HARD FAILURE Iteration 1 (AR-009)
- Item: Ship shared backend tracer-bullet (single-tenant)
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - Traceback (most recent call last):
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/lib/python3.14/site-packages/_pytest/main.py", line 316, in wrap_session
  - config.hook.pytest_sessionstart(session=session)
  - ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/lib/python3.14/site-packages/pluggy/_hooks.py", line 512, in __call__
  - return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
- Primary actionable signal: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T00:27:57Z Iteration 1 (AR-009)
- Clear `get_storage` cache during `init` before constructing storage; cached instances can point at new repo paths without schema initialization.
- Keep `storage.backend` defaulted to `local` until shared client operations are implemented; `shared` mode must fail fast with explicit guidance.
- When expanding config surface, update both `src/agent_recall/templates/config.yaml` and CLI `INITIAL_CONFIG` together to avoid drift.

## 2026-02-13T00:34:08Z Iteration 2 (AR-009)
- For new shared-backend slices, keep local default behavior unchanged and gate shared mode behind explicit `storage.backend: shared` + `storage.shared.base_url`.
- Treat `http(s)` shared URLs as a separate follow-up slice; fail fast with a clear message instead of partially emulating network behavior.
- Prove shared-state behavior with two storage instances pointing at the same shared URL, not with a single in-process instance.

## 2026-02-13T00:39:15Z Iteration 3 (AR-009)
- Keep `.agent/config.yaml` local even in shared mode; only synchronize tier files (`GUARDRAILS.md`, `STYLE.md`, `RECENT.md`) through shared storage paths.
- For filesystem shared backends, read tiers from the shared path first so one workstation does not serve stale local tier content.
- Do not silently emulate HTTP shared storage behavior in file-sync paths; keep HTTP transport deferred until implemented.

## 2026-02-13T01:13:29Z HARD FAILURE Iteration 1 (AR-009)
- Item: Ship shared backend tracer-bullet (single-tenant)
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - error: Failed to get PID of child process
  - Caused by: ESRCH: No such process
- Primary actionable signal: error: Failed to get PID of child process
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T01:18:54Z HARD FAILURE Iteration 1 (AR-009)
- Item: Ship shared backend tracer-bullet (single-tenant)
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 1%]
  - tests/test_checkpoints.py ..Traceback (most recent call last):
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/lib/python3.14/site-packages/_pytest/main.py", line 318, in wrap_session
  - session.exitstatus = doit(config, session) or 0
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T01:19:51Z HARD FAILURE Iteration 1 (AR-009)
- Item: Ship shared backend tracer-bullet (single-tenant)
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 1%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py ................................F.FFF.............. [ 32%]
  - tests/test_codex_fixtures.py ..................... [ 43%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T01:45:00Z Iteration 4 (AR-009)
- When testing CLI with `isolated_filesystem`, explicitly clear module-level caches like `get_storage.cache_clear()` between invocations to simulate fresh process state.
- `FileStorage` reads from shared tier if available but does *not* mirror to local on read; mirroring happens only on write. Tests should assert content in command output or shared file, not local file.


## 2026-02-13T01:46:58Z Iteration 5 (AR-009)
- When implementing retry loops for network/IO, catch specific exceptions (`sqlite3.OperationalError`, `OSError`) and use exponential backoff.
- Implement fallback logic silently if the goal is continuity, but ensure errors are raised if fallback also fails or isn't configured.
- Pass local DB paths to shared storage backends during initialization so they can self-configure fallback behavior.

## 2026-02-13T07:10:00Z Iteration 1 (AR-018)
- Keep command parity checks scoped to explicit surfaces in the command contract; do not require TUI-only commands to exist in CLI.

## 2026-02-13T08:00:00Z Iteration 2 (AR-018)
- When reporting command parity, surface both extra CLI and extra TUI commands so drift is visible from either side.

## 2026-02-13T09:12:00Z Iteration 3 (AR-018)
- Keep deprecated CLI aliases (like `config-llm`) out of the command contract so parity only tracks supported surfaces.

## 2026-02-13T09:45:00Z Iteration 4 (AR-015)
- When adding CLI features, update CLI reference docs and add at least one CLI regression test that asserts config persistence.

## 2026-02-13T02:30:00Z Iteration 6 (AR-009)
- Use `respx` to verify HTTP clients against mocked endpoints without requiring a running server in unit tests.
- Always use `response.raise_for_status()` in HTTP client methods to ensure errors propagate correctly.
- Verify that Pydantic models are instantiated with valid enum members (e.g. `SessionStatus.ACTIVE`) and required fields in tests.

## 2026-02-13T02:02:56Z HARD FAILURE Iteration 4 (AR-010)
- Item: Add tenant isolation and namespace safety
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py ................................................... [ 30%]
  - tests/test_codex_fixtures.py ..................... [ 40%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-4.log, agent_recall/ralph/.runtime/validate-4.log

## 2026-02-13T02:04:25Z HARD FAILURE Iteration 1 (AR-010)
- Item: Add tenant isolation and namespace safety
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py ................................................... [ 30%]
  - tests/test_codex_fixtures.py ..................... [ 40%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T02:04:38Z HARD FAILURE Iteration 2 (AR-010)
- Item: Add tenant isolation and namespace safety
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py .............F.FF...FFF.FFF.FFF.F.
  - ___________________________ test_cli_sync_no_compact ___________________________
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-2.log, agent_recall/ralph/.runtime/validate-2.log

## 2026-02-13T02:10:18Z HARD FAILURE Iteration 1 (AR-010)
- Item: Add tenant isolation and namespace safety
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py ................................................... [ 30%]
  - tests/test_codex_fixtures.py ..................... [ 40%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T02:10:38Z HARD FAILURE Iteration 2 (AR-010)
- Item: Add tenant isolation and namespace safety
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 5%]
  - tests/test_cli.py ........................F.F.FFFFFFFFF.............. [ 30%]
  - tests/test_codex_fixtures.py ..................... [ 40%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-2.log, agent_recall/ralph/.runtime/validate-2.log

## 2026-02-13T02:50:00Z Iteration 1 (AR-010)
- Shared storage backends must validate namespace configuration at initialization time; reject default tenant/project IDs to prevent accidental data mixing.
- Add `strict_namespace_validation` flag to SQLiteStorage for filesystem shared backends; validate before every write operation in strict mode.
- When updating storage interfaces, ensure all existing tests that create shared storage configs include explicit tenant_id and project_id to avoid NamespaceValidationError.
- Write negative assertions in leakage tests that explicitly assert data from other tenants/projects returns None/empty; document the isolation guarantees.

## 2026-02-13T03:30:00Z Iteration 1 (AR-011)
- When requiring shared-backend auth, fail fast during HTTP client initialization if the configured API key env var is missing.

## 2026-02-13T04:05:00Z Iteration 2 (AR-011)
- Enforce shared-backend RBAC client-side in the HTTP client before issuing write/promote/delete calls.
- Keep promotion gating scoped to chunk creation so log/session ingest flows stay usable for writers.

## 2026-02-13T02:46:21Z Iteration 1 (AR-016)
- Use iteration+item_id as the duplicate detection key for tier file entries rather than content hash similarity to allow same-item updates across iterations.
- Keep validation separate from write policy: deduplication is a write-time decision, while validation checks structural integrity (headers, malformed sections).
- Initialize tier files with canonical headers automatically when writing to empty files to ensure validation passes.
- Implement bounded append with entry-count limits (50-100 per tier) to prevent unbounded file growth during long-running loops.

## 2026-02-13T03:00:00Z Iteration 1 (AR-017)
- When implementing deduplication, normalize content for comparison but preserve original formatting in stored output to maintain readability.
- Apply size budgets by removing oldest entries first (based on timestamp), keeping the most recent content.
- Provide dry-run mode for destructive operations so users can preview changes before committing them.
- Ensure compaction operations are idempotent: running the same compaction twice should produce no changes on the second run.

## 2026-02-13T03:09:57Z Iteration 3 (AR-011)
- Scope item: Add auth, RBAC, and audit trail for shared backend
- Keep changes isolated and verifiable before commit.
- Runtime validation signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-3.log, agent_recall/ralph/.runtime/validate-3.log

## 2026-02-13T03:14:06Z HARD FAILURE Iteration 1 (AR-011)
- Item: Add auth, RBAC, and audit trail for shared backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 3%]
  - tests/test_cli.py ................................................... [ 21%]
  - tests/test_codex_fixtures.py ..................... [ 28%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T03:15:22Z HARD FAILURE Iteration 1 (AR-011)
- Item: Add auth, RBAC, and audit trail for shared backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - testpaths: tests
  - asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
  - tests/test_background_sync.py .. [ 0%]
  - tests/test_checkpoints.py ......... [ 3%]
  - tests/test_cli.py .................FF.F..FFFFFFFFFFFFFF.............. [ 21%]
  - tests/test_codex_fixtures.py ..................... [ 28%]
- Primary actionable signal: testpaths: tests
- Runtime logs: agent_recall/ralph/.runtime/agent-1.log, agent_recall/ralph/.runtime/validate-1.log

## 2026-02-13T05:12:08Z Iteration 3 (AR-011)
- When emitting audit events, keep it best-effort and default-on with a config toggle so shared write flows remain consistent in tests.

## 2026-02-13T06:00:00Z Iteration 4 (AR-007)
- Keep shared-storage docs explicit about opting in via `storage.backend: shared` and shared namespace keys (tenant/project).

## 2026-02-13T10:32:00Z Iteration 5 (AR-012)
- When adding new DB columns, update SQL placeholders in INSERT statements to match the new column count.

## 2026-02-13T04:18:42Z Iteration 5 (AR-015)
- When adding CLI config features, update model defaults, config templates, and init-time defaults together to avoid drift.

## 2026-02-13T04:24:01Z HARD FAILURE Iteration 5 (AR-007)
- Item: Introduce optional shared team memory backend
- Validation command: uv run pytest && uv run ruff check . && uv run ty check
- Top validation errors:
  - Traceback (most recent call last):
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/bin/pytest", line 10, in <module>
  - sys.exit(console_main())
  - ~~~~~~~~~~~~^^
  - File "/Users/natecard/OnHere/Repos/self-docs/.venv/lib/python3.14/site-packages/_pytest/config/__init__.py", line 223, in console_main
  - code = main()
- Primary actionable signal: Traceback (most recent call last):
- Runtime logs: agent_recall/ralph/.runtime/agent-5.log, agent_recall/ralph/.runtime/validate-5.log

## 2026-02-13T10:58:12Z Iteration 6 (AR-012)
- When adding new CLI command groups, update the command contract so parity reports do not flag extra CLI drift.

## 2026-02-13T12:00:00Z Iteration 1 (AR-012)
- Keep compaction curation filtering explicit and default to approved; allow pending-only via config to avoid promoting unreviewed items.

## 2026-02-13T12:30:00Z Iteration 2 (AR-012)
- When validating curation workflows, assert rejected entries never reach chunk indexing while approved ones do.

## 2026-02-13T18:15:00Z Iteration 3 (AR-013)
- When adding adapter payloads, gate writes behind config defaults or explicit CLI flags to avoid breaking existing flows.
