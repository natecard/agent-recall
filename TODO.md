# Agent Recall Product TODO

This document converts the prioritized backlog into concrete milestones and ticket-sized tasks.

## Milestone Plan

| Milestone | Focus | Priorities Covered | Exit Criteria |
|---|---|---|---|
| M1 | Safety and API parity foundations | 1, 4, 5, 9, 18 | External compaction is safe by default, API contracts are stable, UX is clear |
| M2 | External compaction production readiness | 3, 6, 7, 8 | Queue/review flow works end-to-end with full automated coverage |
| M3 | Vector backend architecture + adapters | 2 (stages 1-2) | Dual-write and provider adapters are functional behind flags |
| M4 | Hybrid/vector read path migration | 2 (stages 3-4), 12 | Hybrid retrieval and vector-primary mode are production-usable |
| M5 | Governance and observability | 10, 11, 15, 17 | Policy enforcement and telemetry are in place with measurable outcomes |
| M6 | Advanced productivity features | 13, 14, 16 | PR-aware context, topic threads, and memory packs are usable and documented |

## Ticket Backlog

### Priority 1 - Remote API parity for external compaction (M1)
- [x] AR-101 (S) Write API contract for `GET /entries/by-source-session` including auth, tenant/project scoping, paging, and sort order.
- [x] AR-102 (M) Implement `GET /entries/by-source-session` in shared backend server with `source_session_id` and `limit` support.
- [x] AR-103 (S) Add SDK/client request wiring and normalize 404/empty behavior.
- [x] AR-104 (M) Add integration tests for local SQLite, shared-file, and HTTP shared backend parity.

### Priority 2 - Pluggable memory backend (Markdown + vector-native) (M3, M4)
- [x] AR-201 (M) Define `MemoryStore` abstraction for write/read/search/export operations.
- [x] AR-202 (M) Implement Markdown-backed `MemoryStore` adapter and preserve current behavior as baseline.
- [x] AR-203 (S) Add feature flags for `memory.mode`: `markdown`, `hybrid`, `vector_primary`.
- [x] AR-204 (M) Implement local vector store adapter (first backend) with deterministic schema.
- [x] AR-205 (L) Implement TurboPuffer adapter with auth, tenancy mapping, and retry policy.
- [x] AR-206 (M) Define embedding provider abstraction and shared request/response contract.
- [x] AR-207 (M) Implement local embedding provider adapter (local model path and caching).
- [x] AR-208 (M) Implement external embedding provider adapter (API-backed) with timeout/cost guardrails.
- [x] AR-209 (M) Build migration command from tier markdown and chunks into vector records.
- [x] AR-210 (M) Implement hybrid query fusion (lexical + vector) with tunable weights.
- [x] AR-211 (M) Implement vector-primary read path with markdown fallback.
- [x] AR-212 (M) Add cost/privacy controls (quota, redaction hooks, retention knobs).
- [x] AR-213 (M) Add rollout and rollback playbook (dual-read validation + cutover checklist).

### Priority 3 - External compaction review queue + patch preview (M2)
- [x] AR-301 (M) Add queue schema/table for proposed notes with states: `pending`, `approved`, `rejected`, `applied`.
- [x] AR-302 (M) Implement CLI commands: `external-compaction queue list|approve|reject`.
- [x] AR-303 (M) Implement patch preview command that renders before/after diffs by tier.
- [x] AR-304 (S) Add "apply approved notes" command with idempotency checks.
- [x] AR-305 (S) Add audit trail fields (`actor`, `timestamp`, `source_session_ids`) on queue actions.

### Priority 4 - Strict schema validation for external notes (M1)
- [x] AR-401 (S) Define canonical Pydantic models for external note payloads.
- [x] AR-402 (S) Enforce schema validation in `export`, `apply`, and MCP tools.
- [x] AR-403 (S) Return structured validation errors with actionable messages and examples.
- [x] AR-404 (S) Publish JSON schema artifact for external agent/tooling integration.

### Priority 5 - Write-scope hardening (`runtime` vs `templates`) (M1)
- [x] AR-501 (S) Centralize write policy checks into a reusable guard module.
- [x] AR-502 (S) Enforce explicit allowlists for writable files and deny all other paths.
- [x] AR-503 (S) Add config gate to disable template writes unless explicitly enabled.
- [x] AR-504 (S) Add negative tests for path traversal, symlink escapes, and policy bypass attempts.

### Priority 6 - External compaction state in SQLite (M2)
- [x] AR-601 (M) Add `external_compaction_state` table and migration.
- [x] AR-602 (M) Move state read/write logic from JSON file to storage layer.
- [x] AR-603 (S) Add one-time backfill from existing JSON state when present.
- [x] AR-604 (S) Add cleanup command for stale/invalid state entries.

### Priority 7 - Smarter merge semantics for tier updates (M2)
- [x] AR-701 (M) Define section-aware insertion rules per tier (`GUARDRAILS`, `STYLE`, `RECENT`).
- [x] AR-702 (M) Implement semantic dedupe strategy beyond line normalization.
- [x] AR-703 (M) Add conflict policy for contradictory notes (`prefer newest`, `queue for review`).
- [x] AR-704 (S) Persist evidence backlinks from merged notes to source sessions.

### Priority 8 - E2E test matrix for external/MCP/shared flows (M2)
- [x] AR-801 (S) Define test matrix across local/shared backend, MCP on/off, and write targets.
- [x] AR-802 (M) Add E2E tests for export -> external notes -> apply happy path.
- [x] AR-803 (M) Add E2E tests for review queue + diff + approval flow.
- [x] AR-804 (M) Add fault-injection tests (malformed notes, partial failures, retries).
- [x] AR-805 (S) Add CI job grouping for new external-compaction suites.

### Priority 9 - CLI safety defaults (M1)
- [x] AR-901 (S) Change `external-compaction apply` default to `--dry-run`.
- [x] AR-902 (S) Require explicit `--commit` (or equivalent) for write execution.
- [x] AR-903 (S) Add machine-readable output mode with stable exit codes for automation.

### Priority 10 - Pipeline observability (M5)
- [x] AR-1001 (M) Define event schema for ingest/extract/compact/apply lifecycle events.
- [x] AR-1002 (M) Add metrics counters and duration histograms to key pipeline steps.
- [x] AR-1003 (S) Add `agent-recall metrics report` command with last-run summaries.
- [x] AR-1004 (S) Document telemetry fields and interpretation guide.

### Priority 11 - Guardrail enforcement mode (M5)
- [x] AR-1101 (M) Implement guardrail rule parser with severity (`warn`/`block`).
- [x] AR-1102 (M) Add enforcement hooks in Ralph loop and external note apply path.
- [x] AR-1103 (S) Add suppression mechanism (`reason`, `expires_at`, `actor`).
- [x] AR-1104 (S) Add tests for violation detection and override workflows.

### Priority 12 - Retrieval feedback loop (M4)
- [x] AR-1201 (S) Add feedback schema and storage for retrieval results.
- [x] AR-1202 (M) Add CLI/TUI feedback capture actions for retrieved items.
- [x] AR-1203 (M) Integrate feedback signal into rerank/fusion scoring.
- [x] AR-1204 (S) Add quality evaluation script comparing pre/post feedback relevance.

### Priority 13 - PR-aware context builder (M6)
- [x] AR-1301 (M) Implement git diff scope extraction (files/modules changed).
- [x] AR-1302 (M) Add scoped retrieval filters keyed by changed files/tags.
- [x] AR-1303 (S) Add `context --for-pr` output template optimized for code review.
- [x] AR-1304 (S) Add tests for rename/move and large-diff edge cases.

### Priority 14 - Cross-session topic threads (M6)
- [x] AR-1401 (M) Implement offline clustering job for related learnings.
- [x] AR-1402 (M) Add storage for thread metadata and linked entries.
- [x] AR-1403 (S) Add CLI query command to inspect topic threads.
- [x] AR-1404 (S) Add summarization output for top active threads.

### Priority 15 - Rule confidence + decay/refresh (M5)
- [x] AR-1501 (M) Define confidence model with reinforcement and decay factors.
- [x] AR-1502 (M) Implement scheduled decay job and rule staleness tagging.
- [x] AR-1503 (S) Add archive/prune command for low-confidence stale rules.
- [x] AR-1504 (S) Add visibility in status output for confidence and age.

### Priority 16 - Memory packs (import/export) (M6)
- [x] AR-1601 (S) Define memory pack file format and versioning scheme.
- [x] AR-1602 (M) Implement `memory-pack export` from tiers/chunks/metadata.
- [x] AR-1603 (M) Implement `memory-pack import` with merge conflict strategies.
- [x] AR-1604 (S) Add compatibility checks and validation reports.

### Priority 17 - Multi-agent attribution (M5)
- [x] AR-1701 (M) Propagate attribution metadata through ingest/extract/compact pipelines.
- [x] AR-1702 (S) Store attribution at note/entry level with immutable source metadata.
- [x] AR-1703 (S) Add CLI filters and summaries by agent/provider.
- [x] AR-1704 (S) Add attribution-aware audit output for compaction actions.

### Priority 18 - Sync/compact status UX polish (M1)
- [x] AR-1801 (S) Add backend-specific status messaging with concrete next-step commands.
- [x] AR-1802 (S) Add clearer warning and remediation text for skipped/deferred compaction.
- [x] AR-1803 (S) Update docs and examples to mirror real command output and flows.

## Delivery Guardrails

- Keep all new behavior behind explicit config flags until each milestone exits.
- Ship with rollback paths for storage and retrieval mode changes.
- Require tests for every ticket marked complete.
- Avoid destructive migrations without backups and dry-run support.
