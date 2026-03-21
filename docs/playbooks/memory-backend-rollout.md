# Memory Backend Rollout and Rollback Playbook

## Scope

This playbook covers migration from markdown-first retrieval to hybrid/vector-primary modes.

## Rollout Checklist

1. Baseline snapshot:
   - Run `agent-recall memory-pack export --output .agent/export/pre-rollout-pack.json`.
   - Commit or archive `.agent/GUARDRAILS.md`, `.agent/STYLE.md`, `.agent/RECENT.md`.
2. Dry-run vector migration:
   - Run `agent-recall memory migrate-vectors --dry-run --format json`.
   - Confirm `rows_migrated > 0` and cost estimate is within budget.
3. Dual-read validation:
   - Keep `memory.mode=markdown`.
   - Run:
     - `agent-recall retrieve "<query>" --backend fts5`
     - `agent-recall retrieve "<query>" --backend hybrid`
   - Compare top results for critical workflows.
4. Incremental cutover:
   - Set `memory.mode=hybrid`.
   - Observe for one iteration window.
   - Set `memory.mode=vector_primary` only after hybrid parity is acceptable.
5. Retention/privacy controls:
   - Configure `memory.privacy.redaction_patterns`.
   - Configure `memory.privacy.retention_days`.
   - Run `agent-recall memory prune-vectors`.

## Rollback Checklist

1. Immediate rollback:
   - `agent-recall memory mode --set markdown`
2. Restore tier/chunk state if needed:
   - `agent-recall memory-pack import --input .agent/export/pre-rollout-pack.json --strategy overwrite`
3. Re-run retrieval sanity checks with `--backend fts5`.
4. Keep vector store files for forensics unless policy requires deletion.
