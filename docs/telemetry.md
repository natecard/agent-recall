# Pipeline Telemetry

`agent-recall` records lightweight local telemetry for core lifecycle stages:

- `ingest`
- `extract`
- `compact`
- `apply`

Telemetry files are written under `.agent/metrics/`:

- `pipeline-events.jsonl`: append-only event log
- `pipeline-metrics.json`: aggregated counters and duration histograms

## Event Schema

Each `pipeline-events.jsonl` line follows this shape:

```json
{
  "id": "uuid",
  "run_id": "sync-20260321T140500-ab12cd34",
  "stage": "extract",
  "action": "complete",
  "created_at": "2026-03-21T14:05:01.123456+00:00",
  "success": true,
  "duration_ms": 42.7,
  "metadata": {
    "source": "cursor",
    "session_id": "cursor-session-123"
  }
}
```

Field guide:

- `run_id`: correlates events that belong to one pipeline run.
- `stage`: lifecycle area (`ingest`, `extract`, `compact`, `apply`).
- `action`: `start`, `complete`, or `error`.
- `success`: boolean outcome when known.
- `duration_ms`: measured duration for completed/error steps.
- `metadata`: contextual details (source/session/backend/counts/error message).

## Aggregates

`pipeline-metrics.json` stores:

- `counters.events_total`: total emitted events.
- `counters.by_stage.<stage>`:
  - `start`, `complete`, `error`
  - `success`, `failure`
- `duration_histograms_ms.<stage>`: bucketed duration counts.

## Interpreting Metrics

- Rising `error` for `extract`: likely LLM/provider or transcript issues.
- Rising `error` for `apply`: likely note payload/schema problems.
- High `compact` duration buckets: compaction workload or provider latency.
- High `ingest` duration with low extracted learnings: noisy transcripts or filtering mismatch.

## CLI Reporting

Use:

```bash
agent-recall metrics report
agent-recall metrics report --format json
```

`--format json` is intended for automation and dashboards.
