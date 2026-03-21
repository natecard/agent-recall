# Memory Pack Format (`agent-recall-memory-pack`)

Memory packs are JSON payloads used for backup, migration, and cross-repo transfer.

## Versioning

- `format`: must be `agent-recall-memory-pack`
- `version`: semantic major/minor string (current: `1.0`)
- Compatibility rule:
  - major `1.x` is accepted
  - other majors are rejected by `memory-pack validate`

## Top-Level Shape

```json
{
  "format": "agent-recall-memory-pack",
  "version": "1.0",
  "created_at": "2026-03-21T00:00:00+00:00",
  "tiers": {
    "GUARDRAILS": "...",
    "STYLE": "...",
    "RECENT": "..."
  },
  "chunks": [
    {
      "id": "...",
      "source": "log_entry|compaction|import|manual",
      "source_ids": ["..."],
      "content": "...",
      "label": "...",
      "tags": ["..."],
      "created_at": "...",
      "token_count": 123,
      "embedding": [0.1, 0.2],
      "embedding_version": 1
    }
  ],
  "metadata": {
    "stats": {},
    "chunk_count": 0,
    "tier_chars": {}
  }
}
```

## Import Merge Strategies

- `skip`: only fill missing tiers; existing tier content remains unchanged
- `append`: append incoming tier content if not already present
- `overwrite`: replace local tier content with pack content

Chunk import de-duplicates by `(content, label)` and skips exact matches.
