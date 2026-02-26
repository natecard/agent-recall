# Release Notes: Semantic Embeddings Feature

**Version:** 0.2.0  
**Release Date:** 2026-02-25

---

## Overview

This release introduces **semantic embeddings** to Agent-Recall, enabling meaning-based search that goes beyond traditional keyword matching. With embeddings, you can now find relevant context even when your query uses different words than your stored notes.

### What Are Semantic Embeddings?

Think of it like this: spelling vs. meaning. Two words can be spelled completely differently but have the same meaning ("authentication" vs "login"), or be spelled similarly but mean different things ("bat" the animal vs "bat" the sports equipment).

Traditional keyword search works like spell-checking - it looks for exact word matches. If you search for "JWT error" but your notes say "token validation failed", keyword search finds nothing. Semantic embeddings solve this by converting your text into numerical vectors that capture **meaning** rather than just words.

---

## What Changed

### Configuration Schema

New configuration options were added under the `retrieval` section:

```yaml
retrieval:
  backend: hybrid        # Options: fts, semantic, hybrid
  embedding_enabled: true # Enable semantic search
  top_k: 10              # Number of results to return
  fusion_k: 20           # Results to consider in hybrid fusion

# Optional: Embedding-specific settings
embedding:
  model: all-MiniLM-L6-v2
  cache_enabled: true
  cache_size: 10000
  use_faiss: false       # For 100K+ chunks
```

### Retrieval Behavior

- **Default (embeddings disabled):** Uses full-text search (FTS) only
- **Semantic mode:** Uses vector similarity search only
- **Hybrid mode (recommended):** Combines FTS and semantic search using rank fusion

### Database Schema

New columns added to the `chunks` table:
- `embedding` (TEXT): JSON-serialized 384-dimensional vector

---

## Migration Guide

### Prerequisites

- Python 3.11+
- SQLite database with existing Agent-Recall data

### Step 1: Backup Your Database

Before migrating, create a backup:

```bash
cp .agent/agent-recall.db .agent/agent-recall.db.backup-$(date +%Y%m%d)
```

### Step 2: Run the Migration Script

```bash
# Preview changes without applying
agent-recall embedding migrate-embeddings --dry-run

# Run migration (creates backup automatically)
agent-recall embedding migrate-embeddings

# Skip backup creation (not recommended)
agent-recall embedding migrate-embeddings --no-backup
```

### Step 3: Index Existing Chunks

```bash
# Index all chunks without embeddings
agent-recall embedding reindex --force

# Or limit to a specific number of chunks
agent-recall embedding reindex --max-chunks 1000
```

### Step 4: Verify

```bash
# Check embedding coverage
agent-recall embedding stats

# Test semantic search
agent-recall embedding search --query "JWT authentication"
```

### Rollback

If you need to rollback:

```bash
# Restore from backup
cp .agent/agent-recall.db.backup-20260225 .agent/agent-recall.db
```

---

## Performance Impact

### Indexing Time

| Chunks | Time (estimated) | Notes |
|--------|------------------|-------|
| 1,000  | 30-60 seconds    | First-time model download: +2-5 min |
| 10,000 | 5-10 minutes     | With warm model cache |
| 100,000| 45-90 minutes    | Consider FAISS for this scale |

### Storage

- Each embedding: ~1.5 KB (384 dimensions × 4 bytes/float)
- 10,000 chunks: ~15 MB additional storage
- 100,000 chunks: ~150 MB additional storage

### Query Latency

| Mode | Latency | Notes |
|------|---------|-------|
| FTS only | 1-5 ms | Baseline |
| Semantic | 10-50 ms | Depends on cache state |
| Hybrid | 15-60 ms | Combines both |
| FAISS | <10 ms | For 100K+ chunks |

### Cache Performance

With in-memory LRU caching enabled (default):
- First query: 10-50 ms (database lookup + deserialization)
- Subsequent queries: <1 ms (cache hit)

---

## Troubleshooting

### "Model not found" Error

**Symptom:** `OSError: model all-MiniLM-L6-v2 not found`

**Solution:** The sentence-transformers library will download the model automatically on first use. Ensure you have internet access. If behind a firewall, manually download:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### "No embeddings found" After Migration

**Symptom:** `agent-recall embedding stats` shows 0% coverage after running `migrate-embeddings`

**Solution:** The migration only adds the database column. You must run the indexer:

```bash
agent-recall embedding reindex --force
```

### Slow Query Performance

**Symptom:** Queries take >100ms even with embeddings

**Solutions:**
1. Enable caching (default on): `embedding.cache_enabled: true`
2. Reduce `top_k`: `retrieval.top_k: 5`
3. For 100K+ chunks, enable FAISS: `embedding.use_faiss: true`

### Hybrid Search Not Finding Results

**Symptom:** Expected results not appearing in hybrid mode

**Solutions:**
1. Verify embeddings exist: `agent-recall embedding stats`
2. Lower the threshold in config: `retrieval.semantic_threshold: 0.3`
3. Try semantic-only mode: `retrieval.backend: semantic`

### Out of Memory During Indexing

**Symptom:** Process killed during `reindex`

**Solution:** Index in batches:

```bash
agent-recall embedding reindex --max-chunks 500
# Repeat until complete
```

---

## Upgrade Checklist

- [ ] Backup database: `cp .agent/agent-recall.db .agent/agent-recall.db.backup`
- [ ] Update `.agent/config.yaml` with new retrieval settings
- [ ] Run migration: `agent-recall embedding migrate-embeddings`
- [ ] Index embeddings: `agent-recall embedding reindex --force`
- [ ] Verify: `agent-recall embedding stats`
- [ ] Test: `agent-recall embedding search --query "your query"`
- [ ] Update team members on new capabilities

---

## Quick Start (5 Minutes)

Want embeddings working with defaults? Here's the fastest path:

```bash
# 1. Add to your .agent/config.yaml
echo 'retrieval:
  backend: hybrid
  embedding_enabled: true' >> .agent/config.yaml

# 2. Run migration (creates backup automatically)
agent-recall embedding migrate-embeddings --force

# 3. Index your existing chunks
agent-recall embedding reindex --force

# 4. Verify it's working
agent-recall embedding stats
```

That's it! Your context retrieval now uses semantic search.

---

## Credits

This feature was implemented using:

- **sentence-transformers** (MIT License) - For the `all-MiniLM-L6-v2` embedding model
- **sqlite-vec** (MIT License) - For vector search in SQLite (optional, with pure Python fallback)
- **Prometheus metrics** (Apache 2.0) - For observability
- **FAISS** (MIT License) - For approximate nearest neighbor search at scale

---

## PRD References

This release implements the following PRD items:

- **AR-501:** Tier format detection layer
- **AR-502:** Tier format structured parsing
- **AR-503:** PRD archive data model
- **AR-504:** PRD archive semantic indexing
- **AR-505:** Context refresh hook
- **AR-506:** Hybrid retrieval rank fusion
- **AR-507:** Configurable retrieval backends
- **AR-508:** Embedding indexer with batch processing
- **AR-509:** Vector similarity search
- **AR-510:** Embedding CLI commands
- **AR-511:** Command contract parity
- **AR-512:** Migration script
- **AR-513:** In-memory embedding cache
- **AR-514:** Prometheus metrics
- **AR-515:** FAISS integration

---

## Questions?

- Check the full documentation: `docs/embeddings-guide.md`
- Run `agent-recall --help` for CLI reference
- Review benchmarks: `docs/BENCHMARKS.md`
