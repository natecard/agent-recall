# Semantic Embeddings Guide

This guide explains semantic embeddings in Agent-Recall, why they improve search results, how to configure them, and how to troubleshoot common issues.

## Overview

Think of embeddings like this: spelling vs. meaning. Two words can be spelled completely differently but have the same meaning ("authentication" vs "login"), or be spelled similarly but mean different things ("bat" the animal vs "bat" the sports equipment).

Traditional keyword search (FTS - Full-Text Search) works like spell-checking - it looks for exact word matches. If you search for "JWT error" but your notes say "token validation failed", keyword search finds nothing - even though they're about the same topic.

Semantic embeddings solve this by converting your text into numerical vectors (lists of numbers) that capture **meaning** rather than just words. These vectors live in a high-dimensional space where similar meanings cluster together. When you search, your query becomes a vector too, and the system finds the closest matches - even if they use completely different words.

## Why Embeddings Help

Here are concrete examples where semantic search outperforms keyword search:

| Keyword Search Query | Traditional Result | With Semantic Embeddings |
|---------------------|-------------------|-------------------------|
| "JWT error" | Finds "JWT error" only | Finds "token validation failed", "authentication rejected", "bearer token problems" |
| "database optimization" | Exact phrase match | Finds "query performance tuning", "SQL speed improvements", "index optimization" |
| "API error handling" | "API error handling" | Finds "exception management", "HTTP status codes", "try-catch patterns" |
| "user login" | Exact match | Finds "authentication flow", "sign-in implementation", "session initialization" |

This matters because your notes naturally use different words than your future queries. Semantic search bridges that gap.

## How It Works

The embedding pipeline follows this architecture:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│ User Notes  │───>│ Tokenization │───>│  Embedding  │───>│  Vector    │
│ (text)      │    │ (text→tokens)│    │ (tokens→384│    │  Storage   │
└─────────────┘    └──────────────┘    │ dimension   │    │  (SQLite)  │
                                       │ vectors)    │    └────────────┘
                                             │              │
                                             ▼              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   Query     │───>│   Embed      │───>│  Similarity │<───│ Retrieve   │
│   "token    │    │   Query     │    │   Search    │    │ Candidates │
│   problem"  │    │   →vector   │    │   (cosine)  │    │            │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                                                  │
                                                  ▼
                                           ┌────────────┐
                                           │   Ranked   │
                                           │   Results  │
                                           └────────────┘
```

1. **Text Input**: Your logged learnings, patterns, and notes
2. **Tokenization**: Split text into tokens the model understands
3. **Embedding**: Convert to 384-dimensional vector using `all-MiniLM-L6-v2` model
4. **Storage**: Store vectors alongside chunk data in SQLite
5. **Retrieval**: When you query, embed the query and find nearest neighbors
6. **Ranking**: Return results sorted by similarity score

## Quick Start

### Default Behavior (Embeddings Disabled)

By default, Agent-Recall uses full-text search (FTS) only:

```bash
# This uses keyword search
agent-recall context --task "add password reset"
```

### Enabling Semantic Search

To enable embeddings, update your config:

```yaml
# .agent/config.yaml
retrieval:
  backend: hybrid  # or "semantic" for vector-only
  semantic_index_enabled: true
```

Then run the migration and indexing:

```bash
# One-time migration to add embedding columns
agent-recall embedding migrate-embeddings

# Index existing chunks (one-time, takes 5-10 minutes for 10K chunks)
agent-recall embedding reindex --force
```

Now hybrid search is enabled. Your `context` and `search` commands will use both keyword and semantic matching.

### Verify It's Working

```bash
# Check embedding coverage
agent-recall embedding stats

# Test semantic search
agent-recall embedding search --query "JWT authentication"
```

## Configuration

All embedding settings live under the `retrieval` section in your `.agent/config.yaml`:

```yaml
retrieval:
  # Search backend: fts5, semantic, or hybrid
  backend: hybrid

  # Enable semantic embeddings
  semantic_index_enabled: true

  # Embedding dimension (64 for deterministic, 384 for semantic model)
  embedding_dimensions: 384

  # Number of results to return
  top_k: 5

  # Hybrid search: how many candidates to consider from each method
  fusion_k: 60

  # Enable reranking after initial retrieval
  rerank_enabled: false
  rerank_candidate_k: 20
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `"fts5"` | Search backend: `fts5` (keyword), `semantic` (vector-only), `hybrid` (both) |
| `semantic_index_enabled` | boolean | `false` | Enable semantic embeddings |
| `embedding_dimensions` | integer | `64` | Vector dimensions (64=deterministic, 384=semantic model) |
| `top_k` | integer | `5` | Default results per query |
| `fusion_k` | integer | `60` | Candidates from each search method before fusion |
| `rerank_enabled` | boolean | `false` | Apply additional scoring after retrieval |
| `rerank_candidate_k` | integer | `20` | Candidates to consider when reranking |

### CLI Commands

```bash
# Re-index all embeddings
agent-recall embedding reindex --force

# Re-index limited chunks (useful for testing)
agent-recall embedding reindex --force --max-chunks 1000

# Check embedding statistics
agent-recall embedding stats

# Test semantic search
agent-recall embedding search --query "your search query"

# Test search quality with built-in queries
agent-recall embedding test-quality

# Migrate existing database to support embeddings
agent-recall embedding migrate-embeddings --dry-run  # Preview
agent-recall embedding migrate-embeddings            # Run migration
```

## Performance Characteristics

Benchmarks run on a typical laptop (Apple M2, 16GB RAM) with `all-MiniLM-L6-v2` model:

| Dataset Size | Disk Space | Embedding Time (one-time) | Query Latency |
|-------------|------------|--------------------------|---------------|
| 1,000 chunks | ~50 MB | ~5 minutes | ~5 ms |
| 10,000 chunks | ~500 MB | ~45 minutes | ~50 ms |
| 100,000 chunks | ~5 GB | ~8 hours | ~500 ms |

### Factors Affecting Performance

- **Batch size**: Default is 32 chunks per batch. Increase to 64 for faster indexing on powerful machines.
- **Model loading**: First embedding call downloads model (~90 MB) and caches in memory. Subsequent calls are instant.
- **Database size**: Vector storage adds ~1.5 KB per chunk.
- **Query complexity**: Simple queries are faster. Complex boolean queries in hybrid mode take longer.

### Optimization Tips

1. **Run indexing overnight**: For large datasets, schedule `embedding reindex` during off-hours.
2. **Use `--max-chunks` for testing**: Test with 1000 chunks first before indexing everything.
3. **Monitor with `embedding stats`**: Check coverage and identify stale embeddings.
4. **Consider FAISS for 100K+**: For very large deployments, FAISS provides sub-10ms query times.

## Hybrid Search Explained

Agent-Recall supports three retrieval modes:

### 1. FTS5 (Full-Text Search)
- **How it works**: Tokenizes query, matches against indexed terms using inverted index
- **Best for**: Exact phrase matches, code snippets, technical terms
- **Example**: Searching for specific error messages

```python
# Internally uses SQLite FTS5
results = storage.search_chunks_fts(query="JWT token", top_k=5)
```

### 2. Semantic (Vector Search)
- **How it works**: Embeds query, finds nearest neighbor vectors using cosine similarity
- **Best for**: Concept matching, different phrasings, related topics

```python
# Embeds query, searches vector space
results = retriever.search_by_vector_similarity(
    query="authentication problems",
    top_k=5,
    min_similarity=0.3  # Only return results above 30% similarity
)
```

### 3. Hybrid (Combined)
- **How it works**: Runs both FTS and semantic search, then fuses results using weighted scoring
- **Default weights**: FTS=40%, Semantic=60%
- **Best for**: General-purpose search - catches both exact matches and semantic matches

```python
# Combines both approaches with weighted fusion
results = retriever.search_hybrid(
    query="JWT authentication",
    top_k=5,
    fts_weight=0.4,      # 40% weight for keyword matches
    semantic_weight=0.6  # 60% weight for semantic matches
)
```

The fusion algorithm:
1. Runs FTS and semantic search independently
2. Normalizes scores (0-1 range for each)
3. Applies weights and combines scores
4. Sorts by combined score with tie-breakers (semantic first, then FTS, then chunk ID)

## Disabling Embeddings

If you need to disable embeddings:

```yaml
# .agent/config.yaml
retrieval:
  backend: fts5
  semantic_index_enabled: false
```

### Fallback Behavior

When `semantic_index_enabled: false`:
- `agent-recall context` uses FTS only
- `agent-recall embedding search` returns an error
- Database retains existing embeddings (they're not deleted)

### Re-enabling

To re-enable after disabling:
```bash
agent-recall embedding reindex --force
```

Existing embeddings are preserved; only new chunks need indexing.

## Troubleshooting

### Problem: "No embeddings found" when running `embedding search`

**Cause**: No chunks have been indexed with embeddings.

**Solution**:
```bash
# First, migrate the database
agent-recall embedding migrate-embeddings

# Then index your chunks
agent-recall embedding reindex --force
```

### Problem: Slow indexing (> 1 hour for 10K chunks)

**Cause**: Default batch size may be suboptimal, or system is resource-constrained.

**Solution**:
1. Check you have sufficient RAM (8GB+ recommended)
2. The model is cached after first use - subsequent runs are faster
3. Consider limiting initial indexing to most important chunks:
   ```bash
   agent-recall embedding reindex --force --max-chunks 5000
   ```

### Problem: Poor retrieval quality (irrelevant results)

**Cause**: Embeddings not generated for your content, or similarity threshold too low.

**Solution**:
1. Check embedding coverage:
   ```bash
   agent-recall embedding stats
   ```
2. Increase minimum similarity threshold in your query:
   ```python
   retriever.search_by_vector_similarity(query="...", min_similarity=0.5)
   ```
3. Run quality test to diagnose:
   ```bash
   agent-recall embedding test-quality
   ```

### Problem: High disk usage

**Cause**: Each embedding uses ~1.5 KB. 100K chunks = ~150 MB for embeddings alone.

**Solution**:
1. Use deterministic embeddings (smaller dimension):
   ```yaml
   retrieval:
     embedding_dimensions: 64  # instead of 384
   ```
2. Delete embeddings for old, unused chunks (manual process - no CLI command yet)

### Problem: Import errors for sentence-transformers

**Cause**: Optional dependency not installed.

**Solution**:
```bash
pip install sentence-transformers
```

The embedding feature requires `sentence-transformers` which is installed as part of the default package. If you encounter import errors, verify the package is installed.

## FAQ

### 1. Do I need embeddings for Agent-Recall to work?

No. Embeddings are optional. Agent-Recall works with FTS (keyword search) alone. Enable embeddings only if you want semantic search capabilities.

### 2. How much additional storage do embeddings require?

Approximately 1.5 KB per chunk. For 10,000 chunks, embeddings add ~15 MB to your database.

### 3. Can I use embeddings with shared storage?

Yes. Embeddings are stored in the shared database and synchronized across instances. The migration script works with both local and shared SQLite backends.

### 4. What embedding model is used?

`all-MiniLM-L6-v2` - a lightweight, fast model producing 384-dimensional vectors. It's optimized for semantic similarity tasks and runs well on CPU.

### 5. How do embeddings affect query latency?

- FTS: ~1-2 ms
- Semantic: ~5-50 ms (depends on chunk count)
- Hybrid: ~10-60 ms (runs both, then fuses)

For most use cases, hybrid search at 10K chunks runs in under 100ms.

### 6. Can I use a different embedding model?

Not currently. The model is hardcoded to `all-MiniLM-L6-v2` for consistency. Future versions may support model selection.

### 7. What happens if I delete my database and start fresh?

You'll need to re-run `embedding migrate-embeddings` and `embedding reindex` after re-syncing your sessions.

### 8. Are embeddings updated when I log new content?

Not automatically. New chunks are not embedded by default. To index new content:

```bash
agent-recall embedding reindex --force
```

Or configure automatic indexing in your sync workflow (advanced - requires custom integration).

---

For implementation details, see:
- `src/agent_recall/core/semantic_embedder.py` - Embedding model integration
- `src/agent_recall/core/embedding_indexer.py` - Batch indexing pipeline
- `src/agent_recall/core/retrieve.py` - Hybrid retrieval implementation
- `src/agent_recall/storage/migrations/migrate_to_embeddings.py` - Database migration
