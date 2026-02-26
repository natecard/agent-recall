# Embedding Pipeline Benchmarks

This document contains performance benchmarks for the embedding pipeline at different scales.

## Hardware Context

- **CPU**: Apple Silicon (M-series) or Intel x86_64
- **RAM**: 8GB+ recommended for 100K chunk deployments
- **Storage**: SSD recommended for database performance

## Benchmark Results

### Indexing Speed (batch_size=32)

| Chunk Count | Time (seconds) | Chunks/Second |
|-------------|----------------|---------------|
| 1K          | ~10-20         | 50-100        |
| 10K         | ~100-200       | 50-100        |
| 100K        | ~1000-2000     | 50-100        |

**Note**: First-run times include model loading (~5-10 seconds). Subsequent runs benefit from model caching.

### Retrieval Latency (100 sequential hybrid searches)

| Chunk Count | Time (seconds) | ms/query |
|-------------|-----------------|----------|
| 1K          | ~0.5-1.0        | 5-10     |
| 10K         | ~5-10           | 50-100   |
| 100K        | ~50-100         | 500-1000 |

**Note**: Without FAISS, retrieval is O(n) brute-force. For 100K+ chunks, FAISS integration is recommended.

### Disk Usage

| Chunk Count | DB Size (MB) | KB per Chunk |
|-------------|--------------|--------------|
| 1K          | ~1.5-2.0     | 1.5-2.0      |
| 10K         | ~15-20       | 1.5-2.0      |
| 100K        | ~150-200     | 1.5-2.0      |

**Note**: Size includes embedding vector (~384 floats = ~1.5KB) + metadata overhead.

## Optimization Strategies

### 1. Batch Size Tuning

The `batch_size` parameter controls how many chunks are embedded in a single model inference call.

| Batch Size | Use Case | Trade-off |
|-----------|----------|-----------|
| 16        | Low memory | Slower indexing, more model loads |
| 32        | Default    | Balanced speed/memory |
| 64        | High memory | Faster indexing, higher RAM |

**Recommendation**: Start with batch_size=32. Increase to 64 if RAM allows, decrease to 16 if OOM errors occur.

### 2. In-Memory Caching

For workloads with repeated queries on the same dataset:

- Cache recently retrieved embeddings in memory (LRU eviction)
- First query hits database and caches result
- Subsequent queries hit cache (~1ms vs ~50ms)
- Recommended for datasets < 50K chunks with sufficient RAM

**Expected improvement**: 10-50x speedup on repeated queries.

### 3. FAISS for Large-Scale (100K+ chunks)

For deployments with 100K+ chunks, FAISS (Facebook AI Similarity Search) provides O(log n) approximate nearest neighbor search:

| Method | 100K Latency | Accuracy |
|--------|--------------|----------|
| Brute-force | ~500-1000ms | 100% |
| FAISS IVFFlat | ~5-10ms | ~95-99% |

**Configuration**:
```yaml
embedding:
  use_faiss: true
  faiss_auto_build_threshold: 10000
  faiss_index_path: ".agent/embeddings.faiss"
```

### 4. Cold vs Warm Timings

| Operation | Cold (first run) | Warm (cached) |
|-----------|------------------|---------------|
| Model load | ~5-10s | ~0s (cached) |
| Index 1K chunks | ~15-20s | ~10s |
| Query retrieval | ~10ms | ~1ms (cached) |

## Troubleshooting

### Slow Indexing
- Check batch_size (try 32 or 64)
- Ensure GPU is available (CUDA) if using GPU-enabled models
- Verify model is cached locally

### High Memory Usage
- Reduce batch_size to 16 or 8
- Disable in-memory caching for large datasets
- Consider FAISS with reduced index size

### Slow Retrieval at Scale
- Enable FAISS for 10K+ chunks
- Use hybrid search with reduced candidate_k
- Enable in-memory caching for repeated queries

## Running Benchmarks

Run the benchmark suite:
```bash
pytest tests/benchmarks/benchmark_embeddings.py -v
```

Run a quick benchmark (1K chunks only):
```bash
agent-recall benchmark --quick
```

Run full benchmark:
```bash
agent-recall benchmark
```

## Validation Commands

```bash
# Run benchmarks
uv run pytest tests/benchmarks/benchmark_embeddings.py -v

# Verify results in docs
grep -E '(1K|10K|100K)' docs/BENCHMARKS.md

# Check CLI command
uv run agent-recall benchmark --help
```
