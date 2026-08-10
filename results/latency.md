# Retrieval & rerank latency (measured)

Per-stage wall-clock over the 20-question golden set, warm (one-time model load excluded).
Measured on **CPU** — the free/local build — via `eval/compare_retrievers.py --latency`.

| stage | p50 | p95 | mean | max |
|---|---|---|---|---|
| dense (query-embed + pgvector kNN) | 53 ms | 106 ms | 57 ms | 146 ms |
| lexical (Postgres FTS) | 51 ms | 244 ms | 85 ms | 246 ms |
| hybrid (dense + lexical + RRF) | 121 ms | 249 ms | 146 ms | 261 ms |
| **+ cross-encoder rerank** | **24.9 s** | **45.7 s** | 28.8 s | 54.6 s |

## Reading

- **Bi-encoder retrieval is production-grade on CPU: p95 ≤ 250 ms.** Dense (query embed +
  pgvector HNSW) is p95 **106 ms**; the fused hybrid adds the lexical query and RRF for p95
  **249 ms**. This is the honest answer to "how fast is retrieval?" — under a quarter-second.
- **The cross-encoder reranker is the bottleneck — a hardware choice, not an architecture flaw.**
  On CPU it re-scores ~20 (query, chunk) pairs with a 278 M-param model at ~1.2 s/pair →
  **p50 ~25 s**. This is the *same* free/local-inference tax that makes generation slow (see
  [`notes.md`](../notes.md)); it is not pgvector, Postgres, or the fusion. On a GPU or a hosted
  reranker endpoint it drops to sub-second with **zero code change** — the reranker already runs
  only on the ~20-candidate pool, never the corpus.
- **Levers if the CPU rerank latency matters for a live demo:** (1) run the reranker on GPU / a
  hosted endpoint; (2) shrink the candidate pool (`settings.rerank_candidates`, currently 20);
  (3) offer a **"fast mode"** returning the hybrid top-k without reranking — p95 ≤ 250 ms at the
  0.45-vs-0.75 answer@5 trade-off the [retrieval table](results.csv) quantifies.

## Query-embed latency (from `eval/bench_embed.py`)

The dense/hybrid numbers above include query embedding: **MiniLM-384 = 19 ms/query**. The
documented e5-large upgrade would raise that to **153 ms/query** (still fine at query time) but
costs **~50×** on the full re-index (~7 h vs ~8 min) — see [`config.py`](../src/agroteca/config.py)
and `eval/bench_embed.py`.

## Reproduce
```bash
uv run python eval/compare_retrievers.py --k 5 --latency   # per-stage p50/p95
uv run python eval/bench_embed.py                          # embedder throughput + query latency
```
