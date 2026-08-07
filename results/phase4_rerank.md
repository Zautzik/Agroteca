# Phase 4 — Cross-encoder reranking

Adds the **precision stage**: retrieve a candidate pool with hybrid (fast bi-encoders,
good recall), then re-score those ~20 candidates with a **multilingual cross-encoder**
(`jinaai/jina-reranker-v2-base-multilingual`) that reads each `(query, chunk)` pair
*together*. The cross-encoder runs only on the small pool, never the whole corpus.

> **Reconciliation note.** These figures were measured on the **n=19** answerable golden set, *before* the Phase-5 correction of the mislabeled question **q18**. Correcting it grew the set to **n=20** and nudged the numbers up (answer@5: dense **0.35**, hybrid **0.45**, rerank **0.75**; doc@5 rerank **0.90**). Current authoritative figures: [`eval/results.csv`](../eval/results.csv). The story is unchanged — dense → hybrid → rerank, **+6 fixed, 0 regressions**, lexical > hybrid at k=5.

## Results (answerable golden set, n=19, k=5)

| method | answer@5 | doc@5 |
|---|---|---|
| dense (baseline) | 0.32 | 0.63 |
| lexical | 0.47 | 0.84 |
| hybrid (RRF) | 0.42 | 0.63 |
| **rerank** | **0.74** | **0.89** |

**The full arc: retrieval@5 `0.32 → 0.42 → 0.74`** (dense → hybrid → +rerank).

## Reading
- **Reranking fixed 6 questions hybrid missed** (q01, q02, q03, q09, q16, q21) with
  **zero regressions**. Pure gain.
- The precision jump (0.42 → 0.74) confirms the Phase-2 diagnosis: the answer chunk was
  usually *in the pool* but not on top; the cross-encoder floats it up. `doc@5` also rose
  (0.63 → 0.89) because rerank draws from a 20-candidate pool.
- **Methodology note (the real lesson):** a single reranked query looked *noisy* by eye
  (weeds table / title page / bibliography on top), yet the aggregate is **0.74**. Trust
  the metric over the golden set — never eyeball one query.

## Still missed (honest headroom): q07, q08, q15, q20, q22
Reranking can only reorder what retrieval fetched — so these are likely **recall** gaps
(the answer chunk isn't in the candidate pool at all). Levers: a stronger embedder
(e5-large / BGE-M3), a larger candidate count, or per-document chunking review.

## Reproduce
```bash
uv run python eval/compare_retrievers.py --k 5
```
