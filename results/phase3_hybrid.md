# Phase 3 — Hybrid retrieval (dense + lexical + RRF)

Same corpus/index as the baseline (10,330 chunks, MiniLM-384, unchanged embeddings).
The only change: a lexical retriever (Postgres FTS, `simple` config, OR-of-content-terms)
fused with the dense retriever via Reciprocal Rank Fusion (k=60, 60 candidates each).

## Results (answerable golden set, n=19)

| method | answer@5 | doc@5 | answer@10 | doc@10 |
|---|---|---|---|---|
| dense (baseline) | 0.32 | 0.63 | 0.32 | 0.63 |
| lexical | 0.47 | 0.84 | 0.58 | 0.84 |
| **hybrid (RRF)** | **0.42** | 0.63 | **0.63** | **0.79** |

## Honest reading
- **Hybrid ≥ dense at every k** (ship criterion met): answer@5 0.32→0.42; answer@10 0.32→0.63 (~2×).
  RRF fixed q05, q14 vs dense with **zero regressions**.
- **Lexical alone beats dense** — expected: distinctive-term questions + a weak 384-dim dense model.
- **Equal-weight RRF@5 (0.42) trails lexical@5 (0.47).** Fusing a weak dense retriever with a strong
  lexical one dilutes precision at small k. Non-obvious, and the honest result.
- **Hybrid's real contribution is recall into the candidate pool:** answer@10 = 0.63, doc@10 = 0.79.
  The answer chunk is in the fused top-10 ~63% of the time — the pool a reranker (Phase 4) needs.

## Still failing (targets for Phase 4 / upgrades)
- q07 (Market Gardener "30-inch bed"): right doc top-ranked, answer chunk never surfaces → reranking.
- q20–q22 (short synthetic notes), q01 (water guide): weak recall → dense-model upgrade + weighted fusion.

## Next levers (measure each; don't stack blindly)
1. **Reranking (Phase 4)** over the fused top-10/20 — highest expected gain given 0.63 pool recall.
2. **Upgrade dense model** to e5-large / BGE-M3 (config + re-index) — lifts the weak half of the fusion.
3. **Weighted RRF** favouring the stronger retriever on this corpus.

## Reproduce
```bash
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/002_tsv_simple.sql
uv run python eval/compare_retrievers.py --k 5
```
