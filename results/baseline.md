# Phase 2 baseline — retrieval@k

The "before" number for the baseline→final story. Dense-only retrieval, no hybrid, no rerank.

> **Reconciliation note.** These are the **n=19** figures, measured *before* the Phase-5 q18 correction (which grew the answerable set to **n=20**). On the current set the dense baseline is **0.35 / 0.65**; authoritative figures live in [`eval/results.csv`](../eval/results.csv). The diagnosis below (chunk-vs-document gap → reranking; recall ceiling → hybrid) is unchanged.

**Setup**
- Corpus: 71 docs, 10,330 chunks (tiers: open + synthetic + local).
- Embedder: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, fastembed/ONNX, CPU).
- Chunking: recursive ~512 tokens / 64 overlap.
- Retriever: dense cosine over pgvector HNSW.

**Results**

| k | answer-chunk (right file + `must_contain`) | document (right file present) |
|---|---|---|
| 5 | 6/19 = **0.32** | 12/19 = **0.63** |
| 10 | 6/19 = 0.32 | 12/19 = 0.63 |

Answer-chunk misses @5: `q01, q02, q03, q05, q07, q08, q09, q14, q15, q16, q20, q21, q22`

**Reading (this is a work order for Phases 3–4, not just a score)**
- `@5 == @10` ⇒ not a depth problem. Two separate issues:
  - **0.32 → 0.63 gap** = right *document*, wrong *chunk* → **reranking (Phase 4)**.
  - **0.63 ceiling** = 7 docs never reach top-10 (crowded water topic, floral strips, short synthetic notes) → **hybrid keyword search (Phase 3)** + stronger embedder.
- Measured on MiniLM-384 (fast baseline). Upgrade to e5-large / BGE-M3 = config change + re-index (a later, measurable improvement).

**Reproduce**
```bash
docker compose up -d
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/001_init.sql
AGROTECA_LOCAL_MODE=1 uv run python -m agroteca.ingest.run
uv run python eval/retrieval_at_k.py --k 5
```
