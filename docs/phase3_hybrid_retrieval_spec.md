# Phase 3 — Hybrid Retrieval + Reciprocal Rank Fusion: Spec + Teaching Guide

> **Historical spec (n=19).** The baseline figures below (`0.32` / `0.63`) are the pre-correction Phase-2 diagnosis that motivated this design. After the Phase-5 q18 relabel the answerable set is **n=20** — current retrieval figures live in [`eval/results.csv`](../eval/results.csv). The *diagnosis* here (chunk-vs-document gap, recall ceiling) is what drove the design and is unchanged.

Phase 2 gave us a baseline and, more valuably, a diagnosis: `retrieval@5 = 0.32` (answer-chunk) /
`0.63` (document). Two failure modes, written in the misses:

- the **0.32 → 0.63 gap** = the right *document* is retrieved but the right *chunk* isn't (a
  reranking problem, Phase 4);
- the **0.63 ceiling** = seven docs never reach top-10 at all (exact-term + crowded-topic recall).

Phase 3 attacks the second one. It is the smallest possible change that should move the number, and
it lets us *attribute* the movement to one idea: **add a second, lexical retriever and fuse it with
the dense one.** No new model, no new data — just a better way to search what we already indexed.

---

## 0. Mental model: why one retriever is never enough

Dense (semantic) search and lexical (keyword) search fail in *opposite* ways:

| | Dense / semantic (Phase 2) | Lexical / keyword (Phase 3) |
|---|---|---|
| Great at | meaning, paraphrase, cross-lingual | exact tokens: codes, names, numbers |
| Blind to | exact strings it never "means" close | synonyms, paraphrase, translation |
| Our failing question | q07 "30-inch bed" (a *number*, not a vibe) | (dense already handles the fuzzy ones) |

The whole point: **a variety code like `WL-323` or an exact phrase like "30-inch bed" has weak
semantic signal but perfect lexical signal.** Dense embeddings smear it into "alfalfa varieties,
generally"; a keyword index matches it exactly. You don't choose between the two — you run both and
combine, so each covers the other's blind spot. That combination is *hybrid retrieval*, and it's the
single highest-value retrieval upgrade you can make.

---

## 1. Lexical retrieval in Postgres (no new infrastructure)

We already populated a `tsvector` column in Phase 2. That's Postgres full-text search — the lexical
half — sitting in the same table as the vectors. No Elasticsearch, no second datastore.

**Three concepts:**
- **`tsvector`** — a document turned into its searchable lexemes (tokens), stored per chunk, indexed
  with GIN for speed.
- **`tsquery`** — a query turned into lexemes to match against a `tsvector`. Build it from user text
  with `websearch_to_tsquery(config, text)` (forgiving; understands quotes/AND/OR).
- **`ts_rank_cd(tsv, query)`** — a relevance score (a BM25-family, cover-density ranking) used to
  order matches. Higher = better.

**The one real decision — which text-search config?** Postgres configs are *language-specific*
(the `spanish` config stems `variedades → variedad`, removes stopwords; `english` stems differently).
In Phase 2 we indexed each chunk in its own language. But a single query can only be parsed with
*one* config, and our corpus is bilingual — a `spanish`-parsed query won't match `english`-stemmed
chunks, and vice-versa.

**Decision: rebuild `tsv` with the language-agnostic `simple` config.** `simple` lowercases and
tokenizes but does **not** stem or drop stopwords. That is exactly right for the job hybrid asks of
lexical search: **match exact tokens** — `WL-323`, `Franquette`, `Rodolia`, `30` — consistently
across ES *and* EN. We lose morphological recall (that's what the dense side is *for*), and we gain
cross-lingual exact-term matching and a single, coherent query path. This is `migrations/002`.

```sql
-- (reference) migrations/002_tsv_simple.sql — rebuild the keyword index, no re-embedding needed
UPDATE chunks SET tsv = to_tsvector('simple', text);
```

```sql
-- (reference) the lexical query
SELECT c.chunk_id, d.source_file, c.text
FROM chunks c JOIN documents d ON c.doc_id = d.doc_id,
     websearch_to_tsquery('simple', %s) AS q
WHERE c.tsv @@ q
ORDER BY ts_rank_cd(c.tsv, q) DESC
LIMIT %s;
```

> **Teaching note.** Notice we changed *how we search*, not *what we stored's vectors*. Rebuilding
> `tsv` is a fast in-place `UPDATE` (seconds) — it never touches the 384-dim embeddings. Keeping the
> lexical and dense signals in one table is why this is cheap.

---

## 2. Fusion: combining two ranked lists the right way

Now each retriever returns its own top-N ranked list. How do you merge them into one?

**The naive way (don't):** add the dense cosine score and the lexical `ts_rank` score. This fails
because the two scores live on **incommensurable scales** — cosine ∈ roughly [0,1], `ts_rank_cd` is
unbounded and differently distributed. Summing them lets one retriever silently dominate, and you'd
have to hand-tune a normalization that breaks on the next corpus.

**The right way — Reciprocal Rank Fusion (RRF):** ignore the scores entirely; fuse by **rank
position**. For a chunk appearing at rank `r` in a retriever's list, it earns `1 / (k + r)`. Sum
those contributions across retrievers; sort by the sum.

```
RRF_score(chunk) = Σ_over_retrievers  1 / (k + rank_in_that_retriever)     # k ≈ 60, rank starts at 1
```

Why RRF is the standard:
- **Scale-free.** It never compares a cosine to a `ts_rank`; only positions. So no normalization, no
  per-corpus tuning.
- **Rewards agreement.** A chunk both retrievers rank highly gets two big contributions and floats to
  the top — exactly the consensus you want.
- **Robust to one retriever whiffing.** If dense misses `WL-323` entirely, the lexical rank-1 still
  injects it near the top via `1/(k+1)`.
- **The `k` constant** (default 60) damps the influence of very top ranks so a single retriever's #1
  can't unilaterally win; larger `k` = flatter, more democratic fusion.

That's the entire algorithm. ~10 lines of Python, no training, and it's what production hybrid search
actually uses.

---

## 3. Module design

A small `retrieve/` package, one responsibility per file (mirrors the `ingest/` split):

```
src/agroteca/retrieve/
  dense.py    # dense_search(conn, query, n, tiers) -> [(chunk_id, source_file, text), ...]
  lexical.py  # lexical_search(conn, query, n, tiers) -> same shape
  fusion.py   # reciprocal_rank_fusion([list_a, list_b], k, top) -> fused list
  hybrid.py   # hybrid_search(conn, query, k, n, tiers) = fuse(dense_n, lexical_n)[:k]
```

Each retriever returns the **same row shape** so fusion is retriever-agnostic — you could add a third
signal later (e.g. a learned sparse vector) and RRF wouldn't change.

---

## 4. Measuring — the whole reason Phase 2 came first

Reuse the golden set and the dual metric (answer-chunk + document). Run **all three** retrievers and
put them side by side:

```
uv run python eval/compare_retrievers.py --k 5
```

Expected shape of the result (the hypothesis we're testing):
- **lexical alone** will *win* the exact-term questions (q07 "30-inch", q16 floral-strip term) and
  *lose* the fuzzy/semantic ones — the mirror image of dense.
- **hybrid** should be **≥ max(dense, lexical)** on the answer-chunk metric, because RRF keeps each
  retriever's wins. If hybrid isn't ≥ dense, something is wired wrong — that's a real check.

Whatever the numbers are, they are *attributable*: the only thing that changed since the baseline is
"added lexical + RRF." That clean attribution is the dividend of eval-first, paid out here.

---

## 5. Tuning knobs (measure each; don't guess)

| Knob | Default | Effect |
|---|---|---|
| `n` (candidates per retriever) | 60 | recall vs latency; too small starves fusion |
| `rrf_k` | 60 | higher = flatter fusion (less top-rank dominance) |
| `tiers` filter | none (all ingested) | deploy vs local; distractor excluded from answerable search |
| lexical config | `simple` | exact-token matching vs stemmed recall |

---

## 6. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| lexical returns nothing | query parsed with a config that doesn't match the `tsv` config | ensure both are `simple`; rebuild via migration 002 |
| hybrid < dense | fusion bug, or n too small | assert row shapes match; raise `n`; unit-test RRF on a toy case |
| exact code still missed | FTS split the token oddly | check `to_tsvector('simple', 'WL-323')` lexemes; codes usually split into whole + parts |
| slow lexical | missing GIN index | `chunks_tsv_gin` exists from Phase 2; `ANALYZE chunks` |

---

## 7. Build checklist

- [ ] `store.py`: switch tsv build to `'simple'`; `migrations/002_tsv_simple.sql`: in-place `UPDATE`.
- [ ] Apply migration 002 (seconds; no re-embed).
- [ ] `retrieve/dense.py`, `lexical.py`, `fusion.py`, `hybrid.py`.
- [ ] `eval/compare_retrievers.py` (dense | lexical | hybrid, dual metric).
- [ ] Run at k=5; record the new number in `results/`; compare to `results/baseline.md`.
- [ ] Ship criterion: **hybrid ≥ dense on answer-chunk@5**, with the exact-term questions as the
      visible before/after example.

---

## Glossary additions
- **Lexical / sparse retrieval** — keyword matching (BM25-family); exact tokens, no semantics.
- **tsvector / tsquery / ts_rank_cd** — Postgres FTS: indexed lexemes / parsed query / relevance score.
- **`simple` config** — language-agnostic tokenization (no stemming/stopwords); best for exact terms.
- **RRF (Reciprocal Rank Fusion)** — scale-free rank-based list merging: `Σ 1/(k+rank)`.
- **Candidate generation** — the top-N each retriever proposes before fusion/reranking.

### Connects to
Phase 3 raises *recall* (get the right chunk into the candidate set at all). Phase 4 (reranking)
raises *precision* (float the answer chunk to the very top). Together they close both gaps the
baseline exposed.
