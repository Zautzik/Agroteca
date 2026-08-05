# 🌱 Agroteca

**An eval-first, bilingual (ES/EN) Retrieval-Augmented Generation system over agricultural documents — built measurement-first, with a governed, provenance-aware corpus.**

> **Status: Phases 1–5 complete.** Governed corpus → ingestion → hybrid retrieval → cross-encoder reranking → grounded, cited, **abstaining** generation.
>
> **The number that matters:** retrieval@5 **0.32 (dense) → 0.42 (hybrid) → 0.74 (reranked)** — every step measured against a golden set built *before* the retriever existed. Serving (API + streaming) and deployment are the remaining roadmap.

---

## What it is

Agroteca answers agronomy questions — *"which alfalfa variety was used in the INIA trials?"*, *"how much rainwater can a roof harvest?"* — **strictly from a curated document corpus**, in Spanish or English, **with citations**, and with the discipline to say **"I don't know"** when the corpus can't answer.

The domain corpus is anchored on **Chile's INIA** agricultural publications, extended with **FAO**, university extension, CGIAR, and public-domain sources, plus the relevant slice of **Chilean agricultural and food-safety law**.

## Why it's built this way

Most RAG projects build the pipeline, eyeball a few demo questions, and call it done — with no way to know whether any change is an improvement. Agroteca inverts that: **the evaluation set was built before the retrieval system**, so every change is measured, not guessed. This is the single most load-bearing decision in the project — and it has already paid for itself (see [The receipts](#the-receipts)). The full story is in [`notes.md`](notes.md).

---

## Architecture

Two machines: an **offline** indexer (run once) and an **online** query path (run per question).

```
 OFFLINE ─ build the index ─────────────┐    ONLINE ─ answer a question ────────────────────────┐
                                         │                                                       │
 PDFs → extract → normalize → chunk →    │  question → embed → ┌ dense  (pgvector) ┐             │
 embed → store (pgvector + tsvector) ────┼──────────────────── ┤   RRF fusion      ├→ rerank →   │
                                         │                     └ lexical (FTS)     ┘   top-k     │
                                         │                                                ↓       │
                                         │                   local LLM:  GROUND · CITE · ABSTAIN  │
 ────────────────────────────────────────┘ ──────────────────────────────────────────────────────┘
```

Each retrieval stage exists to fix the previous stage's blind spot: dense understands paraphrase but misses exact codes (`WL-323`); lexical nails exact codes but misses paraphrase; hybrid gets both; the cross-encoder reranker sharpens precision before the answer reaches the LLM.

---

## The corpus: four governed tiers

Provenance and licensing are treated as first-class. Every document lives in exactly one tier, and the tier controls where it is allowed to appear. Full policy in [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

| Tier | Manifest | Loads when | Role |
|---|---|---|---|
| **Open / deployable** | `data/manifest.csv` | always | open-license / gov / public-domain — real citations, safe to deploy publicly |
| **Local-only** | `data/manifest_local.csv` | `LOCAL_MODE` only | copyrighted books — available to the local farm tool, never shipped in a public demo |
| **Synthetic reference** | `data/manifest_synthetic.csv` | always | original, human-verified concept notes grounded in open sources; cited *as* synthetic |
| **Distractor** | `data/manifest_distractor.csv` | eval only | deliberate off-topic hard-negatives; never a valid answer source |

Two principles decide what may enter the answerable corpus: the **idea/expression** distinction (facts are free; copyrighted expression is not) and **citation integrity** (the system never attributes synthesized text to a real source).

---

## Evaluation

The heart of the project is a **golden set** of 22 hand-verified question/answer pairs (`eval/golden_set.jsonl`), deliberately stratified to stress specific parts of the system:

- **Factual lookup** — basic retrieval + extraction
- **Exact-term** (variety codes, cultivar names, species) — the justification for hybrid keyword+semantic search
- **Synthesis** — multi-sentence reasoning
- **No-answer** — measures abstention / hallucination resistance
- **Cross-lingual** (ES question → EN source, and vice-versa) — forces a multilingual embedder

Every answerable question carries a `must_contain` string verified **verbatim against machine-extracted text** (not the source PDF read by eye — a distinction that caught real OCR line-break bugs). Retrieval is scored with a **dual metric** — *answer-chunk@k* (did the exact answer chunk surface?) vs *document@k* (did the right document surface?) — because the gap between them separates a chunking problem from a retrieval problem.

```bash
uv run python eval/validate_golden.py
# OK: 22 questions validated against 71 answerable manifest entries (open + local-only + synthetic tiers)
```

---

## The receipts

Measured on the answerable golden set (k=5), one variable at a time:

| Stage | answer-chunk@5 | document@5 |
|---|---|---|
| Dense (semantic baseline) | 0.32 | 0.63 |
| + Lexical, fused with RRF (hybrid) | 0.42 | 0.63 |
| **+ Cross-encoder reranking** | **0.74** | **0.89** |

Reranking fixed **six** questions the earlier stages missed, with **zero regressions**. On generation, the system **abstains on the corpus's genuinely unanswerable questions** rather than fabricating, and a **3B local model matched a 7B on abstention at ~3× lower latency** — chosen by measurement, not assumption.

> An honest footnote (because honesty is the brand): lexical search *alone* scored 0.47 at k=5 — higher than the 0.42 hybrid, an RRF nuance at small k. It's exactly why the project didn't stop at hybrid; reranking is what decisively won. The full story is in [`notes.md`](notes.md) and [`results/`](results/).

---

## Roadmap

| Phase | Deliverable | Ship criterion | Status |
|---|---|---|---|
| 1 | Golden set + governed corpus + validator | validator passes; corpus tiered & text-verified | ✅ **done** |
| 2 | Ingestion: normalize → chunk → embed → pgvector + FTS | corpus indexed (10,330 chunks); `retrieval@5` baseline measured | ✅ **done** → 0.32 |
| 3 | Hybrid retrieval (dense + lexical) fused with RRF | the number moves vs the dense baseline | ✅ **done** → 0.42 |
| 4 | Cross-encoder reranking (precision stage) | rerank@5 beats hybrid on the golden set | ✅ **done** → 0.74 |
| 5 | Grounded, **cited**, **abstaining** generation (local LLM) | answers are cited; no-answer questions abstain | ✅ **done** |
| 6 | Serving: FastAPI + token streaming + measured p95 | a request streams a cited answer end-to-end | ⏳ planned |
| 7 | Frontend + deploy + write-up | a stranger opens a URL and gets a cited answer | ⏳ planned |

---

## Design decisions (and why)

| Choice | Rationale |
|---|---|
| **Multilingual MiniLM embeddings** (measured) | cross-lingual questions rule out an English-only model; MiniLM-384 was chosen over e5-large / BGE-M3 by **measured** CPU throughput — with the heavier models documented as an upgrade path |
| **Postgres + pgvector** | metadata filtering, keyword search, and vector search in one store; production-credible; tiers enforced in SQL |
| **Hybrid + Reciprocal Rank Fusion** | dense search misses exact codes (`WL-323`); lexical search misses paraphrase; fusing by *rank* (not score) sidesteps the incomparable score scales |
| **Cross-encoder reranker** | reads each (query, chunk) pair *together* to sharpen the top candidates before the LLM — retrieval quality is the ceiling on answer quality |
| **Local LLM · ground / cite / abstain** | grounded generation with citations and a canonical abstention phrase; rules live in the **system** role to resist prompt injection from retrieved chunks; a local 3B model keeps inference free, offline, and copyrighted chunks on-device |
| **Deterministic eval, RAGAS-ready** | deterministic `must_contain` / retrieval@k checks run cheaply and gate every change; semantic faithfulness (RAGAS) is the documented next layer |

---

## Repository layout

```
data/
  manifest*.csv           # the four governed tiers (open / local / synthetic / distractor)
  synthetic/              # grounded synthetic reference notes (*.md)
  raw/                    # source PDFs (gitignored)
eval/
  golden_set.jsonl        # 22 hand-verified Q/A pairs
  validate_golden.py      # structural + tier-membership validator
  compare_retrievers.py   # dense vs lexical vs hybrid vs rerank, on the golden set
  eval_generation.py      # abstention + citation + latency on generated answers
src/agroteca/
  ingest/                 # extract · normalize · chunk · embed · store
  retrieve/               # dense · lexical · fusion (RRF) · hybrid · rerank
  generate.py             # ground / cite / abstain generation over reranked chunks
migrations/               # pgvector + full-text schema
docs/                     # ingestion spec + teaching notes
results/                  # measured before/after for each phase
DATA_GOVERNANCE.md        # the four-tier provenance/licensing policy
notes.md                  # build journal (the "why", with war stories)
```

## Running it

Python is managed with **[uv](https://docs.astral.sh/uv/)** (Python 3.12). The retrieval and generation stack needs **Postgres + pgvector** (via Docker) and a local **[Ollama](https://ollama.com)** model for generation.

```bash
# 1. evaluation foundation (no services required)
uv run python eval/validate_golden.py

# 2. compare retrievers on the golden set  (needs Postgres + pgvector, corpus ingested)
uv run python eval/compare_retrievers.py --k 5

# 3. a grounded, cited answer from the real corpus  (needs Postgres + Ollama)
uv run python src/agroteca/generate.py
```

---

## Licensing & provenance

Source documents retain their original licenses. Only open-license / public-domain / government material is served by any public deployment; copyrighted material is confined to the local-only tier and is never redistributed. Chilean official legal texts are public domain (Ley 17.336 art. 88). See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

---

*Built as a portfolio project. The interesting parts are the decisions, not the line count.*
