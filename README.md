# 🌱 Agroteca

**An eval-first, bilingual (ES/EN) Retrieval-Augmented Generation system over agricultural documents — built measurement-first, with a governed, provenance-aware corpus.**

> **Status: Phases 1–7 complete.** Governed corpus → ingestion → hybrid retrieval → cross-encoder reranking → grounded, cited, **abstaining** generation → a streaming FastAPI service → a polished bilingual **web app** with a retrieval-transparency drawer. Only public deployment remains.
>
> **The number that matters:** retrieval@5 **0.35 (dense) → 0.45 (hybrid) → 0.75 (reranked)** — every step measured against a golden set built *before* the retriever existed (20 answerable questions).

**Stack:** Python · FastAPI · Postgres + pgvector · full-text search · fastembed (ONNX) · a multilingual cross-encoder reranker · Ollama (local LLM) · `uv` · Docker · a self-contained streaming web front end.

---

## Highlights

- **Eval-first, and it earned its keep.** The 22-question golden set was written *before* the retriever, so every change is measured, not guessed — and it once flagged a "hallucination" that turned out to be a **mislabeled ground-truth answer** (I fixed the label, not the model).
- **A measured retrieval cascade.** dense → hybrid (Reciprocal Rank Fusion) → cross-encoder reranking, each stage's gain proven on the golden set: **retrieval@5 0.35 → 0.45 → 0.75**, six questions fixed with **zero regressions**.
- **Grounded and cited — or an honest "I don't know."** Generation answers only from retrieved context, cites its sources, and abstains (with a canonical, machine-checkable phrase) when the corpus can't answer.
- **A transparency-first web app.** A bilingual, streaming UI where every answer opens a drawer exposing the exact retrieved chunks, their cross-encoder relevance scores, and governance tiers — plus a retrieval-vs-generation latency breakdown. *Show your work.*
- **Provenance as a first-class concern.** A four-tier governed corpus (open / local-only / synthetic / distractor) where the tier controls where a document may appear; copyrighted material is confined to a local-only mode and never shipped publicly.
- **Local-first, measured model choices.** A MiniLM embedder and a 3B local LLM, each chosen by *measured* throughput and quality — not by hype.

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
| Dense (semantic baseline) | 0.35 | 0.65 |
| + Lexical, fused with RRF (hybrid) | 0.45 | 0.65 |
| **+ Cross-encoder reranking** | **0.75** | **0.90** |

Reranking fixed **six** questions the earlier stages missed, with **zero regressions**. On generation, the system **abstains on the corpus's genuinely unanswerable questions** rather than fabricating, and a **3B local model matched a 7B on abstention at ~3× lower latency** — chosen by measurement, not assumption.

> An honest footnote (because honesty is the brand): lexical search *alone* scored 0.50 at answer@5 vs the 0.45 hybrid — but that's a **single question** at n=20, within noise. The evidence that actually matters is `doc@5`: **0.85 lexical vs 0.65 hybrid, a four-document gap** — equal-weight RRF was diluting the stronger retriever's recall. That's why the project didn't stop at hybrid; reranking is what decisively won. The full story is in [`notes.md`](notes.md) and [`results/`](results/).

### Latency & what a public deploy actually serves — measured

Two numbers an interviewer asks for, both measured (`eval/compare_retrievers.py`), not assumed:

- **Latency (CPU, per stage).** Bi-encoder **retrieval is production-grade — p95 ≤ 250 ms** (dense 106 ms, hybrid 249 ms). The **cross-encoder reranker is the bottleneck at ~25 s p50 on CPU** — the same free/local-inference tax as generation, and sub-second on a GPU or hosted reranker with *no code change* (it only ever scores the ~20-candidate pool). Full table + levers, including a hybrid-only "fast mode": [`results/latency.md`](results/latency.md).
- **The deployable corpus is smaller than the index.** ~30% of chunks are copyrighted (`local` tier, 11 docs) and can't be served publicly. On the **open + synthetic** corpus a stranger actually gets, the final cascade scores **0.65 answer@5 / 0.75 doc@5** (vs 0.75 / 0.90 on the full index) — lower *only* because two questions (q06, q12) are answered solely by docs a public server can't ship. Same cascade, zero regressions; see the `deploy` row in [`results.csv`](eval/results.csv).

---

## The web app

A local model on CPU is slow (~minutes per answer), so the front end is built to make a slow, honest system *feel* alive and trustworthy:

- **Watch it grow** — a seed→sprout→flower→fruit reel plays during retrieval, then the answer **streams in token by token**. The wait reads as progress, never a freeze.
- **Radical transparency** — every answer carries a one-click **"Sources & retrieval"** drawer: the exact chunks it grounded on, each with a governance-tier badge, a cross-encoder relevance bar, and a snippet — plus a retrieval-vs-generation latency breakdown. *The answer isn't magic; here's the evidence.*
- **Honest by design** — when the corpus can't answer, the card becomes a distinct **empty-basket** state, never a fabricated guess.
- **Bilingual** — a leaf ES/EN toggle localizes the whole UI (and the example questions); the model already answers in the question's language.
- **Product touches** — light/dark themes, shareable `?q=` links, session history, keyboard shortcuts, a live health dot, a corpus-stats ribbon, live tokens/sec, a real Stop button, copy/export, and 👍/👎 feedback logged server-side.

The default view stays a calm search box — all of that power reveals only on demand.

```bash
# launch the web app (needs Postgres + pgvector and a local Ollama model)
uv run uvicorn agroteca.api:app --reload
# then open http://127.0.0.1:8000  ·  auto-generated API docs at /docs
```

> _Screenshots / a short demo GIF go here — the fastest way to convey the UI to someone skimming the repo._

---

## Roadmap

| Phase | Deliverable | Ship criterion | Status |
|---|---|---|---|
| 1 | Golden set + governed corpus + validator | validator passes; corpus tiered & text-verified | ✅ **done** |
| 2 | Ingestion: normalize → chunk → embed → pgvector + FTS | corpus indexed (10,330 chunks); `retrieval@5` baseline measured | ✅ **done** → 0.35 |
| 3 | Hybrid retrieval (dense + lexical) fused with RRF | the number moves vs the dense baseline | ✅ **done** → 0.45 |
| 4 | Cross-encoder reranking (precision stage) | rerank@5 beats hybrid on the golden set | ✅ **done** → 0.75 |
| 5 | Grounded, **cited**, **abstaining** generation (local LLM) | answers are cited; no-answer questions abstain | ✅ **done** |
| 6 | Serving: FastAPI + token streaming (NDJSON evidence stream) | a request streams a cited answer end-to-end | ✅ **done** |
| 7 | Web app: bilingual streaming UI + retrieval-transparency drawer | grounded/cited answer *or* honest abstention, evidence one click away | ✅ **done** |
| 8 | Public deployment + write-up | a stranger opens a URL and gets a cited answer | ⏳ planned |

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
  generate.py             # ground / cite / abstain generation (+ NDJSON streaming)
  api.py                  # FastAPI service: /ask/stream, /stats, /feedback, /health
  static/index.html       # the self-contained bilingual streaming web app
stream_client.py          # tiny terminal client for the streaming endpoint
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

# 4. the full web app  (needs Postgres + pgvector + Ollama)
uv run uvicorn agroteca.api:app --reload   # then open http://127.0.0.1:8000
```

---

## Licensing & provenance

Source documents retain their original licenses. Only open-license / public-domain / government material is served by any public deployment; copyrighted material is confined to the local-only tier and is never redistributed. Chilean official legal texts are public domain (Ley 17.336 art. 88). See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

---

*Built as a portfolio project. The interesting parts are the decisions, not the line count.*
