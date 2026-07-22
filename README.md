# 🌱 Agroteca

**An eval-first, bilingual (ES/EN) Retrieval-Augmented Generation system over agricultural documents — built measurement-first, with a governed, provenance-aware corpus.**

> Status: **Phase 1 complete** (evaluation foundation + governed corpus). Retrieval pipeline is specified and in progress. This README is honest about what is built vs. planned — see [Roadmap](#roadmap).

---

## What it is

Agroteca answers agronomy questions — *"which alfalfa variety was used in the INIA trials?"*, *"how much rainwater can a roof harvest?"* — strictly from a curated document corpus, in Spanish or English, with citations, and with the discipline to say **"I don't know"** when the corpus can't answer.

The domain corpus is anchored on **Chile's INIA** agricultural publications, extended with **FAO**, university extension, CGIAR, and public-domain sources, plus the relevant slice of **Chilean agricultural and food-safety law**.

## Why it's built this way

Most RAG projects build the pipeline, eyeball a few demo questions, and call it done — with no way to know whether any change is an improvement. Agroteca inverts that: **the evaluation set was built before the retrieval system**, so every future change is measured, not guessed. This is the single most load-bearing decision in the project. Details in [`notes.md`](notes.md).

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

The heart of the project is a **golden set** of 22 hand-verified question/answer pairs (`eval/golden_set.jsonl`), deliberately stratified to stress specific parts of the future system:

- **Factual lookup** — basic retrieval + extraction
- **Exact-term** (variety codes, cultivar names, species) — the justification for hybrid keyword+semantic search
- **Synthesis** — multi-sentence reasoning
- **No-answer** — measures abstention / hallucination resistance
- **Cross-lingual** (ES question → EN source, and vice-versa) — forces a multilingual embedder

Every answerable question carries a `must_contain` string verified **verbatim against machine-extracted text** (not the source PDF read by eye — a distinction that caught real OCR line-break bugs).

```bash
uv run eval/validate_golden.py
# OK: 22 questions validated against 71 answerable manifest entries (open + local-only + synthetic tiers)
```

---

## Roadmap

| Phase | Deliverable | Ship criterion | Status |
|---|---|---|---|
| 1 | Golden set + governed corpus + validator | validator passes; corpus tiered & text-verified | ✅ **done** |
| 2 | Ingestion: normalize → chunk → embed (BGE-M3) → pgvector + BM25 | corpus indexed; `retrieval@5` measured on the golden set | 📝 **spec'd** ([docs/phase2_ingestion_spec.md](docs/phase2_ingestion_spec.md)) |
| 3 | Hybrid retrieval (dense + lexical) fused with Reciprocal Rank Fusion | an exact-term query that dense-only misses now retrieves | ⏳ planned |
| 4 | Reranking + grounded, **cited**, abstaining generation | answers are cited; no-answer questions correctly abstain | ⏳ planned |
| 5 | FastAPI streaming endpoint + cache + latency budget | measured p95 on golden-set queries | ⏳ planned |
| 6 | Frontend + deploy + write-up | a stranger can open a URL and get a cited answer | ⏳ planned |

**Target metrics** (goals, not yet measured): retrieval@5 tracked baseline → final across the build; sub-second retrieval p95. These will be reported with real before/after numbers as the pipeline lands — that's the reason the eval was built first.

---

## Design decisions (and why)

| Choice | Rationale |
|---|---|
| **BGE-M3** embeddings | multilingual (the cross-lingual questions would fail on an English-only model) and emits dense **+** sparse vectors, so hybrid search comes from one model |
| **Postgres + pgvector** | metadata filtering, keyword search, and vector search in one store; production-credible; tiers enforced in SQL |
| **Hybrid + Reciprocal Rank Fusion** | dense search misses exact codes (`WL-323`); lexical search misses paraphrase; fusion gets both |
| **Cross-encoder reranker** | sharpens the top candidates before they reach the LLM — retrieval quality is the ceiling on answer quality |
| **Local-first models** | zero inference cost for embeddings/reranking; reproducible |
| **Deterministic + RAGAS eval** | deterministic `must_contain`/retrieval@k checks run cheaply in CI; RAGAS (faithfulness, context precision/recall) adds LLM-judged depth |

---

## Repository layout

```
data/
  manifest.csv            # open / deployable tier
  manifest_local.csv      # copyrighted, local-only tier
  manifest_synthetic.csv  # grounded synthetic reference notes
  manifest_distractor.csv # hard-negatives
  synthetic/              # the synthetic reference notes (*.md)
  raw/                    # source PDFs (gitignored)
eval/
  golden_set.jsonl        # 22 hand-verified Q/A pairs
  validate_golden.py      # structural + tier-membership validator
docs/
  phase2_ingestion_spec.md# ingestion spec + teaching guide
DATA_GOVERNANCE.md        # the four-tier provenance/licensing policy
notes.md                  # build journal (the "why", with war stories)
```

## Running it

Python is managed with **[uv](https://docs.astral.sh/uv/)** (Python 3.12). Today, the evaluation foundation runs:

```bash
uv run eval/validate_golden.py
```

The ingestion pipeline (Phase 2) and everything downstream are specified in `docs/` and land next.

---

## Licensing & provenance

Source documents retain their original licenses. Only open-license / public-domain / government material is served by any public deployment; copyrighted material is confined to the local-only tier and is never redistributed. Chilean official legal texts are public domain (Ley 17.336 art. 88). See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

---

*Built as a portfolio project. The interesting parts are the decisions, not the line count.*
