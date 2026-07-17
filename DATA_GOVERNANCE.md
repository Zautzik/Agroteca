# Data Governance — Agroteca corpus

Agroteca answers agronomy questions from a curated document corpus. Because it runs both as a **public demo** and as a **local tool on a working farm**, and because its sources range from open government publications to copyrighted books, the corpus is governed as **four explicit tiers**. This document records what each tier is, why it exists, and the two principles that decide what may enter the answerable corpus.

## Two principles

**1. Idea / expression.** Copyright protects the *expression* of a work — its wording, structure, and illustrations — not the *facts, ideas, or methods* it describes. Agroteca may freely state facts and techniques ("a soilless raised-bed mix uses equal parts peat, vermiculite, and compost"); it never reproduces or closely paraphrases protected expression into the corpus.

**2. Citation integrity.** A RAG's value is a faithful citation to a real source. The system therefore never contains a synthesized paraphrase of a source *attributed to that source* — doing so would fabricate attribution, the exact hallucination failure the project's eval-first design exists to prevent. Every answerable document is either a real source we may lawfully serve, or an original, clearly-labeled synthetic note cited **as** a synthetic note.

## The four tiers

| Tier | Manifest | Location | Loaded when | Citation behavior |
|---|---|---|---|---|
| **Open / deployable** | `data/manifest.csv` | `data/raw/` | always | real source, fully redeployable |
| **Local-only (copyright)** | `data/manifest_local.csv` | `data/raw/` | `LOCAL_MODE` only | real source; never shipped in a public deploy |
| **Synthetic reference** | `data/manifest_synthetic.csv` | `data/synthetic/` | always | cited as an original, grounded synthetic note |
| **Distractor** | `data/manifest_distractor.csv` | `data/raw/` | eval only | never a valid answer source |

- **Open** — FAO, USDA, SARE/ATTRA, CGIAR, Chilean government (INIA/CIREN/ODEPA), public-domain classics, and freely-distributed works. Chilean official legal texts are public domain (Ley 17.336 art. 88). Safe to serve publicly.
- **Local-only** — copyrighted commercial books (Fukuoka, Mollison, Stamets, Fortier, Bartholomew, Holzer, …). Present because Agroteca is also the owner's private farm tool; held from the owner's own copies and **excluded from any public deployment**, not redistributed.
- **Synthetic reference** — short, original, human-verified notes stating non-copyrightable facts, used **only** where a topic's knowledge would otherwise be available solely from a local-only source. Each note carries provenance (`grounded_in`) and is labeled synthetic. It is a last resort for coverage gaps, not a substitute for real sources — where an open source exists, we cite the open source instead.
- **Distractor** — deliberately off-topic hard-negatives (e.g., an organic-chemistry textbook that shares the token "organic" with "organic farming"). Never answerable; used to measure retrieval precision and abstention.

## Synthetic-note policy

A synthetic reference note **MUST**: state facts/techniques in original wording; be grounded in and cite an open source or general knowledge (`grounded_in`); be organized by concept, never as a re-expression of one book's structure or worldview; be labeled `synthetic: true` with `role: synthetic_reference`.

It **MUST NOT**: reproduce or closely paraphrase protected prose; carry a source's illustrations, specific anecdotes, or distinctive selection/arrangement; or be attributed to a copyrighted author as if quoted.

Generation method: notes are drafted (LLM-assisted), then human-verified against open sources before entering the tier. Provenance is recorded per note in its YAML front-matter.

## Image-only sources

Some scanned PDFs have no text layer. Policy: **recover, don't invent.** Government scans (e.g., INIA books N°11, N°19) are OCR'd to obtain their *actual* text. We never synthesize content for a document we cannot read — that would be fabrication, not rescue.

## How the app uses the tiers
- **Deployed demo** ingests `manifest.csv` + `manifest_synthetic.csv`.
- **Local (`LOCAL_MODE=1`)** additionally ingests `manifest_local.csv`.
- **Evaluation** loads `manifest_distractor.csv` as non-answerable noise; the eval runner skips questions whose only source is local-only when scoring a *deployed* configuration.

## Honest limitations
- The corpus is intentionally small and curated, not exhaustive; `data/raw/` also holds unmanifested off-topic files that are deliberately excluded.
- Synthetic notes are a coverage aid, not authorities; they are labeled so a reader (and the ranker) can weight them accordingly.
- License classification reflects best-effort provenance judgment; ambiguous items default to the more restrictive tier.
