# ADR 0001: Keep 512-token chunking despite embedder truncation at 128 tokens

**Status:** Accepted

## Context

The shipped embedder is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(`src/agroteca/config.py:35`), whose `max_seq_length` is 128 tokens. Chunks are configured
at `chunk_tokens = 512` (`config.py:98`, ~2048 characters via `chars_per_token = 4`). Tokens
past position 128 are dropped before the embedder's forward pass, so the dense vector encodes
only the first ~128 tokens of every chunk — the tail is still present for lexical FTS and for
the cross-encoder reranker's full-text scoring, but not in the semantic vector used for the
initial dense search (`config.py:88-97`).

This was caught empirically, not assumed: a cosine A/B test showed chunks sharing their first
~128 tokens but with different tails embed identically (cosine 1.0), a flat-zero control that
proves the truncation (`docs/masterclass.md` §4.13, `config.py:88-97`).

## Decision

Kept `chunk_tokens = 512` for both retrieval candidate generation and generation context.
Did **not** rechunk to 128 tokens to eliminate the dense-embedding truncation.

## Evidence

A controlled probe (`eval/embedder_pilot.py`, recorded in `eval/results.csv` row
`"truncation probe: MiniLM re-chunked to 128-tok (NOT shipped)"`) re-chunked the corpus to
128 tokens — small enough that no chunk exceeds the embedder's window — and re-measured the
full pipeline over the golden set (n=20):

| metric (golden set, n=20) | shipped: 512-tok | probe: 128-tok |
|---|---|---|
| dense-stage `document@5` (pre-rerank) | 0.65 | **0.90** |
| final reranked `answer@5` | **0.75** | 0.60 |
| final reranked `answer@1` | **0.75** | 0.45 |
| final reranked `mrr@5` | **0.75** | 0.52 |
| final reranked `document@5` (post-rerank) | 0.90 | 0.90 |

Eliminating the truncation raised the dense stage's own recall substantially (0.65→0.90) — the
embedder can finally "see" more of the chunks that actually contain the answer. But after the
cross-encoder reranks the pool, the *final* answer@5 fell 0.75→0.60 (and answer@1, mrr@5 fell
with it), even though final document@5 stayed at 0.90 in both configs. Per `eval/results.csv`'s
own note: smaller chunks "dilute the exact answer chunk and starve the reranker of context" —
the right document region is still reachable either way, but a 128-token chunk is less likely
to be a single, complete, self-contained unit the reranker (and, by the same mechanism, the
final LLM) can correctly judge or answer from.

Note on scope: `answer@5`/`answer@1`/`mrr@5` here are retrieval-stage metrics (does a top-k
chunk contain the reference answer text), not a generation-output eval — this experiment did
not re-run `eval/eval_generation.py`. The generation-context implication below is this ADR's
reasoned interpretation of the mechanism, not itself a directly measured generation-quality
result.

Rejected the 128-token rechunk on this evidence: a fix for the dense stage that measurably hurt
the number that matters (final answer quality). 512 was already near-optimal for this cascade.

## Insight

Retrieval granularity and the generation context window are coupled today by using one chunk
size for both jobs. They don't have to be — a chunk small enough to embed cleanly within a
128-token dense encoder is not the same size as a chunk large enough to give a reranker or an
LLM enough self-contained context to judge or answer from. This is a retrieval-precision vs.
context-completeness tradeoff, and it's currently resolved by picking one chunk size that's a
compromise for both jobs rather than serving each job at its own natural size.

## Rejected alternative

Reconstructing a larger context window at read time from the `char_start`/`char_end` offsets
already stored per chunk, by re-extracting text from the original source files. Rejected as the
implementation path for a future fix: the database never persists the full assembled document
text (only each chunk's own `text`), so this would depend on the raw PDFs/markdown under
`settings.raw_dir`/`settings.synthetic_dir` still existing at read time, with deterministic
re-extraction — a re-extraction pipeline the app doesn't otherwise need and that could silently
drift from what was actually ingested.

## Next step (not built — documented intent only)

Decouple retrieval granularity from generation context: keep ranking on the existing small
chunks (no change to dense/lexical/hybrid/rerank), but at generation time, expand each
retrieved chunk to include its sibling chunks before building the LLM's context block.

This needs no schema change. `chunks` already carries `doc_id` and `chunk_index`
(`migrations/001_init.sql:16-30`) — a stable per-document ordering populated on every ingest
(`src/agroteca/ingest/run.py`). The ingestion spec already named this intent for `chunk_index`:
"position within the document — for ordering and context expansion"
(`docs/phase2_ingestion_spec.md:61`). The mechanism would be: for each retrieved chunk, `SELECT`
its neighbors by `doc_id` + `chunk_index` and concatenate their already-stored `text` in order
to build a wider generation window, while the ranking signal (dense + lexical + rerank score)
stays computed on the original small chunk.

As of this writing, `chunk_index`, `char_start`, and `char_end` are write-only — populated on
every insert (`src/agroteca/ingest/store.py`) but never `SELECT`ed by any retrieval or
generation code (`src/agroteca/retrieve/`, `src/agroteca/generate.py`, `src/agroteca/api.py`).
The capability is schema-ready, not code-ready.
