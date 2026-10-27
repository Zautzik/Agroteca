# Agroteca

[![CI](https://github.com/Zautzik/Agroteca/actions/workflows/ci.yml/badge.svg)](https://github.com/Zautzik/Agroteca/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An eval-first Retrieval-Augmented Generation (RAG) pipeline achieving 0.80 retrieval@5 accuracy via hybrid search, cross-encoder reranking and bilingual query translation.

**[Try it live →](https://agroteca.vercel.app)** A free public deployment of the same retrieval pipeline, serving only the open-license part of the corpus: measured live over HTTP at 0.70 retrieval@5, against a 0.85 ceiling for what it can legally serve. Every answer cites its source — or says it doesn't know.

Retrieval@5 (defined precisely in [Benchmark Results](#benchmark-results-measured) below) is scored against a 22-question golden set written *before* the retriever existed, over a bilingual (ES/EN) agricultural document corpus. Every stage of the cascade — dense, hybrid, reranked, translated — and the live demo itself is a measured line in [`eval/results.csv`](eval/results.csv), not an internal estimate. The same discipline extends to serving: an async connection pool sized and load-bearing under FastAPI's dependency injection, and explicit model warmup that pays CUDA/VRAM initialization once at process start instead of on a user's first request.

**Stack (reference build, self-hosted):** Python · FastAPI · Postgres + pgvector · Postgres full-text search · `sentence-transformers` / `transformers` (PyTorch, CUDA) · a multilingual cross-encoder reranker · Ollama (self-hosted LLM) · `psycopg_pool` · Docker · `uv`.

**Stack (public demo, free tiers):** Vercel (serverless FastAPI) · Neon (Postgres + pgvector) · the same embedder through ONNX Runtime · the same reranker through Jina's API · generation through Groq · parity-checked against the reference build ([`results/parity.md`](results/parity.md)).

---

## System Architecture

```
 OFFLINE ── build the index ────────────┐    ONLINE ── answer a question ─────────────────────────┐
                                         │                                                         │
 PDFs → extract → normalize → chunk →   │  question → embed → ┌ dense  (pgvector) ┐                │
 embed → store (pgvector + tsvector) ───┼──────────────────── ┤   RRF fusion      ├→ rerank(top-5) │
                                         │                     └ lexical (FTS)     ┘         ↓      │
                                         │                        LLM: GROUND · CITE · ABSTAIN      │
 ────────────────────────────────────── ┘ ────────────────────────────────────────────────────────┘
```

### Concurrency: `psycopg_pool.AsyncConnectionPool`

The served API holds one process-lifetime `psycopg_pool.AsyncConnectionPool` (`min_size=5, max_size=20, timeout=30.0`), opened in the FastAPI `lifespan`. `/stats` and `/retrieve` take a connection through a `Depends(get_conn)` dependency — a route asks for a connection, it never touches the pool object:

```python
async def get_conn(request: Request) -> AsyncIterator[AsyncConnection]:
    async with request.app.state.pool.connection() as conn:
        yield conn
```

Two properties this buys, both load-bearing under concurrent load rather than incidental:

- **Checkout is non-blocking.** An `AsyncConnectionPool` awaits a free connection instead of blocking a worker thread, so concurrent requests queue on the pool's own scheduler, not on OS threads. Each connection is validated on checkout (`check=AsyncConnectionPool.check_connection`) and transparently replaced if a proxy — a Docker port-forward, a managed-Postgres load balancer — silently dropped it while idle.
- **A connection is held only for retrieval, never for generation.** `/ask/stream` deliberately does *not* use `Depends(get_conn)` — it checks out a connection manually, inside its own response generator, and releases it (an explicit `async with`) before the connection-free generation phase runs. That's not a style choice; it's the fix for a real bug (below).

**`/ask/stream`'s streaming design, and the bug that shaped it.** The original version *did* use `Depends(get_conn)`: retrieve using the injected connection, then construct and return `StreamingResponse`. That's a defect — Starlette sends the ASGI response's headers as soon as the response object exists, before it iterates the body. With retrieval running first, the client received zero bytes, not even a status line, for the entire CPU-bound retrieval phase (which can run into minutes). Browsers apply a materially stricter time-to-first-byte timeout than they do to an already-open, slow stream, and that's what fired — found by an actual user hitting the deployed app, not by our own test suite, which never sat in that failure window long enough to trip it. The fix: retrieval now runs *inside* the response's own async generator, so `StreamingResponse` returns — and headers flush — immediately, while the connection checkout/release still brackets only the retrieval portion. Verified directly: a short-timeout `curl` shows `200 OK` with full headers in well under a second, then correctly times out waiting on the still-slow body.

A platform-specific correctness note, found by testing, not assumed: on Windows, `asyncio`'s default `ProactorEventLoop` cannot run psycopg's async driver at all (a hard third-party incompatibility, not a config issue) — the serving process requires a selector-based event loop.

**The connection pool isn't the concurrency bottleneck — a non-preemptible thread is.** An adversarial self-audit asked whether a disconnect mid-request leaves a pooled connection stuck. It doesn't: an `async with` block runs its cleanup on task cancellation exactly as it does on any other exception, verified against Starlette's actual disconnect path, not assumed. But every rerank/embed call runs on one dedicated worker thread (`agroteca.torch_runtime`, deliberately single-threaded — see [Deterministic Startup](#deterministic-startup-explicit-model-warmup)), and cancelling the *task* awaiting that thread doesn't stop the *thread* — Python threads aren't preemptible. A disconnect mid-rerank leaves the call running to completion regardless (~500 s on CPU), and without a cap, every other request's call queues silently behind it. `run_torch` now caps in-flight admissions at 2 and rejects immediately past that, rather than queuing unboundedly behind possibly-orphaned work — verified with a live concurrency test, not by inspection.

A follow-up audit tested that same pool against a live `EXPLAIN ANALYZE` on the real corpus (`Index Scan using chunks_embedding_hnsw` — the ANN index is hit, not a sequential scan) and against `psycopg_pool`'s own installed source, confirming the `async with conn:` wrapper around every checkout commits-or-rolls-back before a connection is ever returned to the pool — a failed or timed-out query cannot poison it. Two other claims from that audit — that GPU transfers needed `pin_memory`/`non_blocking` tuning, and that per-call `del`/`empty_cache()` was needed for VRAM safety — didn't hold: the former targets a thread the event loop never touches, and the latter conflates CPython's deterministic refcounting with its (irrelevant, here) cyclic garbage collector. What *did* ship from that pass: `run_torch` now reclaims CUDA memory only past a measured 1 GiB reserved-but-unallocated gap, on the same worker thread that owns the CUDA context — not on a schedule, and not on every call.

### Deterministic Startup: Explicit Model Warmup

`src/agroteca/warmup.py` loads the embedder and cross-encoder reranker onto the configured device and runs one real forward pass through each — `torch.inference_mode()`, then `torch.cuda.synchronize()` on GPU — **inside the same process that will serve requests**, before `uvicorn` accepts its first connection:

```python
def load_and_warm() -> None:
    require_cuda()                 # fail loud if AGROTECA_DEVICE=cuda but no GPU is visible
    embed_passages(WARMUP_BATCH)   # dense embedder: init CUDA context, allocate its VRAM
    warm_reranker()                # cross-encoder: same
    if settings.resolved_device == "cuda":
        torch.cuda.synchronize()   # block until warmup kernels actually finish, not just enqueue
```

`require_cuda()` is a deliberate fail-fast gate: with `AGROTECA_DEVICE=cuda` (the default), a container that lost GPU access — a bad `--gpus` flag, a missing `nvidia-container-toolkit` install — refuses to start rather than silently degrading to CPU. (`AGROTECA_DEVICE=auto` is available where that guarantee isn't wanted — see [Deployment](#deployment-fully-containerized).) The same warmup call is also the Dockerfile's `ENTRYPOINT` preflight, run before the image's `CMD`.

**An honest scope note on "deterministic":** determinism here means *the same process that serves requests pays model-load and kernel-init cost exactly once, at startup* — not a fixed millisecond bound in the abstract. That distinction matters concretely: on GPU this yields sub-second, consistent per-query reranking. On CPU it does not — eager-mode PyTorch inference over full-length (~512-token) chunks measures **~500 s** for a single 20-candidate rerank call on this system's reference hardware, against the ~25-30 s the project's previous ONNX-based backend achieved on the same CPU (that comparison, and why the gap exists — no batching bug; a quantization/graph-optimization gap — is in [`DEPLOY.md`](DEPLOY.md)). Warmup makes the *first* request as fast as every subsequent one; it does not make CPU inference fast. Treat `AGROTECA_DEVICE=cuda` as a production requirement, not a speed option.

### Retrieval Cascade

Each stage exists to fix the previous stage's blind spot: dense (pgvector cosine kNN) understands paraphrase but misses exact codes (`WL-323`); lexical (Postgres full-text search, `ts_rank_cd`) nails exact codes but misses paraphrase; Reciprocal Rank Fusion combines both rankings by position (sidestepping their incomparable score scales); a cross-encoder reranker reads each `(query, chunk)` pair *together* over the fused top-20 to sharpen precision before generation. Because the corpus mixes Spanish and English, the question is also retrieved in translation (ES↔EN): the two candidate pools are merged, and each chunk keeps the higher of its two reranker scores, so an English answer earns its score from the English question and a Spanish one from the Spanish. Retrieval-only latency and accuracy are measurable independently of generation via a dedicated `/retrieve` endpoint (no LLM call) — see [API Surface](#api-surface).

---

## Benchmark Results (measured)

Retrieval@5 = *answer-chunk@5*: did the top-5 results contain a chunk that is both (a) from the correct source document and (b) contains the verified answer phrase, verbatim, in machine-extracted text — not membership in the right document alone (that weaker signal is `document@5`, shown for reference). Scored over the 22-question golden set (20 answerable) via [`eval/compare_retrievers.py`](eval/compare_retrievers.py); the underlying scoring primitives are in [`eval/_scoring.py`](eval/_scoring.py), reused by every eval script in this repo including the concurrent, HTTP-level [`eval/eval_pipeline.py`](eval/eval_pipeline.py).

| Stage | retrieval@5 | answer@1 | document@5 | MRR@5 |
|---|---|---|---|---|
| Dense only (pgvector cosine kNN) | 0.35 | 0.30 | 0.65 | 0.33 |
| + Lexical, fused with RRF (hybrid) | 0.45 | 0.15 | 0.65 | 0.25 |
| + Cross-encoder reranking | 0.75 | 0.75 | 0.90 | 0.75 |
| **+ Bilingual query translation (the served config)** | **0.80** | **0.80** | **0.90** | **0.80** |
| — reranked cascade, deployable corpus only¹ | 0.65 | — | 0.75 | — |
| — **live public demo**, measured over HTTP² | **0.70** | — | — | — |

¹ Excludes the copyrighted local-only tier (~30% of the index), which a public deployment cannot legally serve. Same cascade, zero regressions; the gap is corpus size, not a retrieval regression. Full breakdown: [`eval/results.csv`](eval/results.csv), `deploy` row.

² The deployable corpus plus query translation, on the public demo's own runtimes (ONNX embedder, Jina rerank API, Groq translator), scored by [`eval/eval_pipeline.py`](eval/eval_pipeline.py) against [agroteca.vercel.app](https://agroteca.vercel.app) itself: 14/20, 0 errors, p50 2.1 s per retrieval ([`results/hosted_eval.md`](results/hosted_eval.md)). Three golden answers exist only in copyrighted books the demo can't serve, so its ceiling is 17/20 = 0.85; it gets 14 of the 17 it legally can. Against the deployable baseline (0.65), translation adds the same cross-lingual recovery measured locally. Full breakdown: [`eval/results.csv`](eval/results.csv), `hosted` row.

Reranking fixed six questions the earlier stages missed with **zero regressions**. `retrieval@5` alone hides the sharper story: `answer@1` and `MRR@5` jump `0.15 → 0.75` — whenever the answer is present in the fused pool, the reranker pins it to rank 1. (Hybrid's `answer@1` *dips below* dense's 0.30: equal-weight RRF trades top-1 precision for pool recall, which is exactly what the reranker exists to repair.) Query translation then recovered a cross-lingual miss (q08) that had survived three embedder configurations, again with zero regressions ([`eval/query_translation.py`](eval/query_translation.py), `serving` row) — the same 0.80 as the embedder upgrade below, with no re-index.

A negative result kept in the record rather than discarded: a controlled `e5-large` (1024-dim) full re-index *doubled* dense-only recall (0.35 → 0.75) but moved the reranked cascade's number by only one question (0.75 → 0.80, within n=20 noise) — the cross-encoder had been masking a weak embedder the whole time. The upgrade was not shipped (8× query latency, a schema migration, for a noise-level final gain); the finding was kept. Full writeup: [`eval/results.csv`](eval/results.csv), `experiment` rows.

---

## Deployment: Fully Containerized

Two profiles run the same retrieval models. The containerized GPU stack below is the measured reference. A free public demo (`AGROTECA_PROFILE=hosted`: Neon, the same embedder through ONNX Runtime, the same reranker through Jina's API, generation through Groq) is described in [`DEPLOY.md`](DEPLOY.md#public-demo-at-0-agroteca_profilehosted), including the parity check it must pass first.

```bash
docker compose up -d
```

One command builds and starts the full stack — the served API (`api`, GPU-scheduled) and Postgres + pgvector (`db`, health-checked, with a named volume) — from the committed [`docker-compose.yml`](docker-compose.yml) and [`Dockerfile`](Dockerfile). Requires a CUDA-capable host with `nvidia-container-toolkit`; the image's `ENTRYPOINT` runs a GPU preflight (`warmup.py`) before the API process starts, and refuses to start on a misconfigured GPU-less host rather than silently degrading. `AGROTECA_DEVICE=cpu docker compose up -d` or `AGROTECA_DEVICE=auto docker compose up -d` override this for GPU-less smoke testing (see the [warmup determinism note](#deterministic-startup-explicit-model-warmup) above for what that trades away).

A fresh database needs its schema and corpus before it can answer anything — two more one-line commands, run once:

```bash
# schema
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/001_init.sql
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/002_tsv_simple.sql

# ingest the deployable corpus (LOCAL_MODE is off in the image; the copyrighted tier never ships)
docker compose run --rm api python -m agroteca.ingest.run
```

Generation runs against an Ollama-compatible endpoint the API points at (`AGROTECA_GEN_BASE_URL`) — deliberately not itself a compose service; point it at a GPU host you already run, or swap it for a different provider (a small, contained change — see [`DEPLOY.md`](DEPLOY.md)). Model weights for the embedder and reranker are baked into the image at build time (a dedicated build stage runs the same warmup that gates startup, before the final layer is assembled), so a running container has no network dependency for its own inference — only for reaching Postgres and the LLM endpoint, both externally configured.

Full recipe, env var reference, and the CPU-vs-GPU cost tradeoff: **[`DEPLOY.md`](DEPLOY.md)**.

---

## API Surface

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness; used by the Dockerfile `HEALTHCHECK` |
| `/stats` | GET | Corpus size (chunks / documents / tiers), one round-trip, no model |
| `/retrieve` | POST | Retrieval only — hybrid search + rerank, no LLM call. For eval, latency profiling, and integration testing without paying for generation |
| `/ask/stream` | POST | Full pipeline: an NDJSON stream — `progress` frames (`translating` / `reranking` / `generating`, backing the UI's stepping-stones indicator) interleaved with a `meta` frame (sources, scores, tiers, retrieval latency), then token frames and a `done` frame |
| `/feedback` | POST | Logs a 👍/👎 vote server-side |

`/retrieve` and `/ask/stream` share one retrieval implementation (`agroteca.generate.retrieve_ranked_async`) — the split exists purely so a retrieval-only caller (an eval harness, a monitoring probe) never pays generation cost or latency.

---

## Evaluation & Observability

- **[`eval/golden_set.jsonl`](eval/golden_set.jsonl)** — 22 hand-verified questions, written before the retriever existed, stratified across factual lookup, exact-term matching, synthesis, cross-lingual (ES↔EN), and deliberately unanswerable cases (abstention testing). Every answerable question's `must_contain` phrase is verified against machine-extracted text, not the source PDF read by eye.
- **[`eval/validate_golden.py`](eval/validate_golden.py)** — structural + corpus-membership validation of the golden set against the ingested manifests; run before trusting any eval result.
- **[`eval/compare_retrievers.py`](eval/compare_retrievers.py)** — in-process dense/lexical/hybrid/rerank comparison, `--latency` for per-stage p50/p95, `--open-only` for the deployable-corpus number.
- **[`eval/eval_pipeline.py`](eval/eval_pipeline.py)** — an HTTP-level, `asyncio`-concurrent harness that fires the golden set at a *live* `/retrieve` endpoint (not an in-process function call), so accuracy and latency are measured the way a real client experiences them — network, ASGI, and connection-pool overhead included. Pydantic-validates the golden set's structure before firing a single request; outputs a Markdown results table.

```bash
uv run python eval/validate_golden.py
uv run python eval/eval_pipeline.py --base-url http://127.0.0.1:8000
```

---

## Corpus Governance

Every document is assigned to exactly one of four tiers, and the tier — enforced in SQL, not just documentation — controls where it may appear:

| Tier | Loads when | Role |
|---|---|---|
| Open / deployable | always | Open-license, government, or public-domain — safe for any public deployment |
| Local-only | `AGROTECA_LOCAL_MODE=1` only | Copyrighted sources — never ingested or served publicly |
| Synthetic reference | always | Original, human-verified notes grounded in open sources; cited as synthetic |
| Distractor | eval only | Deliberate off-topic hard negatives; never a valid answer source |

The Dockerfile enforces this at the image layer, not just at runtime: `data/raw` (the copyrighted tier) is never `COPY`'d into any image, so a public deployment cannot serve it even under a configuration error. Full policy: [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

---

## Configuration Reference

All configuration is environment-driven (`AGROTECA_` prefix), read once at process start — no source edits for a deploy. Selected knobs most relevant to infrastructure planning:

| Env var | Default | Effect |
|---|---|---|
| `AGROTECA_DB_URL` | local dev Postgres | Connection string for the async pool |
| `AGROTECA_DEVICE` | `cuda` | `cuda` (fail-fast if unavailable) / `cpu` (explicit, slow) / `auto` (resolve at startup, opts out of fail-fast) |
| `AGROTECA_LOCAL_MODE` | `0` | Keep `0` for any public-facing deployment — gates the copyrighted tier |
| `AGROTECA_GEN_BASE_URL` / `AGROTECA_GEN_MODEL` | local Ollama | Swap generation to any Ollama-compatible host, or a different provider entirely |
| `AGROTECA_RERANK_CANDIDATES` | `20` | Reranker pool size — the main lever on rerank latency |
| `AGROTECA_TRANSLATE_QUERIES` | `true` | Bilingual query translation for cross-lingual retrieval; fails soft to original-only |

Full table: [`DEPLOY.md`](DEPLOY.md).

---

## Repository Layout

```
src/agroteca/
  ingest/           # extract · normalize · chunk · embed · store
  retrieve/         # dense · lexical · fusion (RRF) · hybrid · rerank (sync + async)
  generate.py        # ground / cite / abstain generation; retrieval-only phase shared with /retrieve
  api.py             # FastAPI service and its async connection-pool DI
  warmup.py          # explicit model warmup / the AGROTECA_DEVICE fail-fast gate
  torch_runtime.py    # single dedicated worker thread for torch inference calls
migrations/          # pgvector + full-text schema
eval/                # golden set, validators, in-process and live-HTTP eval harnesses
scripts/             # one-off ops scripts (e.g. the resumable e5-large re-index driver)
data/                # governed-tier manifests, synthetic notes, raw sources (gitignored)
Dockerfile, docker-compose.yml, docker-entrypoint.sh   # the containerized deployment
DEPLOY.md            # full deploy recipe, env var reference, CPU/GPU cost tradeoff
DATA_GOVERNANCE.md   # the four-tier provenance/licensing policy
results/, eval/results.csv   # measured results backing every claim above
```

---

## Licensing & Provenance

Code is MIT-licensed (see [`LICENSE`](LICENSE)). Source documents retain their original licenses. Only open-license, government, or public-domain material is served by any public deployment; copyrighted material is confined to the local-only tier and is never redistributed. See [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).
