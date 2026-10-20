# Deploying Agroteca

The app is already production-hardened — config-driven models, a health-checked connection
pool, input guardrails, a startup warm-up. **Deploying is now configuration + a host, not code
changes.** The one real decision is *where generation runs* (see [Generation](#generation-the-one-real-decision)).

## What a public deploy serves

Only the **open + synthetic** tiers — **7,246 chunks across 60 documents**. The copyrighted
`local` tier (3,084 chunks) is never ingested when `LOCAL_MODE` is off, so a public server
*physically* cannot contain it — and the [`Dockerfile`](Dockerfile) reinforces that at the
image layer: it COPYs `data/synthetic` but never `data/raw`, so the tier can't ride along even
by accident. On that corpus the final cascade scores **0.65 answer@5 / 0.75 doc@5** (vs
0.75 / 0.90 on the full local index) — the honest number a stranger gets. See the `deploy` row
in [`eval/results.csv`](eval/results.csv).

## Configuration (all via `AGROTECA_` env vars)

| Env var | Default | For deploy |
|---|---|---|
| `AGROTECA_DB_URL` | `postgresql://postgres:agroteca@localhost:5433/agroteca` | your managed Postgres + pgvector URL |
| `AGROTECA_LOCAL_MODE` | `false` | **keep `false`** — never ingest/serve the copyrighted tier |
| `AGROTECA_GEN_BASE_URL` | `http://localhost:11434` | your Ollama-compatible LLM host (see below) |
| `AGROTECA_GEN_MODEL` | `qwen2.5:3b` | a faster/larger model available on that host |
| `AGROTECA_GEN_TIMEOUT` | `300` | read timeout (s); lower it for a fast hosted model |
| `AGROTECA_GEN_NUM_PREDICT` | `1024` | max answer tokens |
| `AGROTECA_DB_POOL_MAX` | `8` | pool ceiling; size to expected concurrency |
| `AGROTECA_DEVICE` | `cuda` | embedder/reranker device; `warmup.py` refuses to start if this is `cuda` and no GPU is visible. Set `cpu` explicitly for GPU-less local iteration (much slower -- see below), or `auto` to resolve cuda-if-available-else-cpu at startup (same code/image both places, but opts OUT of the fail-loud guarantee -- a "production" box that silently lost its GPU would serve on CPU instead of refusing to start) |
| `AGROTECA_RERANK_CANDIDATES` | `20` | lower to trade a little recall for reranker speed |

Changing `AGROTECA_EMBED_MODEL` also needs a re-index **and** a matching `VECTOR(n)` column
(migration) — it is not a hot swap.

## Generation: the one real decision

The app speaks the **Ollama HTTP API** (via the `ollama` client). Two paths:

- **Zero code change — hosted Ollama.** Run Ollama on a GPU box (or a managed Ollama host),
  point `AGROTECA_GEN_BASE_URL` at it, and pick a model with `AGROTECA_GEN_MODEL`. Generation
  drops from minutes (CPU) to sub-second. **Recommended.**
- **A different provider** (OpenAI / Anthropic / Together / …). These don't speak the Ollama API,
  so it's a small, contained change: swap `_client` and the three `.chat(...)` calls in
  [`src/agroteca/generate.py`](src/agroteca/generate.py) — the whole LLM surface. The prompt,
  grounding rules, abstention phrase, and NDJSON streaming protocol stay identical.

The embedder and cross-encoder reranker run **in-process** on `torch` (`sentence-transformers` /
`transformers`), CUDA by default — `AGROTECA_DEVICE=cuda` — for fast retrieval. **CPU is no
longer viable for real use**: `AGROTECA_DEVICE=cpu` measures ~500 s to rerank one query's
20-candidate pool (vs. ~25-30 s for the old ONNX/fastembed backend this replaced -- see
[`results/latency.md`](results/latency.md), now stale). `compute_score()`'s own batching is
already correct (confirmed by reading its source); the gap is that this eager-mode PyTorch path
has no quantization or graph optimization, unlike the ONNX weights it replaced. A GPU is now a
hard requirement in practice, not just a speed lever. Both models' CUDA context and VRAM are warmed inside
the server process's own startup (`src/agroteca/warmup.py`, called from `api.py`'s lifespan)
before it accepts requests — see that module for why a separate pre-exec warmup step can't do
this warming itself.

## Ship it as a container (recommended)

The [`Dockerfile`](Dockerfile) is a two-stage build: a `uv sync --frozen` off the committed
`uv.lock` (reproducible dep tree, dependency layer cached independently of source), then a
slim non-root runtime with a `/health` healthcheck and a warm-up-aware start period. One image
serves the API *and*, with an override command, runs the ingest — same code, same corpus rules.

```bash
docker build -t agroteca .

# One-shot ingest into your managed Postgres (LOCAL_MODE is already 0 in the image).
docker run --rm --gpus all \
  -e AGROTECA_DB_URL="postgresql://user:pass@host:5432/agroteca" \
  agroteca python -m agroteca.ingest.run

# Serve. Needs a CUDA-capable host + nvidia-container-toolkit for --gpus (AGROTECA_DEVICE=cuda
# is the image default); pass -e AGROTECA_DEVICE=cpu instead to run without a GPU.
docker run -d --name agroteca -p 8000:8000 --gpus all \
  -e AGROTECA_DB_URL="postgresql://user:pass@host:5432/agroteca" \
  -e AGROTECA_GEN_BASE_URL="https://your-ollama-host" \
  -e AGROTECA_GEN_MODEL="<fast-model>" \
  -e AGROTECA_GEN_TIMEOUT=60 \
  agroteca
```

Do **not** mount a volume over `/app/.cache` — the models are baked into the image (below), and
an empty volume there would shadow them and force a re-download despite `AGROTECA_LOCAL_MODE`
having no network access story for that. First boot still isn't instant: `docker-entrypoint.sh`
runs a fail-fast GPU preflight, then the server process itself loads both models onto CUDA and
runs one real forward pass through each before it accepts requests (>90 s unwarmed → 27 s
first-request retrieval, on the old CPU numbers this hasn't been re-measured against yet).
`docker inspect --format '{{.State.Health.Status}}' agroteca` reports `healthy` only after
`Application startup complete`; put a reverse proxy (Caddy / nginx) in front for TLS and don't
expose 8000 directly.

**Model weights are baked in, not downloaded at runtime.** The `model-bake` build stage in the
[`Dockerfile`](Dockerfile) downloads the embedder + reranker weights and runs one warm pass
each (on CPU — `docker build` has no GPU access, and none is needed just to populate the cache),
then the runtime stage `COPY`s that cache in. This is why the running container has no network
dependency for its models and why mounting a cache volume over it (the old, pre-CUDA pattern)
is now actively wrong.

## Or run from source

```bash
# 1. Postgres + pgvector (managed, or the bundled compose on a VM)
docker compose up -d
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/001_init.sql
docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/002_tsv_simple.sql

# 2. Ingest ONLY the deployable corpus (LOCAL_MODE off -> no copyrighted docs in the DB).
#    The reference stack (torch, sentence-transformers, Ollama, PyMuPDF) is the `local` extra.
uv sync --extra local
AGROTECA_LOCAL_MODE=0 uv run python -m agroteca.ingest.run

# 3. Point generation at a fast host, then run the server (production: no --reload)
export AGROTECA_DB_URL="postgresql://user:pass@host:5432/agroteca"
export AGROTECA_GEN_BASE_URL="https://your-ollama-host"
export AGROTECA_GEN_MODEL="<fast-model>"
export AGROTECA_GEN_TIMEOUT=60                    # a fast host doesn't need 300s
uv run uvicorn agroteca.api:app --host 0.0.0.0 --port 8000
```

Put a reverse proxy (Caddy / nginx) in front for TLS. `--reload` is dev-only.

### Running natively on Windows

The command above fails on native Windows with `Psycopg cannot use the 'ProactorEventLoop' to
run in async mode`, and the pool then times out. That is not a bug in this app: uvicorn
*hardcodes* `ProactorEventLoop` on win32 (`uvicorn/loops/asyncio.py`), which it needs for
subprocess support, and psycopg's async mode requires a selector loop. Setting the asyncio
event-loop policy yourself does nothing, because uvicorn passes its own `loop_factory` to
`asyncio.run()` and that bypasses the policy entirely.

The fix is one flag, and no code change. Uvicorn treats an unrecognised `--loop` value as an
import string for a custom loop factory (`uvicorn/config.py`, `Config.get_loop_factory`), so
you can hand it the selector loop directly:

```bash
uv run uvicorn agroteca.api:app --host 127.0.0.1 --port 8000 --loop asyncio:SelectorEventLoop
```

Verified on Windows 11: startup completes, the pool opens, and `/health`, `/stats` and
`/ask/stream` all serve. Linux and the container are unaffected and need no flag.

**This is a development convenience, not a way to run the app.** The flag trades one failure for
a subtler one. `ProactorEventLoop` is Windows' scalable IOCP backend; `SelectorEventLoop` is the
legacy `select()` one, which scans its descriptor set linearly and is capped. Sockets that
accumulate faster than Windows retires them from `TIME_WAIT` will eventually exhaust it.

Observed, not theorised: a single `/ask/stream` request running ~20 minutes on CPU, with the web
page open and polling `/health` every 20s, produced ~279 short-lived connections and killed the
server outright:

```
OSError: [WinError 10055] ... the system lacked sufficient buffer space or a queue was full
  File "...\selectors.py", line 314, in _select
    r, w, x = select.select(r, w, w, timeout)
```

The request had already returned `200` and streamed its first frames; the process died mid-stream,
so the client saw a truncated body (`curl` exit 18). Nothing in this app leaks the sockets — the
backend simply doesn't scale to them.

Use the flag for a quick look at the UI or a short request. For anything long-running, concurrent,
or user-facing, run the container — Linux uses `epoll` and has none of this.

## Public demo at $0 (`AGROTECA_PROFILE=hosted`)

A free, card-free deployment of the **same models** the eval measures, so a public number is
measured rather than assumed:

| Piece | Local profile (reference) | Hosted profile (public demo) |
|---|---|---|
| Database | Postgres + pgvector (Docker) | Neon Free (pgvector, scales to zero; open + synthetic tiers only) |
| Query embedding | MiniLM via sentence-transformers / torch | the same MiniLM via ONNX Runtime, in the app (no torch) |
| Rerank | jina-reranker-v2 via torch | the same jina-reranker-v2 via Jina's rerank API |
| Generation + query translation | Ollama (`qwen2.5:3b`) | an OpenAI-compatible API (Groq) |
| App | uvicorn / Docker | Vercel Hobby (`app.py`, `vercel.json`) |

Ingestion always runs on the machine that holds the PDFs, with the local stack, so the index is
built by the reference implementation in both profiles.

**Hosted settings** (set in the Vercel project's environment variables, never committed):

| Env var | Value |
|---|---|
| `AGROTECA_PROFILE` | `hosted` |
| `AGROTECA_DB_URL` | Neon **pooled** connection string |
| `AGROTECA_DB_POOL_MIN` | `1` |
| `AGROTECA_JINA_API_KEY` | Jina key |
| `AGROTECA_GEN_API_KEY` | Groq key |
| `AGROTECA_OPENAI_MODEL` | a Groq model id that exists in your console today (default `llama-3.3-70b-versatile`) |
| `AGROTECA_GEN_TIMEOUT` / `AGROTECA_TRANSLATE_TIMEOUT` | `60` / `15` |
| `AGROTECA_ONNX_MODEL_FILE` | a variant that **passed** `eval/parity_check.py` |
| `AGROTECA_ONNX_MODEL_REVISION` | the embedder repo commit the parity check ran against |
| `AGROTECA_DAILY_REQUEST_CAP` / `AGROTECA_PER_IP_HOURLY_CAP` | `20` / `5` |
| `AGROTECA_IP_HASH_SALT` | a long random string |

**Order of operations** (each step has a check; don't skip ahead):

1. Create the Neon project. With its **direct** (non-pooled) URL in `AGROTECA_DB_URL`, run
   `uv run python scripts/apply_migrations.py`, then
   `AGROTECA_LOCAL_MODE=0 uv run python -m agroteca.ingest.run`.
   Check: `SELECT tier, count(*) FROM chunks GROUP BY tier` shows only `open` and `synthetic`.
2. `uv run python eval/parity_check.py --db` (same URL, Jina key in `.env`). Check
   `results/parity.md`: pick an ONNX variant with min cosine >= 0.999 whose size leaves room in
   the function's temporary disk (aim under ~300 MB); the rerank section must show strong rank
   agreement and the score range the UI's low-confidence cutoff assumes.
3. Import the repo in Vercel, set the variables above, deploy a preview. Check `/health`, then
   one Spanish and one English question on a phone-sized screen.
4. Measure what is deployed: with the per-IP cap temporarily `0`,
   `uv run python eval/eval_pipeline.py --base-url <preview-url>`; record it as a `hosted` row in
   `eval/results.csv` and restore the cap.
5. Only then point the domain at it.

**Free-tier ceilings, and what happens at each.** No card is on file anywhere, so every ceiling
stops the service rather than billing it:

| Service | Free allowance | At the ceiling |
|---|---|---|
| Neon | 0.5 GB storage, 100 CU-hours/month, scale-to-zero after 5 min idle | compute suspends until next month |
| Vercel Hobby | function limits per the Hobby plan; personal, non-commercial use | requests fail until reset |
| Groq | per-model rate limits (requests/tokens per minute and day) | 429 until the window resets |
| Jina | **10M tokens per key, once**, non-commercial | rerank calls fail; the demo stops answering |

The Jina allowance is the binding one: a bilingual question reranks up to ~40 chunks twice,
roughly 40-50k tokens, so 10M tokens is on the order of 200 questions. The daily and per-IP caps
exist to make that last; `AGROTECA_RERANK_CANDIDATES` trades recall for tokens if needed.

**Cold starts.** A new serverless instance downloads the ONNX model on its first request and Neon
wakes from zero, so the first question after an idle period is slow; later ones are not.

## Operational notes

- **Readiness.** Startup warms both models and opens the pool; the process serves only after
  `Application startup complete`. Poll `GET /health` for readiness. Expect a slower first boot —
  the reranker warm-up is deliberate; it's what makes the *first request* fast (>90 s → 27 s).
- **The pool self-heals.** Connections are validated on checkout, so an idle-dropped DB
  connection is replaced transparently instead of 500-ing a request.
- **Guardrails are on.** Questions are length-bounded, feedback votes are validated, a hostile
  1 MB payload is rejected at the edge.
- **Concurrency.** Each streamed request holds a pooled connection only for retrieval, then
  releases it before the (connection-free) generation. The API's async pool is sized in code
  (`min_size=5, max_size=20, timeout=30.0` in `api.py`'s lifespan) rather than via
  `AGROTECA_DB_POOL_MAX`, which now only governs the sync pool the CLI/ingest/eval scripts use.

## Cost reality (say it plainly)

The shipped default (`AGROTECA_DEVICE=cuda`) needs a GPU box — `warmup.py` refuses to start
otherwise. `AGROTECA_DEVICE=cpu`/`auto` run without one, but the rerank step alone now measures
~500 s/query (see [Generation](#generation-the-one-real-decision) above) — CPU is a correctness
fallback for local dev, not a usable free tier the way the pre-CUDA-refactor build was. A GPU is
the cost of this system actually working, not just the cost of making it fast.
