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

# 2. Ingest ONLY the deployable corpus (LOCAL_MODE off -> no copyrighted docs in the DB)
AGROTECA_LOCAL_MODE=0 uv run python -m agroteca.ingest.run

# 3. Point generation at a fast host, then run the server (production: no --reload)
export AGROTECA_DB_URL="postgresql://user:pass@host:5432/agroteca"
export AGROTECA_GEN_BASE_URL="https://your-ollama-host"
export AGROTECA_GEN_MODEL="<fast-model>"
export AGROTECA_GEN_TIMEOUT=60                    # a fast host doesn't need 300s
uv run uvicorn agroteca.api:app --host 0.0.0.0 --port 8000
```

Put a reverse proxy (Caddy / nginx) in front for TLS. `--reload` is dev-only.

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
