# Deploying Agroteca

The app is already production-hardened — config-driven models, a health-checked connection
pool, input guardrails, a startup warm-up. **Deploying is now configuration + a host, not code
changes.** The one real decision is *where generation runs* (see [Generation](#generation-the-one-real-decision)).

## What a public deploy serves

Only the **open + synthetic** tiers. The copyrighted `local` tier is never ingested when
`LOCAL_MODE` is off, so a public server physically cannot contain it. On that corpus the final
cascade scores **0.65 answer@5 / 0.75 doc@5** (vs 0.75 / 0.90 on the full local index) — the
honest number a stranger gets. See the `deploy` row in [`eval/results.csv`](eval/results.csv).

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
| `AGROTECA_ORT_THREADS` | *(all cores)* | set to bound ONNX threads if rerank requests overlap |
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

The embedder and cross-encoder reranker run **in-process** (fastembed / ONNX). CPU by default
(reranker ~25 s/query — the free-tier tax); pass a CUDA provider / `cuda=True` on a GPU box, or
use a hosted reranker, to make retrieval sub-second. See [`results/latency.md`](results/latency.md).

## Steps

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
  releases it before the (connection-free) generation. Size `AGROTECA_DB_POOL_MAX` to your
  concurrency; cap `AGROTECA_ORT_THREADS` if multiple rerank requests can overlap.

## Cost reality (say it plainly)

On CPU this is a correct, slow system (~25 s retrieval + minutes of generation). The hardening
makes it **deployable**; a GPU — or a hosted model + reranker — makes it **fast**. That's a
dollar figure, not a code change, and the config above is exactly where you spend it.
