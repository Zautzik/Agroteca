# syntax=docker/dockerfile:1
#
# Agroteca -- GPU-serving image. The embedder + cross-encoder run on CUDA in-process (torch);
# see src/agroteca/warmup.py for how the CUDA-context/VRAM warmup actually happens (in the
# uvicorn process's own lifespan -- a separate pre-exec process cannot warm it) and why
# docker-entrypoint.sh's `python warmup.py` step is a fail-fast GPU preflight, not that warmup.
# One image both serves the API and (with an override command) ingests the corpus -- same code,
# same corpus rules. See DEPLOY.md.

# ---- build stage: resolve the locked dependency tree into a self-contained venv ----
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

# The nvidia/cuda base ships no Python; let uv manage its own 3.12 interpreter rather than pull
# in apt/deadsnakes for one Python version. (torch's own wheels already bundle the CUDA/cuDNN
# runtime libs they need -- this base image's userspace CUDA/cuDNN is headroom for anything else
# that might eventually link against it directly, not a hard torch requirement.)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv arrives as a static binary -- no pip bootstrap, no global site-packages to reason about.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_PYTHON=3.12

WORKDIR /app

# Install DEPENDENCIES ONLY first, so this heavy layer (torch + transformers: the `local` extra) is
# keyed on the lockfile alone and survives every source edit -- only a dependency change pays
# the reinstall.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra local

# Then the project itself -- a thin, fast-changing layer on top of the cached dependency tree.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY warmup.py ./warmup.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra local


# ---- model-bake stage: download the embedder + reranker weights and run one warm forward pass
# each, so they land on disk (and any load-time bug surfaces at BUILD time, not deploy time).
# Forced onto CPU for this one step -- `docker build` has no GPU access by default, and none is
# needed just to populate the HF cache directory the runtime stage copies forward.
FROM builder AS model-bake
ENV HF_HOME=/build-cache/huggingface \
    AGROTECA_DEVICE=cpu
RUN python warmup.py


# ---- runtime stage: non-root, model weights baked in, zero network dependency ----
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    AGROTECA_LOCAL_MODE=0 \
    AGROTECA_DEVICE=cuda \
    HF_HUB_OFFLINE=1

# HF_HUB_OFFLINE=1 matters as much as the COPY below: without it, `from_pretrained` still makes
# a network call on every cold start to check for a newer revision before falling back to the
# baked-in cache -- silently reintroducing the "runtime network dependency" the model-bake stage
# exists to remove, as a multi-second stall on the FIRST call rather than an outright failure.

WORKDIR /app

# A non-root runtime user. No writable cache volume is declared: the whole point of the
# model-bake stage is that this image is immutable and network-independent at runtime, so
# nothing should be mounted over /app/.cache -- that would shadow the baked-in weights and
# force a re-download, defeating it. See DEPLOY.md.
RUN useradd --create-home --uid 10001 agroteca \
    && mkdir -p /app/.cache \
    && chown -R agroteca:agroteca /app

COPY --from=builder --chown=agroteca:agroteca /app/.venv /app/.venv
# Source is copied because the project is installed editable (its path finder points at
# /app/src, matching the builder's WORKDIR). Static assets ride inside src/agroteca/static.
COPY --chown=agroteca:agroteca src ./src
COPY --chown=agroteca:agroteca warmup.py ./warmup.py
# Ship the migrations and the synthetic tier so the SAME image can run ingest / DB setup.
# data/raw (the copyrighted `local` tier) is deliberately never copied -- it cannot ship here.
COPY --chown=agroteca:agroteca migrations ./migrations
COPY --chown=agroteca:agroteca data/synthetic ./data/synthetic
# The baked model weights from model-bake -- COPYing them in is what makes the container
# immutable / offline-startable; nothing below ever reaches out to Hugging Face at runtime.
COPY --from=model-bake --chown=agroteca:agroteca /build-cache/huggingface /app/.cache/huggingface

COPY --chown=agroteca:agroteca docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

USER agroteca
EXPOSE 8000

# Readiness is real: the process warms both models on CUDA and opens the DB pool BEFORE it
# serves, so the start period is generous (a cold first boot still pays real CUDA-context /
# cuDNN-autotune cost, just never a network download). /health flips to 200 only once
# "Application startup complete".
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# docker-entrypoint.sh runs `python warmup.py` (a fail-fast GPU preflight -- see its docstring
# for why this step, run before CMD, does NOT pre-warm the process CMD execs) then execs CMD.
# Override CMD (e.g. `docker run agroteca python -m agroteca.ingest.run`) for a one-shot ingest
# container; the preflight still runs first either way.
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "agroteca.api:app", "--host", "0.0.0.0", "--port", "8000"]
