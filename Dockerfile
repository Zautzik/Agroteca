# syntax=docker/dockerfile:1
#
# Agroteca -- one image that both serves the API and (with an override command) ingests the
# corpus. The build is reproducible off uv.lock; the copyrighted `local` tier is physically
# absent because nothing below ever COPYs data/raw. See DEPLOY.md.

# ---- build stage: resolve the locked dependency tree into a self-contained venv ----
FROM python:3.12-slim AS builder

# uv arrives as a static binary -- no pip bootstrap, no global site-packages to reason about.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# Compile bytecode at install time (cheaper first import); copy across the cache-mount
# boundary since hardlinks don't survive it; never fetch a managed interpreter -- use the base.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install DEPENDENCIES ONLY first, so this heavy layer is keyed on the lockfile alone and
# survives every source edit -- only a dependency change pays the reinstall.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then the project itself -- a thin, fast-changing layer on top of the cached dependency tree.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- runtime stage: slim, no build tooling, non-root ----
FROM python:3.12-slim

# The embedder + cross-encoder download on first warm-up into this cache. Pin HF/fastembed to
# a stable path and mount a volume there so a restart doesn't re-pull ~1.5 GB. To bake the
# models into the image instead (immutable / offline deploys), add a build step that runs one
# inference before the final stage -- see DEPLOY.md ("Baking the models in").
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    AGROTECA_LOCAL_MODE=0

WORKDIR /app

# A non-root runtime user that owns the writable model cache.
RUN useradd --create-home --uid 10001 agroteca \
    && mkdir -p /app/.cache \
    && chown -R agroteca:agroteca /app

COPY --from=builder --chown=agroteca:agroteca /app/.venv /app/.venv
# Source is copied because the project is installed editable (its path finder points at
# /app/src, matching the builder's WORKDIR). Static assets ride inside src/agroteca/static.
COPY --chown=agroteca:agroteca src ./src
# Ship the migrations and the synthetic tier so the SAME image can run ingest / DB setup.
# data/raw (the copyrighted `local` tier) is deliberately never copied -- it cannot ship here.
COPY --chown=agroteca:agroteca migrations ./migrations
COPY --chown=agroteca:agroteca data/synthetic ./data/synthetic

USER agroteca
VOLUME ["/app/.cache"]
EXPOSE 8000

# Readiness is real: the process warms both models and opens the pool BEFORE it serves, so the
# start period is generous (a cold first boot also downloads the models). /health flips to 200
# only once "Application startup complete".
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# Serve. Override with `python -m agroteca.ingest.run` for a one-shot ingest container.
CMD ["uvicorn", "agroteca.api:app", "--host", "0.0.0.0", "--port", "8000"]
