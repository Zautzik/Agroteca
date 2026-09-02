#!/bin/sh
# Runs BEFORE any container command (the served API, or an `agroteca.ingest.run` override --
# see DEPLOY.md): a fail-fast GPU preflight, since both paths load the embedder onto
# AGROTECA_DEVICE. See src/agroteca/warmup.py for why this can't pre-warm the process CMD execs
# next -- that process warms itself, in its own code, before doing real work.
set -e
python warmup.py
exec "$@"
