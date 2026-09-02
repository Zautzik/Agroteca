"""Dockerfile ENTRYPOINT preflight shim -- see src/agroteca/warmup.py for the real logic and
the important note on why this can't pre-warm the uvicorn process that starts after it."""
from agroteca.config import settings
from agroteca.warmup import load_and_warm

if __name__ == "__main__":
    load_and_warm()
    print(f"warmup: {settings.rerank_model} + {settings.embed_model} loaded and warmed on {settings.device}")
