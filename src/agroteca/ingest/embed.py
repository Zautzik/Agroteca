"""Stage 4 — embeddings via fastembed (ONNX runtime, CPU-friendly, no PyTorch).

Model: intfloat/multilingual-e5-large (1024-dim, multilingual).
E5 REQUIRES task prefixes — 'passage: ' for documents, 'query: ' for questions —
so we bake them in here; nothing downstream should ever embed raw text.
"""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from agroteca.config import settings


# E5 models require "query:"/"passage:" prefixes; MiniLM/mpnet do not.
_IS_E5 = "e5" in settings.embed_model.lower()


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # Downloads the ONNX weights on first use, then caches on disk.
    return TextEmbedding(model_name=settings.embed_model, threads=settings.ort_threads)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    """Embed document chunks."""
    if _IS_E5:
        texts = [f"passage: {t}" for t in texts]
    return list(_model().embed(texts, batch_size=settings.batch_size))


def embed_query(text: str) -> np.ndarray:
    """Embed a single query."""
    if _IS_E5:
        text = f"query: {text}"
    return next(iter(_model().embed([text])))
