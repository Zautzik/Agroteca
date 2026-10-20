"""Stage 4 — embeddings via sentence-transformers (PyTorch, GPU-resident on `settings.device`).

The model is config-driven (`settings.embed_model`); the SHIPPED default is
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim). E5 models — a measured but deliberately
NOT-shipped upgrade (see eval/results.csv, docs/masterclass.md 4.13) — REQUIRE task prefixes
('passage: ' for documents, 'query: ' for questions); `_is_e5` decides per configured model and
the prefix helpers apply them, so nothing downstream ever embeds raw text under an E5 model.
NOTE: `settings.embed_dim` and the `VECTOR(n)` column in migrations/001_init.sql must both match
the configured model's dimension — swapping the model is a schema change, not a config flip.
"""
from functools import lru_cache

import numpy as np

from agroteca.config import settings


def _is_e5(model_name: str) -> bool:
    """E5 models require 'query:'/'passage:' prefixes; MiniLM/mpnet do not."""
    return "e5" in model_name.lower()


def _as_passages(texts: list[str], is_e5: bool) -> list[str]:
    """Document-side prefixing. A missing prefix on an E5 model silently degrades retrieval."""
    return [f"passage: {t}" for t in texts] if is_e5 else list(texts)


def _as_query(text: str, is_e5: bool) -> str:
    """Query-side prefixing (the counterpart to `_as_passages`)."""
    return f"query: {text}" if is_e5 else text


_IS_E5 = _is_e5(settings.embed_model)


@lru_cache(maxsize=1)
def _model():
    # Imported here, not at module top: the hosted profile ships without torch (it embeds through
    # ONNX Runtime -- see agroteca.ingest.onnx_embed), so importing this module must not need it.
    from sentence_transformers import SentenceTransformer
    # Downloads the weights on first use, then caches on disk; resident on settings.device
    # for the process lifetime (moving a loaded model between devices is not free).
    return SentenceTransformer(settings.embed_model, device=settings.resolved_device)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    """Embed document chunks. Always the torch path: ingestion runs on the developer machine,
    so the index is built by the reference implementation in every profile."""
    import torch
    with torch.inference_mode():
        vecs = _model().encode(_as_passages(texts, _IS_E5), batch_size=settings.batch_size,
                               convert_to_numpy=True, show_progress_bar=False)
    return list(vecs)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query: torch locally, the same model through ONNX Runtime when hosted."""
    if settings.hosted:
        from agroteca.ingest.onnx_embed import embed_query_onnx
        return embed_query_onnx(_as_query(text, _IS_E5))
    import torch
    with torch.inference_mode():
        return _model().encode(_as_query(text, _IS_E5), convert_to_numpy=True, show_progress_bar=False)
