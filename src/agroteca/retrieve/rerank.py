"""Cross-encoder reranking — the precision stage of the cascade.

Bi-encoder retrieval (dense + lexical, fused) is fast and gives good *recall* but
imperfect *precision* — the right chunk is often in the pool but not on top. A
cross-encoder reads each (query, chunk) pair *together* and re-scores them, floating
the true answer to the front. It's slow, so we only run it on the hybrid pool
(~20 candidates), never the whole corpus.
"""
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from agroteca.config import settings
from agroteca.retrieve.hybrid import hybrid_search


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    # Lazy + cached: the ~1.1 GB model downloads on first use, not on import.
    return TextCrossEncoder(model_name=settings.rerank_model)


def rerank_search(conn, query: str, k: int = 5,
                  candidates: int | None = None, tiers: list[str] | None = None):
    """Retrieve a hybrid candidate pool, then re-rank it with the cross-encoder.

    Returns the top-k rows as [(chunk_id, source_file, text), ...].
    """
    candidates = candidates or settings.rerank_candidates
    pool = hybrid_search(conn, query, k=candidates, tiers=tiers)
    if not pool:
        return []
    texts = [row[2] for row in pool]                       # each candidate's text
    scores = list(_reranker().rerank(query, texts))        # cross-encoder score per pair
    ranked = sorted(zip(pool, scores), key=lambda p: p[1], reverse=True)
    return [row for row, _score in ranked[:k]]


def rerank_scored(conn, query: str, k: int = 5,
                  candidates: int | None = None, tiers: list[str] | None = None):
    """Like rerank_search, but keeps the cross-encoder score for transparency.

    Returns [(chunk_id, source_file, text, score), ...] — the score lets the UI
    show a relevance meter and flag low-confidence matches.
    """
    candidates = candidates or settings.rerank_candidates
    pool = hybrid_search(conn, query, k=candidates, tiers=tiers)
    if not pool:
        return []
    texts = [row[2] for row in pool]
    scores = list(_reranker().rerank(query, texts))
    ranked = sorted(zip(pool, scores), key=lambda p: p[1], reverse=True)
    return [(cid, sf, text, float(score)) for (cid, sf, text), score in ranked[:k]]
