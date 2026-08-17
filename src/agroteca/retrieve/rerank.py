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
    return TextCrossEncoder(model_name=settings.rerank_model, threads=settings.ort_threads)


def warm() -> None:
    """Eagerly load AND run one inference. Call once at server startup so the first real
    request pays neither the model load nor ONNX Runtime's first-run graph optimization
    (the big one), and so the lru_cache cold-start can't race across FastAPI's threadpool."""
    list(_reranker().rerank("warm", ["warming the cross-encoder graph"]))


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


def rerank_scored_bilingual(conn, query: str, translated: str, k: int = 5,
                            candidates: int | None = None, tiers: list[str] | None = None):
    """Bilingual variant of rerank_scored: union the hybrid pools of the original and the
    translated query, then re-score by the MAX cross-encoder score over both — so an English
    answer chunk earns its score from the English query member, a Spanish chunk from the Spanish
    one. `translated` is passed IN (this stays LLM-free; the caller owns translation), so the eval
    path never depends on the LLM. Measured to recover a cross-lingual miss no embedder swap could
    (eval/query_translation.py). Returns [(chunk_id, source_file, text, score), ...].
    """
    candidates = candidates or settings.rerank_candidates
    seen, pool = set(), []
    for row in (hybrid_search(conn, query, k=candidates, tiers=tiers)
                + hybrid_search(conn, translated, k=candidates, tiers=tiers)):
        if row[0] not in seen:                    # dedupe by chunk_id, keep first occurrence
            seen.add(row[0]); pool.append(row)
    if not pool:
        return []
    texts = [row[2] for row in pool]
    best = [float("-inf")] * len(texts)
    for q in (query, translated):
        for i, s in enumerate(_reranker().rerank(q, texts)):
            if s > best[i]:
                best[i] = s
    ranked = sorted(zip(pool, best), key=lambda p: p[1], reverse=True)
    return [(cid, sf, text, float(score)) for (cid, sf, text), score in ranked[:k]]
