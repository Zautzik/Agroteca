"""Cross-encoder reranking — the precision stage of the cascade.

Bi-encoder retrieval (dense + lexical, fused) is fast and gives good *recall* but
imperfect *precision* — the right chunk is often in the pool but not on top. A
cross-encoder reads each (query, chunk) pair *together* and re-scores them, floating
the true answer to the front. It's slow, so we only run it on the hybrid pool
(~20 candidates), never the whole corpus.
"""
from functools import lru_cache

import httpx

from agroteca.config import settings
from agroteca.retrieve.hybrid import hybrid_search, hybrid_search_async
from agroteca.torch_runtime import run_torch


class _CrossEncoder:
    """Torch/CUDA wrapper around the Jina reranker's own `compute_score`, so call sites below
    keep the same `.rerank(query, texts) -> scores` contract the old fastembed backend had."""

    def __init__(self, model_name: str, device: str, revision: str):
        # Imported here, not at module top: the hosted profile ships without torch/transformers.
        from transformers import AutoModelForSequenceClassification
        # trust_remote_code=True: jina-reranker-v2 ships a custom architecture + scoring method
        # (not a stock AutoModelForSequenceClassification) -- this runs code from the HF repo.
        # `revision` pins that code to a reviewed commit (settings.rerank_model_revision) so a
        # later push to the repo can't silently change what gets executed here.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, dtype="auto", trust_remote_code=True,
        ).to(device).eval()

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        import torch
        pairs = [[query, t] for t in texts]
        with torch.inference_mode():
            scores = self._model.compute_score(pairs, max_length=1024)
        return scores if isinstance(scores, list) else [scores]


class _JinaAPIReranker:
    """The same jina-reranker-v2-base-multilingual, served by Jina's rerank API (hosted profile).

    Keeps the `.rerank(query, texts) -> scores` contract: one score per input text, in INPUT
    order. The API returns results sorted by relevance with each result's original `index`, so
    the scores are scattered back into input order here -- the callers sort on their own."""

    def __init__(self, url: str, model: str, api_key: str, timeout: float,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise RuntimeError("AGROTECA_JINA_API_KEY is not set (required by profile=hosted)")
        self._model = model
        self._client = httpx.Client(timeout=timeout, transport=transport,
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._url = url

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = self._client.post(self._url, json={
            "model": self._model, "query": query, "documents": texts,
            "top_n": len(texts), "return_documents": False,
        })
        resp.raise_for_status()
        scores = [float("-inf")] * len(texts)
        for item in resp.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        if any(s == float("-inf") for s in scores):
            raise RuntimeError("Jina rerank response did not score every document")
        return scores


@lru_cache(maxsize=1)
def _reranker():
    # Lazy + cached: the ~1.1 GB model downloads on first use, not on import.
    if settings.hosted:
        return _JinaAPIReranker(settings.jina_rerank_url, settings.jina_rerank_model,
                                settings.jina_api_key, settings.jina_timeout)
    return _CrossEncoder(settings.rerank_model, settings.resolved_device, settings.rerank_model_revision)


def warm() -> None:
    """Eagerly load AND run one inference. Call once at server startup (agroteca.warmup) so the
    first real request pays neither the model load nor CUDA's first-kernel/cuDNN-autotune cost,
    and so the lru_cache cold-start can't race across concurrent requests."""
    _reranker().rerank("warm", ["warming the cross-encoder graph"])


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
    scores = _reranker().rerank(query, texts)               # cross-encoder score per pair
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
    scores = _reranker().rerank(query, texts)
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


async def rerank_scored_async(conn, query: str, k: int = 5,
                              candidates: int | None = None, tiers: list[str] | None = None):
    """Async twin of rerank_scored for the FastAPI serving path. The cross-encoder call is
    CPU-bound torch inference, so it runs on the single dedicated torch worker thread
    (agroteca.torch_runtime) rather than the event loop or asyncio.to_thread's ad-hoc pool."""
    candidates = candidates or settings.rerank_candidates
    pool = await hybrid_search_async(conn, query, k=candidates, tiers=tiers)
    if not pool:
        return []
    texts = [row[2] for row in pool]
    scores = await run_torch(_reranker().rerank, query, texts)
    ranked = sorted(zip(pool, scores), key=lambda p: p[1], reverse=True)
    return [(cid, sf, text, float(score)) for (cid, sf, text), score in ranked[:k]]


async def rerank_scored_bilingual_async(conn, query: str, translated: str, k: int = 5,
                                        candidates: int | None = None, tiers: list[str] | None = None):
    """Async twin of rerank_scored_bilingual for the FastAPI serving path."""
    candidates = candidates or settings.rerank_candidates
    seen, pool = set(), []
    for row in (await hybrid_search_async(conn, query, k=candidates, tiers=tiers)
                + await hybrid_search_async(conn, translated, k=candidates, tiers=tiers)):
        if row[0] not in seen:                    # dedupe by chunk_id, keep first occurrence
            seen.add(row[0]); pool.append(row)
    if not pool:
        return []
    texts = [row[2] for row in pool]

    def _score_both() -> list[float]:
        best = [float("-inf")] * len(texts)
        for q in (query, translated):
            for i, s in enumerate(_reranker().rerank(q, texts)):
                if s > best[i]:
                    best[i] = s
        return best

    best = await run_torch(_score_both)
    ranked = sorted(zip(pool, best), key=lambda p: p[1], reverse=True)
    return [(cid, sf, text, float(score)) for (cid, sf, text), score in ranked[:k]]
