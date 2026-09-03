import asyncio
import json
import time

import httpx
from ollama import AsyncClient, Client

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.rerank import (
    rerank_scored,
    rerank_scored_async,
    rerank_scored_bilingual,
    rerank_scored_bilingual_async,
    rerank_search,
)
from agroteca.retrieve.translate import translate_query

# One reusable client. Short connect timeout (if Ollama is down, fail fast) but a generous
# read timeout, because CPU prompt-eval can take a while to reach the first token; a truly
# hung model still errors after gen_timeout instead of hanging forever. Model + host are
# config knobs, so a deploy swaps to a hosted/smaller model via env vars, not source edits.
_client = Client(host=settings.gen_base_url,
                 timeout=httpx.Timeout(settings.gen_timeout, connect=10.0))
# Async twin, for the FastAPI serving path -- see stream_ndjson_async. Keeping both (rather
# than routing the sync CLI/eval path through the async client) means /ask/stream never blocks
# the event loop on Ollama I/O, without touching the sync call sites below.
_async_client = AsyncClient(host=settings.gen_base_url,
                            timeout=httpx.Timeout(settings.gen_timeout, connect=10.0))
_GEN_OPTS = {"num_predict": settings.gen_num_predict}

SYSTEM_PROMPT = """You are an agronomy assistant. Answer the QUESTION using ONLY the CONTEXT chunks provided.

RULES:
1. GROUND — base every statement only on the CONTEXT. Never use outside knowledge.
2. CITE — after each claim, name its source in parentheses, e.g. (Libro_INIA_04.pdf, p. 37).
3. ABSTAIN — if the CONTEXT does not contain the answer, reply with EXACTLY this line and nothing else:
   No encuentro la respuesta en el contexto disponible.
   Never invent an answer. Never bend this rule to please the user.
4. Answer in the same language as the QUESTION."""

# The canonical abstention line rule 3 mandates. Abstention is a *measurable* metric only because
# this string is exact and machine-detectable -- keep the detector, the eval, and the UI agreeing
# on one phrase. `is_abstention` is deliberately lenient on case/whitespace but not on wording.
ABSTAIN_PHRASE = "No encuentro la respuesta en el contexto disponible."
_ABSTAIN_NEEDLE = "no encuentro la respuesta en el contexto disponible"


def is_abstention(text: str) -> bool:
    """True iff the reply is the canonical abstention line (case- and whitespace-insensitive).

    A near-miss ('No tengo la respuesta') must NOT count, or 'the model abstained' becomes a vibe
    instead of a metric."""
    return _ABSTAIN_NEEDLE in " ".join(text.split()).lower()


def _cite(source_file: str, page) -> str:
    """Source label for a chunk — with the page number when we have it (verifiable citations)."""
    return f"[Fuente: {source_file}, p. {page}]" if page else f"[Fuente: {source_file}]"


def format_context(rows, pages: dict | None = None) -> str:
    """rows: list of (chunk_id, source_file, text). `pages`: {chunk_id: page} so each
    chunk's source label carries the page it came from. Returns ONE string, chunks
    separated by a blank line.
    """
    pages = pages or {}
    chunks = [f"{_cite(sf, pages.get(cid))}\n{text.strip()}" for (cid, sf, text) in rows]
    return "\n\n".join(chunks)


def ask_ollama(system: str, user: str) -> str:
    """Send a system+user prompt to the local model, return its text reply."""
    resp = _client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options=_GEN_OPTS,
    )
    return resp["message"]["content"]


def answer(conn, question: str, k: int = 5) -> str:
    """Retrieve the top-k chunks for the question, then generate a grounded, cited answer."""
    rows = rerank_search(conn, question, k=k)              # 1. real chunks from the corpus
    pages = {cid: pg for cid, (_t, pg) in _meta_for(conn, [r[0] for r in rows]).items()}
    context = format_context(rows, pages)                  # 2. label each with source + page
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"  # 3. build the user message
    return ask_ollama(SYSTEM_PROMPT, user)                 # 4. ground + cite (or abstain)

def answer_stream(conn, question: str, k: int = 5):
    """Like answer(), but YIELDS the reply token-by-token as the model generates it."""
    rows = rerank_search(conn, question, k=k)              # retrieval (the pre-token wait)
    pages = {cid: pg for cid, (_t, pg) in _meta_for(conn, [r[0] for r in rows]).items()}
    context = format_context(rows, pages)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    stream = _client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        stream=True,                                       # <- Ollama now yields chunks, not one blob
        options=_GEN_OPTS,
    )
    for chunk in stream:
        yield chunk["message"]["content"]                  # <- hand over each token as it arrives


def _meta_for(conn, chunk_ids):
    """Look up governance tier + page for a set of chunk ids -> {chunk_id: (tier, page)}.
    `page` was already stored per chunk (chunk.py computes char offsets, run.py maps them
    to a page) — it just never reached the citation until now."""
    if not chunk_ids:
        return {}
    rows = conn.execute(
        "SELECT chunk_id, tier, page FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)
    ).fetchall()
    return {cid: (tier, page) for cid, tier, page in rows}


def prepare_ndjson(conn, question: str, k: int = 5) -> tuple[str, str]:
    """DB phase (needs a connection): retrieve + score + tier lookup. Returns
    (meta_line, context) so the caller can RELEASE the DB connection before the long,
    DB-free generation stream — a pooled connection must not idle through CPU generation.

      meta_line -> {"type":"meta","sources":[{source,tier,score,snippet}...],"retrieval_ms":N}
    """
    t0 = time.perf_counter()
    # Bilingual retrieval when enabled: translate the question and rerank the union pool by max
    # score, so a Spanish question can reach an English answer chunk. Fails soft -- translate_query
    # returns None on any error, and we fall back to the original-only cascade.
    translated = translate_query(question) if settings.translate_queries else None
    if translated:
        rows = rerank_scored_bilingual(conn, question, translated, k=k)
    else:
        rows = rerank_scored(conn, question, k=k)
    meta = _meta_for(conn, [r[0] for r in rows])
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    sources = [
        {"source": sf, "tier": meta.get(cid, ("?", None))[0], "page": meta.get(cid, ("?", None))[1],
         "score": round(sc, 3), "snippet": text.strip()[:240]}
        for (cid, sf, text, sc) in rows
    ]
    meta_line = json.dumps({"type": "meta", "sources": sources, "retrieval_ms": retrieval_ms,
                            "translated_query": translated}) + "\n"
    pages = {cid: meta.get(cid, ("?", None))[1] for (cid, sf, text, sc) in rows}
    context = format_context([(cid, sf, text) for (cid, sf, text, _s) in rows], pages)
    return meta_line, context


async def _meta_for_async(conn, chunk_ids):
    """Async twin of _meta_for for the FastAPI serving path."""
    if not chunk_ids:
        return {}
    cur = await conn.execute(
        "SELECT chunk_id, tier, page FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)
    )
    rows = await cur.fetchall()
    return {cid: (tier, page) for cid, tier, page in rows}


async def retrieve_ranked_async(conn, question: str, k: int = 5):
    """The retrieval-only phase, shared by prepare_ndjson_async (the /ask/stream + generation
    path) and the /retrieve route (eval/introspection -- no generation): translate + hybrid
    rerank + tier/page lookup. translate_query is a blocking Ollama call, so it runs in a
    worker thread rather than the event loop.

    Returns (rows, meta, retrieval_ms, translated):
      rows -> [(chunk_id, source_file, text, score), ...], best first
      meta -> {chunk_id: (tier, page)}
    """
    t0 = time.perf_counter()
    translated = await asyncio.to_thread(translate_query, question) if settings.translate_queries else None
    if translated:
        rows = await rerank_scored_bilingual_async(conn, question, translated, k=k)
    else:
        rows = await rerank_scored_async(conn, question, k=k)
    meta = await _meta_for_async(conn, [r[0] for r in rows])
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    return rows, meta, retrieval_ms, translated


def build_meta_and_context(rows, meta, retrieval_ms, translated) -> tuple[str, str]:
    """Turn raw retrieve_ranked_async(_with_progress) results into the NDJSON meta_line and
    the LLM's context block. Shared by prepare_ndjson_async and the progress-reporting
    /ask/stream path, so the two never drift on how a `meta` frame is built."""
    sources = [
        {"source": sf, "tier": meta.get(cid, ("?", None))[0], "page": meta.get(cid, ("?", None))[1],
         "score": round(sc, 3), "snippet": text.strip()[:240]}
        for (cid, sf, text, sc) in rows
    ]
    meta_line = json.dumps({"type": "meta", "sources": sources, "retrieval_ms": retrieval_ms,
                            "translated_query": translated}) + "\n"
    pages = {cid: meta.get(cid, ("?", None))[1] for (cid, sf, text, sc) in rows}
    context = format_context([(cid, sf, text) for (cid, sf, text, _s) in rows], pages)
    return meta_line, context


async def prepare_ndjson_async(conn, question: str, k: int = 5) -> tuple[str, str]:
    """Async twin of prepare_ndjson, for a route holding an injected AsyncConnection. Same
    contract: returns (meta_line, context) so the caller can let the connection go back to
    the pool (via FastAPI's Depends teardown) before the DB-free generation stream starts."""
    rows, meta, retrieval_ms, translated = await retrieve_ranked_async(conn, question, k)
    return build_meta_and_context(rows, meta, retrieval_ms, translated)


async def retrieve_ranked_with_progress(conn, question: str, k: int = 5):
    """Async-generator twin of retrieve_ranked_async for the /ask/stream progress UI: yields
    ("progress", stage) events at each real phase boundary, then a final ("result", payload)
    event with the same (rows, meta, retrieval_ms, translated) tuple retrieve_ranked_async
    returns.

    Stage granularity matches actual architecture boundaries -- translating, then reranking
    (hybrid search + the cross-encoder together: hybrid is sub-second, so splitting it out as
    its own stage would flash by unnoticed and only add a stage the UI has to handle for no
    real signal). Finer-grained progress *within* reranking (e.g. "candidate 7/20") isn't
    available without abandoning the reranker's single batched forward pass, which is what
    keeps it as fast as it currently is (see agroteca.retrieve.rerank) -- so this reports
    real phase transitions, not a fabricated percentage.
    """
    t0 = time.perf_counter()
    translated = None
    if settings.translate_queries:
        yield ("progress", "translating")
        translated = await asyncio.to_thread(translate_query, question)

    yield ("progress", "reranking")
    if translated:
        rows = await rerank_scored_bilingual_async(conn, question, translated, k=k)
    else:
        rows = await rerank_scored_async(conn, question, k=k)

    meta = await _meta_for_async(conn, [r[0] for r in rows])
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    yield ("result", (rows, meta, retrieval_ms, translated))


def stream_ndjson(question: str, context: str):
    """Generation phase (no DB connection held): yield token frames, then a done frame."""
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    g0 = time.perf_counter()
    stream = _client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        stream=True,
        options=_GEN_OPTS,
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield json.dumps({"type": "token", "text": piece}) + "\n"
    yield json.dumps({"type": "done", "generation_ms": int((time.perf_counter() - g0) * 1000)}) + "\n"


async def stream_ndjson_async(question: str, context: str):
    """Async twin of stream_ndjson, for the FastAPI serving path: yield token frames, then a
    done frame, via ollama.AsyncClient so generation never blocks the event loop."""
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    g0 = time.perf_counter()
    stream = await _async_client.chat(
        model=settings.gen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        stream=True,
        options=_GEN_OPTS,
    )
    async for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield json.dumps({"type": "token", "text": piece}) + "\n"
    yield json.dumps({"type": "done", "generation_ms": int((time.perf_counter() - g0) * 1000)}) + "\n"


def answer_ndjson(conn, question: str, k: int = 5):
    """Backward-compatible one-shot: prepare (DB) then stream (generation) in one call.
    The served API calls prepare_ndjson + stream_ndjson directly so it can return the
    pooled connection to the pool between the two phases."""
    meta_line, context = prepare_ndjson(conn, question, k=k)
    yield meta_line
    yield from stream_ndjson(question, context)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # clean ES accents in the console

    conn = store.connect()
    question = "Según la guía de cosecha de agua lluvia, ¿cuánta agua se puede captar por cada milímetro de lluvia caída sobre un metro cuadrado de techo?"
    print(answer(conn, question))
    conn.close()