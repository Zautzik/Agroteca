import json
import time

from ollama import Client

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.rerank import rerank_search, rerank_scored

# One reusable client: pins the Ollama host and a read timeout so a hung or slow model
# surfaces an error instead of hanging the stream forever. Model + host are config knobs,
# so a deploy swaps to a hosted/smaller model via env vars rather than editing source.
_client = Client(host=settings.gen_base_url, timeout=settings.gen_timeout)
_GEN_OPTS = {"num_predict": settings.gen_num_predict}

SYSTEM_PROMPT = """You are an agronomy assistant. Answer the QUESTION using ONLY the CONTEXT chunks provided.

RULES:
1. GROUND — base every statement only on the CONTEXT. Never use outside knowledge.
2. CITE — after each claim, name its source in parentheses, e.g. (Libro_INIA_04.pdf, p. 37).
3. ABSTAIN — if the CONTEXT does not contain the answer, reply with EXACTLY this line and nothing else:
   No encuentro la respuesta en el contexto disponible.
   Never invent an answer. Never bend this rule to please the user.
4. Answer in the same language as the QUESTION."""


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
    rows = rerank_scored(conn, question, k=k)
    meta = _meta_for(conn, [r[0] for r in rows])
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    sources = [
        {"source": sf, "tier": meta.get(cid, ("?", None))[0], "page": meta.get(cid, ("?", None))[1],
         "score": round(sc, 3), "snippet": text.strip()[:240]}
        for (cid, sf, text, sc) in rows
    ]
    meta_line = json.dumps({"type": "meta", "sources": sources, "retrieval_ms": retrieval_ms}) + "\n"
    pages = {cid: meta.get(cid, ("?", None))[1] for (cid, sf, text, sc) in rows}
    context = format_context([(cid, sf, text) for (cid, sf, text, _s) in rows], pages)
    return meta_line, context


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