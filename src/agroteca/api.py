import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from psycopg import AsyncConnection
from pydantic import BaseModel, Field

from agroteca.config import settings
from agroteca.generate import (
    build_meta_and_context,
    retrieve_ranked_async,
    retrieve_ranked_with_progress,
    stream_ndjson_async,
)
from agroteca.ingest import store
from agroteca.warmup import load_and_warm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load + warm both GPU-resident models in THIS process (the one that will actually serve
    # requests) before accepting connections -- see agroteca/warmup.py for why this must happen
    # here and not in a separate pre-exec script. Also guards against the lru_cache cold-start
    # racing across concurrent requests.
    load_and_warm()

    # The served API's pooled DB connections — reused across requests instead of a fresh TCP
    # handshake + auth per question. min_size=5/max_size=20/timeout=30.0 give headroom against
    # request starvation under concurrent load; construct+open here (not at import time) since
    # an async pool must open from a running event loop.
    pool = store.make_async_pool(min_size=5, max_size=20, timeout=30.0)
    await pool.open(wait=True, timeout=30.0)
    app.state.pool = pool
    yield
    await app.state.pool.close()


app = FastAPI(title="Agroteca", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
FEEDBACK_LOG = settings.root / "feedback.jsonl"


async def get_conn(request: Request) -> AsyncIterator[AsyncConnection]:
    """DI hook: check out a pooled async connection for the route. Routes take it via
    `Depends(get_conn)` instead of touching the pool directly; FastAPI closes this generator
    (returning the connection to the pool) as soon as the route function returns."""
    async with request.app.state.pool.connection() as conn:
        yield conn


class Question(BaseModel):
    # Bounded: an unbounded string would flow straight into the FTS regex + tsquery.
    question: str = Field(..., min_length=1, max_length=2000)


class Feedback(BaseModel):
    question: str = Field(..., max_length=2000)
    vote: Literal["up", "down"]
    answer_preview: str = Field("", max_length=5000)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_file: str
    page: int | None = None
    tier: str
    score: float
    text: str


class RetrieveResponse(BaseModel):
    sources: list[RetrievedChunk]
    retrieval_ms: int
    translated_query: str | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats(conn: AsyncConnection = Depends(get_conn)):
    """Cheap corpus stats for the UI ribbon (no model, one round-trip per count)."""
    chunks = (await (await conn.execute("SELECT count(*) FROM chunks")).fetchone())[0]
    docs = (await (await conn.execute("SELECT count(*) FROM documents")).fetchone())[0]
    tiers = (await (await conn.execute("SELECT count(DISTINCT tier) FROM chunks")).fetchone())[0]
    return {"chunks": chunks, "documents": docs, "tiers": tiers}


@app.post("/feedback")
def feedback(fb: Feedback):
    """Append a thumbs-up/down to a JSONL log — the seed of a feedback loop."""
    rec = {"ts": time.time(), "vote": fb.vote,
           "question": fb.question, "answer_preview": fb.answer_preview[:200]}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/manifest.webmanifest")
def manifest():
    icon = ("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' "
            "viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌱</text></svg>")
    return JSONResponse({
        "name": "Agroteca", "short_name": "Agroteca",
        "start_url": "/", "display": "standalone",
        "background_color": "#f4eede", "theme_color": "#3f5b2e",
        "icons": [{"src": icon, "sizes": "any", "type": "image/svg+xml"}],
    })


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: Question, conn: AsyncConnection = Depends(get_conn)):
    """Retrieval-only: hybrid search + rerank, no LLM generation. Exists for eval/introspection
    (e.g. eval/eval_pipeline.py) so measuring retrieval accuracy/latency doesn't pay for a full
    CPU/GPU generation call per query -- /ask/stream bundles both and would badly pollute a
    latency measurement meant to isolate retrieval."""
    rows, meta, retrieval_ms, translated = await retrieve_ranked_async(conn, payload.question)
    sources = [
        RetrievedChunk(chunk_id=cid, source_file=sf, tier=meta.get(cid, ("?", None))[0],
                       page=meta.get(cid, ("?", None))[1], score=round(sc, 4), text=text)
        for (cid, sf, text, sc) in rows
    ]
    return RetrieveResponse(sources=sources, retrieval_ms=retrieval_ms, translated_query=translated)


@app.post("/ask/stream")
async def ask_stream(payload: Question, request: Request):
    """Stream the answer as NDJSON: meta (sources + scores + tiers + timing), then token
    frames, then a done frame.

    Retrieval -- including the pooled-connection checkout -- happens INSIDE the generator,
    not before StreamingResponse is constructed. That matters: Starlette sends the ASGI
    response's status+headers as soon as StreamingResponse is returned, before it starts
    iterating the body. If retrieval instead ran in the route handler first (the earlier
    design), the client got zero bytes -- not even headers -- for the whole CPU-bound
    retrieval phase (which can run into minutes on CPU). Browsers apply a much stricter
    "time to first byte" timeout than they do to a slow-but-already-started stream, and this
    was the literal, measured cause of a client-side "network error" firing well before the
    backend ever responded. Moving retrieval into the generator means headers flush
    immediately and the browser sees an open, live stream from the first instant, even
    though the first real frame is still minutes away on CPU.

    The pooled connection is held only for the `async with` block (retrieval), then released
    before the connection-free generation phase runs.

    Also emits `{"type": "progress", "stage": ...}` frames at each real phase boundary
    (translating / reranking / generating) -- CPU retrieval alone can now run into the tens
    of minutes, and a bare elapsed-seconds counter gives a waiting user no sense of whether
    anything is actually happening. See agroteca.generate.retrieve_ranked_with_progress for
    why the granularity stops at phase boundaries rather than a fabricated percentage.
    """
    async def gen():
        async with request.app.state.pool.connection() as conn:
            async for kind, value in retrieve_ranked_with_progress(conn, payload.question):
                if kind == "progress":
                    yield json.dumps({"type": "progress", "stage": value}) + "\n"
                else:
                    rows, meta, retrieval_ms, translated = value
        meta_line, context = build_meta_and_context(rows, meta, retrieval_ms, translated)
        yield meta_line
        yield json.dumps({"type": "progress", "stage": "generating"}) + "\n"
        async for line in stream_ndjson_async(payload.question, context):
            yield line

    return StreamingResponse(gen(), media_type="application/x-ndjson")
