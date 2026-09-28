import json
import threading
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
    corpus_topic_counts_async,
    is_abstention,
    retrieve_ranked_async,
    retrieve_ranked_with_progress,
    stream_ndjson_async,
    topics_for_async,
)
from agroteca.ingest import store
from agroteca.suggest import classify, nearby_documents, rank_topics
from agroteca.torch_runtime import TorchWorkerBusy
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
# feedback() is a sync route, so FastAPI dispatches concurrent calls onto DIFFERENT threadpool
# threads. Nothing in the stdlib guarantees an open("a") + write() from independent threads
# never interleaves mid-line -- a lock costs nothing on a request this infrequent and removes
# the question entirely.
_FEEDBACK_LOCK = threading.Lock()


async def get_conn(request: Request) -> AsyncIterator[AsyncConnection]:
    """DI hook: check out a pooled async connection for the route. Routes take it via
    `Depends(get_conn)` instead of touching the pool directly; FastAPI closes this generator
    (returning the connection to the pool) as soon as the route function returns."""
    async with request.app.state.pool.connection() as conn:
        yield conn


class Question(BaseModel):
    # Bounded: an unbounded string would flow straight into the FTS regex + tsquery.
    question: str = Field(..., min_length=1, max_length=2000)


class FeedbackSource(BaseModel):
    """One retrieved chunk as the UI saw it — what the system surfaced when it failed."""
    source: str = Field("", max_length=300)
    score: float | None = None


class Feedback(BaseModel):
    question: str = Field(..., max_length=2000)
    vote: Literal["up", "down"]
    answer_preview: str = Field("", max_length=5000)
    # Everything below is optional so the existing {question, vote, answer_preview}
    # payload keeps working unchanged. These capture the case the log never saw: an
    # abstention, where the system said it could not answer. Those are the highest-signal
    # failures and currently the least instrumented -- the vote buttons don't even render
    # on that branch yet. `top_score` and `retrieved` are what let a later, larger
    # re-run of eval/calibrate_abstention.py work from real traffic instead of the
    # golden set's two no-answer questions.
    abstained: bool = False
    # Deliberately a free string, not a Literal: whether "near miss" vs "total miss" is
    # even distinguishable is still an open question (see eval/calibrate_abstention.py).
    # Logging an unproven taxonomy as a closed enum would bake in the answer.
    kind: str | None = Field(None, max_length=32)
    top_score: float | None = None
    translated_query: str | None = Field(None, max_length=2000)
    retrieved: list[FeedbackSource] = Field(default_factory=list)


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
    """Append a thumbs-up/down to a JSONL log — read back by eval/harvest_feedback.py.

    Retrieval context is recorded alongside the vote so a failure can be triaged later
    without re-running the query (which costs a full rerank, and on CPU that is minutes).
    Lists are truncated on write rather than rejected on input: a payload slightly over
    the expected size should still record the vote, not 422 away a real user's signal."""
    rec = {"ts": time.time(), "vote": fb.vote,
           "question": fb.question, "answer_preview": fb.answer_preview[:200],
           "abstained": fb.abstained, "kind": fb.kind, "top_score": fb.top_score,
           "translated_query": fb.translated_query,
           "retrieved": [{"source": s.source, "score": s.score} for s in fb.retrieved[:8]]}
    with _FEEDBACK_LOCK, open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
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
    try:
        rows, meta, retrieval_ms, translated = await retrieve_ranked_async(conn, payload.question)
    except TorchWorkerBusy as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
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

    If the dedicated torch worker is already saturated (agroteca.torch_runtime.TorchWorkerBusy
    -- see that module for why this can happen even from a single disconnected client), the
    rejection can't be an HTTP status code: by the time reranking runs, StreamingResponse has
    already returned and Starlette has already sent a 200. It surfaces as a terminal `error`
    NDJSON frame instead, so the client's parser gets an explicit end state instead of hanging
    on a `done` frame that will never arrive.
    """
    async def gen():
        try:
            async with request.app.state.pool.connection() as conn:
                async for kind, value in retrieve_ranked_with_progress(conn, payload.question):
                    if kind == "progress":
                        yield json.dumps({"type": "progress", "stage": value}) + "\n"
                    else:
                        rows, meta, retrieval_ms, translated = value
        except TorchWorkerBusy as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return
        meta_line, context = build_meta_and_context(rows, meta, retrieval_ms, translated)
        yield meta_line
        yield json.dumps({"type": "progress", "stage": "generating"}) + "\n"

        # Accumulate the reply as it streams so the server can tell whether the model
        # abstained. Until now only the browser knew -- it string-matched the canonical
        # phrase client-side -- which meant the single highest-signal outcome produced no
        # server-side event at all: nothing logged, nothing offered to the reader.
        spoken: list[str] = []
        async for line in stream_ndjson_async(payload.question, context):
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if frame.get("type") == "token":
                spoken.append(frame.get("text", ""))
            elif frame.get("type") == "done" and is_abstention("".join(spoken)):
                # Emitted BEFORE `done`, because the client finalises its view on `done`.
                yield await _no_answer_frame(request, payload.question, rows, meta)
            yield line

    return StreamingResponse(gen(), media_type="application/x-ndjson")


async def _no_answer_frame(request: Request, question: str, rows, meta) -> str:
    """Build the off-ramp frame for an abstention: what retrieval *did* surface.

    Every field is derived from rows already retrieved or from real corpus metadata -- the
    model is not asked to invent a rephrasing, because a plausible-sounding suggestion that
    also fails costs the reader another multi-minute query and spends the one property this
    system sells. Failing soft is deliberate: an off-ramp is a nicety, and no error here is
    worth converting an honest 'I don't know' into a broken stream."""
    servable = settings.tiers()
    try:
        async with request.app.state.pool.connection() as conn:
            topics = await topics_for_async(conn, [r[0] for r in rows])
            counts = await corpus_topic_counts_async(conn, servable)
    except Exception:
        topics, counts = {}, []

    candidates = [(topics.get(cid), meta.get(cid, ("?", None))[0], score)
                  for (cid, _sf, _text, score) in rows]
    lang = "es" if (request.headers.get("accept-language", "es").lower().startswith("es")) else "en"
    chips = rank_topics(candidates, lang=lang, servable_tiers=frozenset(servable),
                        fallback_counts=counts)
    return json.dumps({
        "type": "no_answer",
        "kind": classify(candidates),
        "top_score": round(max((c[2] for c in candidates), default=0.0), 4),
        "topics": [c.as_dict for c in chips],
        "documents": nearby_documents(
            [(sf, meta.get(cid, ("?", None))[0], score) for (cid, sf, _t, score) in rows],
            servable_tiers=frozenset(servable),
        ),
    }, ensure_ascii=False) + "\n"
