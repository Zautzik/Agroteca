import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agroteca.config import settings
from agroteca.generate import prepare_ndjson, stream_ndjson
from agroteca.ingest import store
from agroteca.ingest.embed import embed_query
from agroteca.retrieve.rerank import warm as warm_reranker

# A pooled DB connection for the served API — reused across requests instead of a fresh
# TCP handshake + auth per question. Opened in the lifespan, below.
pool = store.make_pool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the heavy models once at startup. FastAPI runs sync endpoints in a threadpool,
    # so without this the first few concurrent requests could each trip the
    # lru_cache(maxsize=1) cold-start and load the 1.1 GB reranker several times over.
    warm_reranker()          # load + cache the cross-encoder
    embed_query("warm")      # load + cache the embedder
    pool.open()
    pool.wait()              # block until the pool has a live connection ready
    yield
    pool.close()


app = FastAPI(title="Agroteca", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
FEEDBACK_LOG = settings.root / "feedback.jsonl"


class Question(BaseModel):
    # Bounded: an unbounded string would flow straight into the FTS regex + tsquery.
    question: str = Field(..., min_length=1, max_length=2000)


class Feedback(BaseModel):
    question: str = Field(..., max_length=2000)
    vote: Literal["up", "down"]
    answer_preview: str = Field("", max_length=5000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Cheap corpus stats for the UI ribbon (no model, one round-trip)."""
    with pool.connection() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        tiers = conn.execute("SELECT count(DISTINCT tier) FROM chunks").fetchone()[0]
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


@app.post("/ask/stream")
def ask_stream(payload: Question):
    """Stream the answer as NDJSON: meta (sources + scores + tiers + timing), then token
    frames, then a done frame. The pooled connection is held only for retrieval, then
    returned to the pool BEFORE the (connection-free) generation stream."""
    with pool.connection() as conn:
        meta_line, context = prepare_ndjson(conn, payload.question)

    def gen():
        yield meta_line
        yield from stream_ndjson(payload.question, context)

    return StreamingResponse(gen(), media_type="application/x-ndjson")
