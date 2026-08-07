import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agroteca.config import settings
from agroteca.generate import answer_ndjson
from agroteca.ingest import store

app = FastAPI(title="Agroteca")
STATIC = Path(__file__).parent / "static"
FEEDBACK_LOG = settings.root / "feedback.jsonl"


class Question(BaseModel):
    question: str


class Feedback(BaseModel):
    question: str
    vote: str                 # "up" | "down"
    answer_preview: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Cheap corpus stats for the UI ribbon (no model, one round-trip)."""
    conn = store.connect()
    try:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        tiers = conn.execute("SELECT count(DISTINCT tier) FROM chunks").fetchone()[0]
    finally:
        conn.close()
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
    """Stream the answer as NDJSON: meta (sources + scores + tiers + timing),
    then token frames, then a done frame (generation timing)."""
    conn = store.connect()

    def gen():
        try:
            for line in answer_ndjson(conn, payload.question):
                yield line
        finally:
            conn.close()

    return StreamingResponse(gen(), media_type="application/x-ndjson")
