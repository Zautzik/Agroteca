from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agroteca.generate import answer_ndjson
from agroteca.ingest import store

app = FastAPI(title="Agroteca")
STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream(payload: Question):
    """Stream the answer as NDJSON: a meta frame (retrieved sources + scores +
    tiers + timing), then token frames, then a done frame (generation timing).
    Sync `def` so the blocking, CPU-bound work runs in a threadpool."""
    conn = store.connect()

    def gen():
        try:
            for line in answer_ndjson(conn, payload.question):
                yield line
        finally:
            conn.close()

    return StreamingResponse(gen(), media_type="application/x-ndjson")
