from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse   # add near the top
from agroteca.generate import answer, answer_stream  
from pathlib import Path
from fastapi.responses import FileResponse # add answer_stream to this import

from agroteca.generate import answer
from agroteca.ingest import store

app = FastAPI(title="Agroteca")

STATIC = Path(__file__).parent / "static"

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


class Question(BaseModel):      # the shape of the POST body: {"question": "..."}
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream(payload: Question):
    conn = store.connect()
    def gen():
        try:
            for token in answer_stream(conn, payload.question):
                yield token
        finally:
            conn.close()          # close only after streaming finishes
    return StreamingResponse(gen(), media_type="text/plain")
