"""Tiny streaming client — reads the /ask/stream NDJSON and prints the answer live.

    uv run python stream_client.py
"""
import json
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # clean ES accents in the console

QUESTION = "¿Cuánta agua se puede captar por cada milímetro de lluvia sobre un metro cuadrado de techo?"

print(f"Q: {QUESTION}\n")
print("(retrieving + reranking — the first tokens take a moment on CPU...)\n")

with httpx.stream(
    "POST", "http://127.0.0.1:8000/ask/stream",
    json={"question": QUESTION}, timeout=None,   # generation is slow; don't time out
) as r:
    buf = ""
    for chunk in r.iter_text():
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg["type"] == "meta":
                print(f"[retrieved {len(msg['sources'])} chunks in {msg['retrieval_ms'] / 1000:.0f}s]\n")
            elif msg["type"] == "token":
                print(msg["text"], end="", flush=True)
            elif msg["type"] == "done":
                print(f"\n\n[generated in {msg['generation_ms'] / 1000:.0f}s]")
