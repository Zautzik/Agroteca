"""Tiny streaming client — prints the /ask/stream answer token-by-token as it arrives.

    uv run python stream_client.py
"""
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # clean ES accents in the console

QUESTION = "¿Cuánta agua se puede captar por cada milímetro de lluvia sobre un metro cuadrado de techo?"

print(f"Q: {QUESTION}\n")
print("(retrieving + reranking — the first tokens take a moment on CPU...)\n")

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/ask/stream",
    json={"question": QUESTION},
    timeout=None,  # generation is slow; don't time out
) as r:
    for text in r.iter_text():
        print(text, end="", flush=True)   # flush=True -> shows each token live

print()  # final newline
