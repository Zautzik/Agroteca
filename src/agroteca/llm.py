"""Generation backends behind one small interface: local Ollama (the reference stack) or any
OpenAI-compatible chat-completions API (the hosted profile; Groq by default).

Three calls cover every use in the codebase -- a one-shot reply, a sync token stream (CLI/eval)
and an async token stream (the FastAPI path). The OpenAI-compatible client is plain httpx over
server-sent events rather than a vendor SDK: one fewer dependency in a size-limited serverless
bundle, and the wire format is small enough to test with a mock transport.
"""
import json
from collections.abc import AsyncIterator, Iterator

import httpx

from agroteca.config import settings


def _messages(system: str | None, user: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}]


# ---------------------------------------------------------------- OpenAI-compatible (hosted)

_DONE = object()


def _sse_piece(line: str):
    """Parse one OpenAI-style SSE line: returns a text delta, None (nothing to emit), or _DONE."""
    if not line.startswith("data:"):
        return None                       # blank keep-alives, comments, event: lines
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return _DONE
    choices = json.loads(data).get("choices") or []
    if not choices:
        return None
    return (choices[0].get("delta") or {}).get("content") or None


def _sse_deltas(lines: Iterator[str]) -> Iterator[str]:
    """Yield the text deltas from an OpenAI-style SSE stream (`data: {...}` lines, `[DONE]`)."""
    for line in lines:
        piece = _sse_piece(line)
        if piece is _DONE:
            return
        if piece:
            yield piece


class OpenAICompatible:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float,
                 transport: httpx.BaseTransport | None = None,
                 async_transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise RuntimeError("AGROTECA_GEN_API_KEY is not set (required by profile=hosted)")
        headers = {"Authorization": f"Bearer {api_key}"}
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = httpx.Timeout(timeout, connect=10.0)
        self._client = httpx.Client(timeout=self._timeout, headers=headers, transport=transport)
        self._headers = headers
        self._async_transport = async_transport

    def _body(self, system, user, max_tokens, stream):
        return {"model": self._model, "messages": _messages(system, user),
                "max_tokens": max_tokens, "temperature": 0, "stream": stream}

    def chat(self, system: str | None, user: str, max_tokens: int) -> str:
        resp = self._client.post(self._url, json=self._body(system, user, max_tokens, False))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def achat(self, system: str | None, user: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers,
                                     transport=self._async_transport) as client:
            resp = await client.post(self._url, json=self._body(system, user, max_tokens, False))
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def chat_stream(self, system: str | None, user: str, max_tokens: int) -> Iterator[str]:
        with self._client.stream("POST", self._url,
                                 json=self._body(system, user, max_tokens, True)) as resp:
            resp.raise_for_status()
            yield from _sse_deltas(resp.iter_lines())

    async def achat_stream(self, system: str | None, user: str, max_tokens: int) -> AsyncIterator[str]:
        # A client per call: an AsyncClient is bound to the event loop that created it, and a
        # serverless instance may serve requests from different loops over its lifetime.
        async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers,
                                     transport=self._async_transport) as client:
            async with client.stream("POST", self._url,
                                     json=self._body(system, user, max_tokens, True)) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    piece = _sse_piece(line)
                    if piece is _DONE:
                        return
                    if piece:
                        yield piece


# ---------------------------------------------------------------- Ollama (local reference)

class Ollama:
    def __init__(self, host: str, model: str, timeout: float):
        from ollama import AsyncClient, Client   # local-only dependency
        t = httpx.Timeout(timeout, connect=10.0)
        self._client = Client(host=host, timeout=t)
        self._async = AsyncClient(host=host, timeout=t)
        self._model = model

    def chat(self, system: str | None, user: str, max_tokens: int) -> str:
        resp = self._client.chat(model=self._model, messages=_messages(system, user),
                                 options={"num_predict": max_tokens})
        return resp["message"]["content"]

    async def achat(self, system: str | None, user: str, max_tokens: int) -> str:
        resp = await self._async.chat(model=self._model, messages=_messages(system, user),
                                      options={"num_predict": max_tokens})
        return resp["message"]["content"]

    def chat_stream(self, system: str | None, user: str, max_tokens: int) -> Iterator[str]:
        for chunk in self._client.chat(model=self._model, messages=_messages(system, user),
                                       stream=True, options={"num_predict": max_tokens}):
            yield chunk["message"]["content"]

    async def achat_stream(self, system: str | None, user: str, max_tokens: int) -> AsyncIterator[str]:
        stream = await self._async.chat(model=self._model, messages=_messages(system, user),
                                        stream=True, options={"num_predict": max_tokens})
        async for chunk in stream:
            yield chunk["message"]["content"]


def make_llm(timeout: float):
    if settings.hosted:
        return OpenAICompatible(settings.openai_base_url, settings.gen_api_key,
                                settings.openai_model, timeout)
    return Ollama(settings.gen_base_url, settings.gen_model, timeout)
