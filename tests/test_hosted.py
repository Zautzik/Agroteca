"""Hosted-profile pieces, tested without network, database, model download or API keys:
ONNX mean pooling, the Jina rerank client, OpenAI-compatible SSE streaming, the quota guards,
and the promise that importing the served app under profile=hosted never loads torch."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from agroteca import quota
from agroteca.ingest.onnx_embed import mean_pool
from agroteca.llm import OpenAICompatible, _sse_deltas
from agroteca.retrieve.rerank import _JinaAPIReranker

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- ONNX mean pooling

def test_mean_pool_ignores_padding_tokens():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])   # 3rd token is padding
    mask = np.array([[1, 1, 0]])
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_matches_plain_mean_without_padding():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(1, 7, 384)).astype(np.float32)
    np.testing.assert_allclose(mean_pool(hidden, np.ones((1, 7), dtype=np.int64)),
                               hidden.mean(axis=1), rtol=1e-6)


# ---------------------------------------------------------------- Jina rerank client

def _jina(handler) -> _JinaAPIReranker:
    return _JinaAPIReranker("https://api.jina.test/v1/rerank", "jina-reranker-v2-base-multilingual",
                            "test-key", 5.0, transport=httpx.MockTransport(handler))


def test_jina_scores_come_back_in_input_order():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        # The API returns results sorted by relevance, each carrying its input index.
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]})

    scores = _jina(handler).rerank("q", ["a", "b", "c"])
    assert scores == [0.5, 0.1, 0.9]
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["top_n"] == 3 and seen["body"]["documents"] == ["a", "b", "c"]
    assert seen["body"]["model"] == "jina-reranker-v2-base-multilingual"


def test_jina_rejects_a_response_that_skips_a_document():
    def handler(request):
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    with pytest.raises(RuntimeError, match="every document"):
        _jina(handler).rerank("q", ["a", "b"])


def test_jina_http_errors_are_raised_not_swallowed():
    # A non-rate-limit failure (bad key) still raises; 429/503 have their own retry path below.
    with pytest.raises(httpx.HTTPStatusError):
        _jina(lambda r: httpx.Response(401, json={"detail": "bad key"})).rerank("q", ["a"])


def test_jina_requires_a_key():
    with pytest.raises(RuntimeError, match="JINA_API_KEY"):
        _JinaAPIReranker("https://x", "m", "", 5.0)


def test_jina_empty_pool_makes_no_request():
    def handler(request):
        raise AssertionError("no HTTP call expected for an empty pool")

    assert _jina(handler).rerank("q", []) == []


# ---------------------------------------------------------------- OpenAI-compatible streaming

_SSE = [
    'data: {"choices":[{"delta":{"role":"assistant"}}]}',
    "",
    ": keep-alive comment",
    'data: {"choices":[{"delta":{"content":"Hola"}}]}',
    'data: {"choices":[{"delta":{"content":""}}]}',
    'data: {"choices":[{"delta":{"content":" mundo"}}]}',
    'data: {"choices":[]}',
    "data: [DONE]",
    'data: {"choices":[{"delta":{"content":"after-done"}}]}',
]


def test_sse_parser_yields_only_content_and_stops_at_done():
    assert list(_sse_deltas(iter(_SSE))) == ["Hola", " mundo"]


def _sse_body() -> bytes:
    return ("\n".join(_SSE) + "\n").encode()


def test_openai_compatible_sync_stream_and_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse_body(),
                              headers={"content-type": "text/event-stream"})

    llm = OpenAICompatible("https://api.groq.test/openai/v1", "k", "some-model", 5.0,
                           transport=httpx.MockTransport(handler))
    assert "".join(llm.chat_stream("SYS", "USER", 64)) == "Hola mundo"
    assert seen["url"] == "https://api.groq.test/openai/v1/chat/completions"
    assert seen["body"]["messages"] == [{"role": "system", "content": "SYS"},
                                        {"role": "user", "content": "USER"}]
    assert seen["body"]["stream"] is True and seen["body"]["max_tokens"] == 64


def test_openai_compatible_async_stream():
    async def handler(request):
        return httpx.Response(200, content=_sse_body(),
                              headers={"content-type": "text/event-stream"})

    llm = OpenAICompatible("https://api.groq.test/openai/v1", "k", "m", 5.0,
                           async_transport=httpx.MockTransport(handler))

    async def collect():
        return [p async for p in llm.achat_stream(None, "USER", 64)]

    assert "".join(asyncio.run(collect())) == "Hola mundo"


def test_openai_compatible_one_shot_reply():
    def handler(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "respuesta"}}]})

    llm = OpenAICompatible("https://x/v1", "k", "m", 5.0, transport=httpx.MockTransport(handler))
    assert llm.chat(None, "q", 16) == "respuesta"


def test_openai_compatible_requires_a_key():
    with pytest.raises(RuntimeError, match="GEN_API_KEY"):
        OpenAICompatible("https://x/v1", "", "m", 5.0)


# ---------------------------------------------------------------- quota guards

def test_ip_hash_is_salted_and_never_the_raw_ip():
    a, b = quota.hash_ip("203.0.113.7", "salt-1"), quota.hash_ip("203.0.113.7", "salt-2")
    assert a != b and "203.0.113.7" not in a and len(a) == 64
    assert a == quota.hash_ip("203.0.113.7", "salt-1")


def test_client_ip_prefers_first_forwarded_hop():
    assert quota.client_ip({"x-forwarded-for": "198.51.100.1, 10.0.0.1"}, "10.9.9.9") == "198.51.100.1"
    assert quota.client_ip({}, "10.9.9.9") == "10.9.9.9"
    assert quota.client_ip({}, None) == "unknown"


class _FakeConn:
    """Stands in for the async connection: returns a scripted count per statement."""

    def __init__(self, daily: int, ip: int):
        self._counts = {"usage_daily": daily, "usage_ip_hourly": ip}
        self.calls = []

    async def execute(self, sql, params=None):
        table = "usage_daily" if "usage_daily" in sql.split("(")[0] else "usage_ip_hourly"
        self.calls.append(table)
        count = self._counts[table]

        class _Cur:
            async def fetchone(self_inner):
                return (count,)
        return _Cur()


@pytest.mark.parametrize("daily,ip,expected", [
    (10, 1, None), (101, 1, "daily_cap"), (10, 6, "ip_cap"), (100, 5, None),
])
def test_admit_enforces_both_caps(monkeypatch, daily, ip, expected):
    monkeypatch.setattr(quota.settings, "daily_request_cap", 100)
    monkeypatch.setattr(quota.settings, "per_ip_hourly_cap", 5)
    monkeypatch.setattr(quota.settings, "ip_hash_salt", "s")
    assert asyncio.run(quota.admit(_FakeConn(daily, ip), "1.2.3.4")) == expected


def test_ip_rejection_does_not_consume_the_daily_cap(monkeypatch):
    monkeypatch.setattr(quota.settings, "daily_request_cap", 100)
    monkeypatch.setattr(quota.settings, "per_ip_hourly_cap", 5)
    monkeypatch.setattr(quota.settings, "ip_hash_salt", "s")
    conn = _FakeConn(daily=1, ip=6)
    assert asyncio.run(quota.admit(conn, "1.2.3.4")) == "ip_cap"
    assert conn.calls == ["usage_ip_hourly"]          # daily counter never touched


def test_admit_is_a_no_op_when_caps_are_zero(monkeypatch):
    monkeypatch.setattr(quota.settings, "daily_request_cap", 0)
    monkeypatch.setattr(quota.settings, "per_ip_hourly_cap", 0)
    conn = _FakeConn(10**9, 10**9)
    assert asyncio.run(quota.admit(conn, "1.2.3.4")) is None and conn.calls == []


# ---------------------------------------------------------------- the hosted import promise

def test_hosted_app_import_does_not_load_torch_or_ollama():
    code = ("import sys, app; "
            "print(','.join(m for m in ('torch','transformers','sentence_transformers','ollama') "
            "if m in sys.modules))")
    env = {**os.environ, "AGROTECA_PROFILE": "hosted"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------- local Ollama path unchanged

def test_ollama_wrapper_sends_the_same_request_as_before(monkeypatch):
    """The local profile must call Ollama exactly as the pre-refactor code did."""
    ollama = pytest.importorskip("ollama")
    calls = []

    class FakeClient:
        def __init__(self, host, timeout):
            calls.append(("init", host))

        def chat(self, **kw):
            calls.append(("chat", kw))
            if kw.get("stream"):
                return iter([{"message": {"content": "a"}}, {"message": {"content": "b"}}])
            return {"message": {"content": "reply"}}

    monkeypatch.setattr(ollama, "Client", FakeClient)
    monkeypatch.setattr(ollama, "AsyncClient", FakeClient)
    from agroteca.llm import Ollama
    llm = Ollama("http://localhost:11434", "qwen2.5:3b", 30.0)
    assert llm.chat("SYS", "USER", 1024) == "reply"
    assert "".join(llm.chat_stream("SYS", "USER", 1024)) == "ab"
    chat_calls = [c[1] for c in calls if c[0] == "chat"]
    assert chat_calls[0] == {"model": "qwen2.5:3b",
                             "messages": [{"role": "system", "content": "SYS"},
                                          {"role": "user", "content": "USER"}],
                             "options": {"num_predict": 1024}}
    assert chat_calls[1]["stream"] is True and chat_calls[1]["options"] == {"num_predict": 1024}
    # translation sends no system message, as before
    llm.chat(None, "Q", 128)
    last = [c[1] for c in calls if c[0] == "chat"][-1]
    assert last["messages"] == [{"role": "user", "content": "Q"}]
    assert last["options"] == {"num_predict": 128}


# ---------------------------------------------------------------- merged with main's async eval path

def test_openai_compatible_async_one_shot():
    """ask_ollama_async (used by eval/eval_generation.py) goes through achat on both backends."""
    async def handler(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatible("https://x/v1", "k", "m", 5.0,
                           async_transport=httpx.MockTransport(handler))
    assert asyncio.run(llm.achat("SYS", "USER", 16)) == "ok"


def test_ask_ollama_async_uses_the_backend(monkeypatch):
    from agroteca import generate

    class FakeBackend:
        async def achat(self, system, user, max_tokens):
            return f"{system}|{user}|{max_tokens}"

    monkeypatch.setattr(generate, "_llm", FakeBackend())
    out = asyncio.run(generate.ask_ollama_async("S", "U"))
    assert out == f"S|U|{generate.settings.gen_num_predict}"


def test_hosted_feedback_writes_the_abstention_context(monkeypatch):
    """Hosted feedback goes to Postgres with the same fields main records in feedback.jsonl."""
    from fastapi.testclient import TestClient
    from agroteca import api

    executed = []

    class FakeConn:
        async def execute(self, sql, params=None):
            executed.append((sql, params))

    class FakePool:
        def connection(self):
            class _Ctx:
                async def __aenter__(self_inner):
                    return FakeConn()

                async def __aexit__(self_inner, *a):
                    return False
            return _Ctx()

    monkeypatch.setattr(api.settings, "profile", "hosted")
    api.app.state.pool = FakePool()
    client = TestClient(api.app)            # no `with`: lifespan (DB + models) is not started
    resp = client.post("/feedback", json={
        "question": "q", "vote": "down", "abstained": True, "kind": "near_miss",
        "top_score": 0.12, "retrieved": [{"source": "a.pdf", "score": 0.12}] * 10,
    })
    assert resp.status_code == 200
    sql, params = executed[0]
    assert "INSERT INTO feedback" in sql and "retrieved" in sql
    assert params[3] is True and params[4] == "near_miss"
    assert len(params[7].obj) == 8                  # truncated to 8, like the local log


def test_quota_rejection_is_a_429_with_a_readable_detail(monkeypatch):
    from fastapi.testclient import TestClient
    from agroteca import api

    async def fake_admit(conn, ip):
        return "daily_cap"

    class FakePool:
        def connection(self):
            class _Ctx:
                async def __aenter__(self_inner):
                    return object()

                async def __aexit__(self_inner, *a):
                    return False
            return _Ctx()

    monkeypatch.setattr(api.settings, "daily_request_cap", 20)
    monkeypatch.setattr(api, "admit", fake_admit)
    api.app.state.pool = FakePool()
    resp = TestClient(api.app).post("/ask/stream", json={"question": "hola"})
    assert resp.status_code == 429
    assert "limit" in resp.json()["detail"]


def test_reasoning_effort_is_sent_only_when_configured():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    t = httpx.MockTransport(handler)
    OpenAICompatible("https://x/v1", "k", "openai/gpt-oss-120b", 5.0, transport=t,
                     reasoning_effort="low").chat(None, "q", 32)
    OpenAICompatible("https://x/v1", "k", "plain-model", 5.0, transport=t).chat(None, "q", 32)
    assert bodies[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in bodies[1]       # a non-reasoning model would reject it


def test_onnx_runtime_points_hf_caches_at_writable_tmp(monkeypatch, tmp_path):
    """Vercel's filesystem is read-only outside /tmp; huggingface_hub's xet backend crashed the
    first deploy by writing under ~/.cache. _runtime must default HF_HOME into the cache dir and
    disable xet -- without overriding values set explicitly in the environment."""
    import os
    from agroteca.ingest import onnx_embed

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")                 # explicit value must survive
    monkeypatch.setattr(onnx_embed.settings, "onnx_cache_dir", tmp_path)

    class Stop(Exception):
        pass

    def boom(*a, **kw):
        raise Stop
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    onnx_embed._runtime.cache_clear()
    with pytest.raises(Stop):
        onnx_embed._runtime()
    onnx_embed._runtime.cache_clear()
    assert os.environ["HF_HOME"] == str(tmp_path / "hf-home")
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


def _jina_with_waits(handler, retries=2):
    from agroteca.retrieve.rerank import _JinaAPIReranker
    waits = []
    r = _JinaAPIReranker("https://jina.test/v1/rerank", "m", "k", 5.0,
                         transport=httpx.MockTransport(handler), max_retries=retries,
                         sleep=waits.append)
    return r, waits


def test_jina_rate_limit_is_retried_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})

    r, waits = _jina_with_waits(handler)
    assert r.rerank("q", ["a"]) == [0.7]
    assert len(calls) == 2 and waits == [3.0]           # honoured the server's Retry-After


def test_jina_rate_limit_that_persists_is_a_clean_busy_not_a_crash():
    from agroteca.retrieve.rerank import RerankerBusy
    from agroteca.torch_runtime import TorchWorkerBusy

    r, waits = _jina_with_waits(lambda request: httpx.Response(429))
    with pytest.raises(RerankerBusy) as info:
        r.rerank("q", ["a"])
    assert isinstance(info.value, TorchWorkerBusy)      # -> the API's existing 503 path
    assert waits == [2.0, 4.0]                          # backoff, bounded retries


def test_retrieve_returns_503_not_500_when_the_reranker_is_busy(monkeypatch):
    """End to end through FastAPI: a persistent upstream rate limit must reach the client as a
    readable 503, the way the hosted demo's visitors would see it."""
    from fastapi.testclient import TestClient
    from agroteca import api
    from agroteca.retrieve.rerank import RerankerBusy

    async def busy(conn, question):
        raise RerankerBusy("The demo is busy right now")

    async def no_quota(request):
        return None

    async def fake_conn():
        yield object()

    monkeypatch.setattr(api, "retrieve_ranked_async", busy)
    monkeypatch.setattr(api, "_quota_rejection", no_quota)
    api.app.dependency_overrides[api.get_conn] = fake_conn
    try:
        client = TestClient(api.app)          # no `with`: lifespan (models, DB pool) not started
        resp = client.post("/retrieve", json={"question": "hola"})
    finally:
        api.app.dependency_overrides.clear()
    assert resp.status_code == 503 and "busy" in resp.json()["detail"]
    assert resp.json()["reason"] == "busy"


def test_ask_stream_busy_is_a_tagged_error_frame_the_ui_can_localize(monkeypatch):
    """The stream has already sent its 200 by the time reranking runs, so a busy rejection is
    an NDJSON error frame. It carries reason="busy" so the page can say "busy, try again in a
    minute" in its own language instead of showing a raw error."""
    from fastapi.testclient import TestClient
    from agroteca import api
    from agroteca.retrieve.rerank import RerankerBusy

    async def busy(conn, question):
        raise RerankerBusy("The demo is busy right now")
        yield  # pragma: no cover -- makes this an async generator, like the real one

    async def no_quota(request):
        return None

    class FakePool:
        def connection(self):
            class _Ctx:
                async def __aenter__(self_inner):
                    return object()

                async def __aexit__(self_inner, *a):
                    return False
            return _Ctx()

    monkeypatch.setattr(api, "retrieve_ranked_with_progress", busy)
    monkeypatch.setattr(api, "_quota_rejection", no_quota)
    api.app.state.pool = FakePool()
    resp = TestClient(api.app).post("/ask/stream", json={"question": "hola"})
    frames = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert resp.status_code == 200
    assert frames == [{"type": "error", "reason": "busy", "message": "The demo is busy right now"}]


def test_public_demo_page_never_tells_visitors_to_check_uvicorn():
    """The page's error copy: the local build may suggest checking the server, the hosted demo
    must not (a visitor can't act on it), and busy/quota states exist in both languages."""
    from agroteca import api

    page = (Path(api.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    for line in page.splitlines():
        if "errorHosted:" in line:
            assert "uvicorn" not in line
    for key in ("errorHosted:", "busy:", "daily_cap:", "ip_cap:"):
        assert page.count(key) == 2, key           # one Spanish, one English
    assert "(isHosted() ? t.errorHosted : t.error)" in page
    assert "msg.reason === 'busy'" in page
    assert "!known ? ''" in page                   # no footer claim until /stats says which build


def test_stats_reports_the_profile_so_the_ui_never_claims_offline_when_hosted(monkeypatch):
    from fastapi.testclient import TestClient
    from agroteca import api

    class Cur:
        async def fetchone(self):
            return (7,)

    class Conn:
        async def execute(self, sql, params=None):
            return Cur()

    async def fake_conn():
        yield Conn()

    monkeypatch.setattr(api.settings, "profile", "hosted")
    api.app.dependency_overrides[api.get_conn] = fake_conn
    try:
        body = TestClient(api.app).get("/stats").json()
    finally:
        api.app.dependency_overrides.clear()
    assert body == {"chunks": 7, "documents": 7, "tiers": 7, "profile": "hosted"}
