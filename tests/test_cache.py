"""The quota-saving cache (agroteca.cache): a repeat must not call the reranker or translator
again, a hit must be keyed on the exact inputs, and any cache failure must fall back to a normal
call. No database: a fake async connection plays Postgres."""
import asyncio

import pytest

from agroteca import cache, generate
from agroteca.retrieve import rerank


class FakeConn:
    """Minimal async psycopg stand-in: a dict behind the two cache statements, savepoints that
    roll back on error, and an optional failure switch."""

    def __init__(self, fail: bool = False):
        self.store, self.fail = {}, fail

    def transaction(self):
        class _Tx:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False          # don't swallow: cache.get/put catch it themselves
        return _Tx()

    async def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("relation kv_cache does not exist")
        key = params[0]
        row = None
        if sql.startswith("UPDATE kv_cache"):
            row = (self.store[key],) if key in self.store else None
        elif sql.startswith("INSERT INTO kv_cache"):
            self.store.setdefault(key, params[1].obj)       # psycopg Jsonb wrapper

        class _Cur:
            async def fetchone(self_inner):
                return row
        return _Cur()


@pytest.fixture
def cache_on(monkeypatch):
    monkeypatch.setattr(cache.settings, "cache_enabled", True)


def test_keys_are_stable_and_input_sensitive():
    a = cache.make_key("rerank", "m", "q", ["x", "y"])
    assert a == cache.make_key("rerank", "m", "q", ["x", "y"])
    assert a != cache.make_key("rerank", "m", "q", ["y", "x"])        # order matters
    assert a != cache.make_key("rerank", "m", "q2", ["x", "y"])
    assert a != cache.make_key("translate", "m", "q", ["x", "y"])
    assert a.startswith("rerank:") and len(a) == len("rerank:") + 64


def test_cache_is_off_locally_by_default(monkeypatch):
    monkeypatch.setattr(cache.settings, "cache_enabled", None)
    monkeypatch.setattr(cache.settings, "profile", "local")
    assert cache.enabled() is False
    monkeypatch.setattr(cache.settings, "profile", "hosted")
    assert cache.enabled() is True


def _count_reranks(monkeypatch):
    calls = []

    class FakeReranker:
        def rerank(self, query, texts):
            calls.append(query)
            return [float(len(t)) for t in texts]

    async def direct(func, *a, **kw):            # skip the torch worker thread in tests
        return func(*a, **kw)

    monkeypatch.setattr(rerank, "_reranker", lambda: FakeReranker())
    monkeypatch.setattr(rerank, "run_torch", direct)
    return calls


def test_repeat_rerank_is_served_from_cache(monkeypatch, cache_on):
    calls = _count_reranks(monkeypatch)
    conn = FakeConn()
    first = asyncio.run(rerank._rerank_cached(conn, "q", ["aa", "b"]))
    second = asyncio.run(rerank._rerank_cached(conn, "q", ["aa", "b"]))
    assert first == second == [2.0, 1.0]
    assert calls == ["q"]                         # the reranker ran once


def test_different_candidates_miss(monkeypatch, cache_on):
    calls = _count_reranks(monkeypatch)
    conn = FakeConn()
    asyncio.run(rerank._rerank_cached(conn, "q", ["aa", "b"]))
    asyncio.run(rerank._rerank_cached(conn, "q", ["aa", "b", "c"]))
    assert calls == ["q", "q"]


def test_wrong_length_cache_entry_is_ignored(monkeypatch, cache_on):
    calls = _count_reranks(monkeypatch)
    conn = FakeConn()
    key = cache.make_key("rerank", rerank.settings.rerank_model, "q", ["aa", "b"])
    conn.store[key] = [9.9]                       # corrupt: one score for two texts
    assert asyncio.run(rerank._rerank_cached(conn, "q", ["aa", "b"])) == [2.0, 1.0]
    assert calls == ["q"]


def test_cache_failure_falls_back_to_a_normal_call(monkeypatch, cache_on):
    calls = _count_reranks(monkeypatch)
    out = asyncio.run(rerank._rerank_cached(FakeConn(fail=True), "q", ["aa"]))
    assert out == [2.0] and calls == ["q"]


def test_cache_disabled_never_touches_the_database(monkeypatch):
    monkeypatch.setattr(cache.settings, "cache_enabled", False)
    _count_reranks(monkeypatch)
    conn = FakeConn(fail=True)                    # would raise if touched
    assert asyncio.run(rerank._rerank_cached(conn, "q", ["aa"])) == [2.0]


def test_repeat_translation_is_cached_but_failures_are_retried(monkeypatch, cache_on):
    calls = []
    results = iter([None, "What is X?"])

    def fake_translate(q):
        calls.append(q)
        return next(results)

    monkeypatch.setattr(generate, "translate_query", fake_translate)
    conn = FakeConn()
    assert asyncio.run(generate._translate_cached(conn, "¿Qué es X?")) is None   # failure...
    assert asyncio.run(generate._translate_cached(conn, "¿Qué es X?")) == "What is X?"  # ...retried
    assert asyncio.run(generate._translate_cached(conn, "¿Qué es X?")) == "What is X?"  # cached
    assert len(calls) == 2


def test_bilingual_rerank_with_cache_matches_max_over_both(monkeypatch, cache_on):
    calls = _count_reranks(monkeypatch)
    pool = [("c1", "a.pdf", "aaa"), ("c2", "b.pdf", "b")]

    async def fake_hybrid(conn, q, k, tiers=None):
        return pool

    monkeypatch.setattr(rerank, "hybrid_search_async", fake_hybrid)
    conn = FakeConn()
    rows = asyncio.run(rerank.rerank_scored_bilingual_async(conn, "es", "en", k=2))
    assert [r[0] for r in rows] == ["c1", "c2"] and [r[3] for r in rows] == [3.0, 1.0]
    asyncio.run(rerank.rerank_scored_bilingual_async(conn, "es", "en", k=2))
    assert calls == ["es", "en"]                  # second request: both queries cached
