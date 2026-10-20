"""Persistent cache for the paid-by-quota calls in the serving path (profile=hosted).

Jina's free rerank allowance is a one-time grant, and most demo visitors ask the same few
example questions -- so the rerank scores for an exact (model, query, candidate texts) input are
stored in Postgres (`kv_cache`, migration 004) and reused. Query translations are cached too:
they cost Groq requests, and a stable translation keeps the bilingual candidate pool identical,
which is what lets the rerank cache hit on a repeat.

Correctness: a hit returns exactly what the same call returned before (retrieval and the corpus
are deterministic; translation runs at temperature 0), keyed on the full inputs, so a hit can
never answer a different question. Failure: every cache operation runs inside a savepoint and
fails soft -- a missing table or a hiccup costs one uncached call, never a broken request.
Off by default locally, where reranking is a free local model.
"""
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable

from psycopg.types.json import Jsonb

from agroteca.config import settings

log = logging.getLogger(__name__)

_GET = "UPDATE kv_cache SET hits = hits + 1 WHERE key = %s RETURNING value"
_PUT = "INSERT INTO kv_cache (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING"


def enabled() -> bool:
    return settings.hosted if settings.cache_enabled is None else settings.cache_enabled


def make_key(kind: str, *parts) -> str:
    """Stable key for an exact input: kind + SHA-256 of the JSON-encoded parts."""
    blob = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{kind}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"


async def get(conn, key: str):
    try:
        async with conn.transaction():          # savepoint: a failure can't poison the request
            cur = await conn.execute(_GET, (key,))
            row = await cur.fetchone()
        return None if row is None else row[0]
    except Exception as exc:                    # noqa: BLE001 -- fail soft, by design
        log.warning("cache get failed (%s): %s", type(exc).__name__, exc)
        return None


async def put(conn, key: str, value) -> None:
    try:
        async with conn.transaction():
            await conn.execute(_PUT, (key, Jsonb(value)))
    except Exception as exc:                    # noqa: BLE001
        log.warning("cache put failed (%s): %s", type(exc).__name__, exc)


async def cached(conn, key: str, compute: Callable[[], Awaitable], valid=lambda v: True):
    """Return the cached value for `key`, or compute it, store it, and return it."""
    if not enabled():
        return await compute()
    hit = await get(conn, key)
    if hit is not None and valid(hit):
        return hit
    value = await compute()
    if value is not None:
        await put(conn, key, value)
    return value
