"""Dense (semantic) retrieval: cosine nearest-neighbours over the pgvector index."""
from agroteca.ingest.embed import embed_query
from agroteca.torch_runtime import run_torch

_SQL = """
    SELECT c.chunk_id, d.source_file, c.text
    FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
    {tier_filter}
    ORDER BY c.embedding <=> %s
    LIMIT %s
"""


def dense_search(conn, query: str, n: int, tiers: list[str] | None = None):
    """Return the n nearest chunks as [(chunk_id, source_file, text), ...]."""
    qv = embed_query(query)
    tier_filter, params = "", ()
    if tiers:
        tier_filter, params = "WHERE c.tier = ANY(%s)", (tiers,)
    sql = _SQL.format(tier_filter=tier_filter)
    return conn.execute(sql, (*params, qv, n)).fetchall()


async def dense_search_async(conn, query: str, n: int, tiers: list[str] | None = None):
    """Async twin of dense_search for the FastAPI serving path (an awaited pooled connection).
    embed_query is a blocking torch call, so it runs on the single dedicated torch worker
    thread (agroteca.torch_runtime) rather than the event loop or asyncio.to_thread's ad-hoc
    pool -- see that module for why ad-hoc threads badly oversubscribe torch's CPU backend."""
    qv = await run_torch(embed_query, query)
    tier_filter, params = "", ()
    if tiers:
        tier_filter, params = "WHERE c.tier = ANY(%s)", (tiers,)
    sql = _SQL.format(tier_filter=tier_filter)
    cur = await conn.execute(sql, (*params, qv, n))
    return await cur.fetchall()
