"""Dense (semantic) retrieval: cosine nearest-neighbours over the pgvector index."""
from agroteca.ingest.embed import embed_query


def dense_search(conn, query: str, n: int, tiers: list[str] | None = None):
    """Return the n nearest chunks as [(chunk_id, source_file, text), ...]."""
    qv = embed_query(query)
    tier_filter, params = "", ()
    if tiers:
        tier_filter, params = "WHERE c.tier = ANY(%s)", (tiers,)
    sql = f"""
        SELECT c.chunk_id, d.source_file, c.text
        FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
        {tier_filter}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    return conn.execute(sql, (*params, qv, n)).fetchall()
