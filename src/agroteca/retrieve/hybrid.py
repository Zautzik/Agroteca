"""Hybrid retrieval: dense + lexical candidates fused with Reciprocal Rank Fusion."""
from agroteca.retrieve.dense import dense_search, dense_search_async
from agroteca.retrieve.fusion import reciprocal_rank_fusion
from agroteca.retrieve.lexical import lexical_search, lexical_search_async


def hybrid_search(conn, query: str, k: int = 5, n: int = 60,
                  tiers: list[str] | None = None, rrf_k: int = 60):
    """Retrieve n candidates from each retriever, fuse, return the top k."""
    dense = dense_search(conn, query, n, tiers)
    lexical = lexical_search(conn, query, n, tiers)
    return reciprocal_rank_fusion([dense, lexical], k=rrf_k, top=k)


async def hybrid_search_async(conn, query: str, k: int = 5, n: int = 60,
                              tiers: list[str] | None = None, rrf_k: int = 60):
    """Async twin of hybrid_search. Dense and lexical run one after another -- psycopg
    serializes operations on a single connection internally, so gathering them wouldn't make
    the two round-trips overlap, only add scheduling overhead."""
    dense = await dense_search_async(conn, query, n, tiers)
    lexical = await lexical_search_async(conn, query, n, tiers)
    return reciprocal_rank_fusion([dense, lexical], k=rrf_k, top=k)
