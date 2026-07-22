"""Hybrid retrieval: dense + lexical candidates fused with Reciprocal Rank Fusion."""
from agroteca.retrieve.dense import dense_search
from agroteca.retrieve.fusion import reciprocal_rank_fusion
from agroteca.retrieve.lexical import lexical_search


def hybrid_search(conn, query: str, k: int = 5, n: int = 60,
                  tiers: list[str] | None = None, rrf_k: int = 60):
    """Retrieve n candidates from each retriever, fuse, return the top k."""
    dense = dense_search(conn, query, n, tiers)
    lexical = lexical_search(conn, query, n, tiers)
    return reciprocal_rank_fusion([dense, lexical], k=rrf_k, top=k)
