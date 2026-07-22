"""Reciprocal Rank Fusion — scale-free merging of ranked lists.

Each item scores Σ 1/(k + rank) across the lists it appears in (rank starts at 1).
Items are keyed by chunk_id (row[0]), so fused results keep the retrievers' row
shape (chunk_id, source_file, text). No score normalization, no tuning per corpus.
"""


def reciprocal_rank_fusion(rankings, k: int = 60, top: int | None = None):
    scores: dict = {}
    rows: dict = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            cid = row[0]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            rows.setdefault(cid, row)
    ordered = sorted(scores, key=scores.get, reverse=True)
    if top is not None:
        ordered = ordered[:top]
    return [rows[cid] for cid in ordered]
