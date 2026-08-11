"""Regression tests for the assumptions the retrieval metrics depend on.

Each guards a claim that is load-bearing yet invisible by inspection — a cross-encoder's
score direction, the reciprocal-rank math, the file-scoped hit rule. None would raise if
silently broken; they would only skew the numbers. Pinning them here makes a refactor that
inverts one fail fast, rather than quietly corrupting an eval.

The reranker test loads the cross-encoder (~1.1 GB) once; the rest are pure and instant.
"""
from _scoring import answer_at_k, answer_rank, norm, reciprocal_rank
from agroteca.retrieve.rerank import _reranker


def test_norm_collapses_whitespace_and_lowercases():
    # Extraction inserts stray whitespace and line breaks; norm is what lets a verbatim
    # must_contain phrase still match. A missing field degrades to "" rather than raising.
    assert norm("  Stocking   DENSITY\n is  20 ") == "stocking density is 20"
    assert norm(None) == ""


# 1. Cross-encoder score direction — higher must mean more relevant.

def test_reranker_scores_a_relevant_chunk_higher_than_an_irrelevant_one():
    # rerank_search ranks by descending score (reverse=True), which is correct only if a
    # higher cross-encoder score means more relevant. A flipped comparator would not raise —
    # it would invert every ranking and still look plausible on individual queries.
    query = "What is the recommended stocking density for tilapia in aquaponics?"
    relevant = ("The recommended stocking density for tilapia in a small-scale aquaponic "
                "system is 20 kg of fish per cubic metre of water.")
    irrelevant = ("Ptolemy's Tetrabiblos is a foundational second-century treatise on "
                  "astrology and the movements of the planets.")

    # rerank returns one score per document, in input order.
    relevant_score, irrelevant_score = list(_reranker().rerank(query, [relevant, irrelevant]))

    # jina scores are logits — unbounded and often negative (here ~ +1.25 vs ~ -3.72).
    assert relevant_score > irrelevant_score


# 2. Rank-derived metrics — answer@k and MRR both fall out of one position.

def test_metrics_all_derive_from_one_rank():
    # answer@k is set membership within the top k...
    assert answer_at_k(3, k=1) is False
    assert answer_at_k(3, k=5) is True
    assert answer_at_k(None, k=5) is False

    # ...MRR keeps the position membership discards: 1/rank, and 0 on a miss. That is the
    # reordering signal answer@k is blind to — a move from rank 5 to 1 changes only MRR.
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


# 3. Answer-chunk hit rule — right file AND phrase in the same chunk, not either alone.

def test_answer_rank_requires_the_right_file_and_returns_the_first_hit():
    # Row 1 has the phrase in the wrong file; row 2 the right file without the phrase; only
    # row 3 satisfies both, so the answer-chunk rank is 3.
    phrase = "stocking density is 20 kg of fish"
    rows = [
        ("astrology.pdf",  "stocking density is 20 kg of fish"),
        ("aquaponics.pdf", "an unrelated introductory paragraph"),
        ("aquaponics.pdf", "the stocking density is 20 kg of fish per cubic metre"),
    ]
    rank = answer_rank(rows, source_file="aquaponics.pdf", must_contain=phrase)
    assert rank == 3
    # Same-file-but-wrong-chunk is indistinguishable by rank; must_contain uniqueness is
    # enforced separately, upstream of this metric.
