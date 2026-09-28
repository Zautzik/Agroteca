"""The no-answer off-ramp is only trustworthy if every suggestion is grounded in something
retrieval actually returned. These pin the three ways that guarantee could silently break:
recommending the planted distractor, advertising a tier this deployment cannot serve, and
sliding back to ranking by corpus composition instead of relevance to the failed query.
"""
from agroteca.suggest import classify, nearby_documents, rank_topics

DEPLOY = frozenset({"open", "synthetic"})


def test_classify_uses_the_calibrated_band():
    # Real observed scores from eval/calibrate_abstention.py.
    assert classify([("agua", "open", 0.311)]) == "no_match"     # q17, genuine no-answer
    assert classify([("agua", "open", 0.413)]) == "no_match"     # q19, genuine no-answer
    assert classify([("agua", "open", 0.354)]) == "no_match"     # q20, retrieval failed
    assert classify([("agua", "open", 0.743)]) == "weak_match"   # q07, confident but wrong
    assert classify([("agua", "open", 0.812)]) == "weak_match"   # healthy median
    assert classify([]) == "no_match"                            # nothing retrieved at all


def test_classify_reads_the_best_candidate_not_the_first():
    # Rows arrive best-first in practice, but the call must not depend on that ordering:
    # one strong candidate anywhere in the pool means retrieval did surface something.
    assert classify([("a", "open", 0.2), ("b", "open", 0.9)]) == "weak_match"


def test_ranks_by_best_match_not_by_how_many_chunks_a_topic_owns():
    # `legal` floods the pool with four mediocre chunks; `water` has one strong chunk.
    # Summing would crown legal -- exactly the doc-count bias this must avoid.
    candidates = [
        ("agua", "open", 0.81),
        ("legal", "open", 0.44), ("legal", "open", 0.42),
        ("legal", "open", 0.41), ("legal", "open", 0.40),
    ]
    chips = rank_topics(candidates, lang="es")
    assert [c.key for c in chips] == ["water", "legal"]
    assert chips[0].score == 0.81


def test_never_suggests_the_planted_distractor():
    # The distractor exists to prove retrieval isn't fooled by an off-domain book.
    # Both its tier and its topic must independently keep it out of the list.
    chips = rank_topics([("chemistry", "distractor", 0.99)], lang="en")
    assert chips == []
    chips = rank_topics([("suelos", "distractor", 0.99)], lang="en")
    assert chips == []                         # blocked by tier even with a real topic
    chips = rank_topics([("chemistry", "open", 0.99)], lang="en")
    assert chips == []                         # blocked by topic even on a servable tier


def test_hides_topics_this_deployment_cannot_serve():
    candidates = [("mushrooms", "local", 0.90), ("agua", "open", 0.50)]
    assert [c.key for c in rank_topics(candidates, servable_tiers=DEPLOY)] == ["water"]
    # ...but with no restriction (local-mode run) the copyrighted tier is fair game.
    assert {c.key for c in rank_topics(candidates)} == {"mushrooms", "water"}


def test_merges_the_bilingual_duplicate_labels():
    # `horticulture`/`horticultura` are the same subject under two arbitrary labels;
    # a reader must never be shown both.
    chips = rank_topics([("horticultura", "open", 0.7), ("horticulture", "open", 0.6)], lang="en")
    assert [c.key for c in chips] == ["horticulture"]
    assert chips[0].score == 0.7               # keeps the better of the two


def test_localises_labels():
    assert rank_topics([("legal", "open", 0.5)], lang="es")[0].label == "Legislación agrícola"
    assert rank_topics([("legal", "open", 0.5)], lang="en")[0].label == "Agricultural law"


def test_unmapped_topic_fails_closed_rather_than_leaking_a_slug():
    assert rank_topics([("brand-new-topic", "open", 0.9)], lang="en") == []


def test_fallback_backfills_but_is_marked_as_not_from_retrieval():
    chips = rank_topics(
        [("agua", "open", 0.8)], lang="en", limit=3,
        fallback_counts=[("agua", 9), ("legal", 7), ("permaculture", 6)],
    )
    assert [c.key for c in chips] == ["water", "legal", "permaculture"]
    assert chips[0].from_retrieval and chips[0].score == 0.8
    # The backfilled ones must be distinguishable: the UI says "related to your question"
    # for one and "what this corpus covers" for the other, and must not mix them up.
    assert not chips[1].from_retrieval and chips[1].score is None
    # The fallback must not duplicate a topic retrieval already supplied.
    assert [c.key for c in chips].count("water") == 1


def test_fallback_sums_merged_topics_before_ranking_them():
    # The DB counts by RAW topic, so a merged subject arrives split across rows.
    # soil_fertility is really 5+2+1=8 and must outrank legal's 7 -- ordering the raw
    # rows as given would rank it on 5 and bury the corpus's largest subject.
    chips = rank_topics(
        [], lang="en", limit=3,
        fallback_counts=[("legal", 7), ("permaculture", 6), ("suelos", 5),
                         ("nutrition", 2), ("compost", 1)],
    )
    assert [c.key for c in chips] == ["soil_fertility", "legal", "permaculture"]
    assert all(not c.from_retrieval for c in chips)


def test_nearby_documents_dedupes_and_keeps_the_best_score():
    docs = nearby_documents([
        ("INIA_04.pdf", "open", 0.41), ("INIA_04.pdf", "open", 0.77),
        ("FAO.pdf", "open", 0.55), ("chem.pdf", "distractor", 0.99),
    ])
    assert [d["source"] for d in docs] == ["INIA_04.pdf", "FAO.pdf"]
    assert docs[0]["score"] == 0.77
