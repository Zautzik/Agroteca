"""Regression tests for the two pure retrieval primitives with silent failure modes.

RRF fuses ranked lists by position; a wrong k or a dropped tie would reorder results without
raising. The lexical query builder drops stopwords and OR-joins content tokens; a leaked
stopword or a mangled exact code (WL-323) would quietly change what full-text search matches.
Both are load-bearing and invisible by inspection, so they are pinned here.
"""
from agroteca.retrieve.fusion import reciprocal_rank_fusion
from agroteca.retrieve.lexical import _to_or_query


# --- Reciprocal Rank Fusion ---------------------------------------------------------------

def test_rrf_rewards_agreement_and_keeps_first_seen_rows():
    # 'b' is only mid-ranked in each list but appears in BOTH; RRF's whole point is that
    # cross-list agreement should beat a single strong placement.
    dense =   [("a", "f", "dense-a"), ("b", "f", "dense-b"), ("c", "f", "dense-c")]  # ranks 1,2,3
    lexical = [("b", "f", "lexical-b"), ("d", "f", "lexical-d")]                     # ranks 1,2

    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    # Scores (k=60): b = 1/62 + 1/61 (both lists) > a = 1/61 > d = 1/62 > c = 1/63.
    assert [row[0] for row in fused] == ["b", "a", "d", "c"]
    # Keyed by chunk_id, 'b' keeps the FIRST row it was seen in (dense), not lexical's.
    assert fused[0] == ("b", "f", "dense-b")


def test_rrf_top_truncates_after_scoring():
    dense =   [("a", "f", "x"), ("b", "f", "x"), ("c", "f", "x")]
    lexical = [("b", "f", "x"), ("d", "f", "x")]
    # top is applied AFTER fusion, so the agreement winner 'b' survives a top=2 cut.
    assert [row[0] for row in reciprocal_rank_fusion([dense, lexical], k=60, top=2)] == ["b", "a"]


# --- lexical OR-query builder -------------------------------------------------------------

def test_or_query_drops_stopwords_and_joins_content_tokens():
    # websearch_to_tsquery ANDs terms, so a whole question matches nothing; we OR the content
    # words instead. Question words and short/stopword tokens must not leak through.
    assert _to_or_query("What is the recommended stocking density?") == "recommended OR stocking OR density"


def test_or_query_preserves_exact_codes_lowercased_and_deduped():
    # The exact-term case the hybrid design exists for: a cultivar code must survive tokenization
    # intact (hyphen kept), lowercased, and de-duplicated across the query.
    assert _to_or_query("Which WL-323 alfalfa? The WL-323 variety.") == "wl-323 OR alfalfa OR variety"


def test_or_query_handles_spanish_stopwords_and_accents():
    # Bilingual corpus: Spanish question words and accented tokens must filter correctly.
    assert _to_or_query("¿Cuál es la densidad recomendada?") == "densidad OR recomendada"
