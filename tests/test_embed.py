"""The E5-vs-not prefix branch is silent-failure-prone: a missing 'query:'/'passage:' prefix on
an E5 model quietly degrades retrieval without ever erroring. These pin the pure decision and its
application, so the branch can't drift when the configured model changes.
"""
from agroteca.ingest.embed import _as_passages, _as_query, _is_e5


def test_detects_e5_models_only():
    assert _is_e5("intfloat/multilingual-e5-large")
    assert _is_e5("intfloat/multilingual-e5-small")
    assert not _is_e5("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert not _is_e5("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def test_prefixes_applied_for_e5_and_withheld_otherwise():
    # E5: prefixed. MiniLM/mpnet: raw text passes through untouched.
    assert _as_passages(["soil health"], is_e5=True) == ["passage: soil health"]
    assert _as_passages(["soil health"], is_e5=False) == ["soil health"]
    assert _as_query("water harvesting", is_e5=True) == "query: water harvesting"
    assert _as_query("water harvesting", is_e5=False) == "water harvesting"


def test_passages_helper_preserves_order_and_count():
    out = _as_passages(["a", "b", "c"], is_e5=True)
    assert out == ["passage: a", "passage: b", "passage: c"]
