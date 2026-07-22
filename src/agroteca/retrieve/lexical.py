"""Lexical (keyword) retrieval: Postgres full-text search over the 'simple' tsv.

`ts_rank_cd` is a BM25-family cover-density ranking. The important subtlety:
`websearch_to_tsquery` ANDs its terms, so a whole question becomes a conjunction
that matches nothing. For *retrieval* we want OR-of-content-terms — so we tokenize
the query, drop stopwords, and join with OR, then rank by how well chunks match.
"""
import re

# Question words + high-frequency ES/EN stopwords — noise in an OR keyword query.
_STOP = {
    "the","a","an","of","and","to","in","for","that","with","as","are","is","on","be","by",
    "this","it","or","from","at","what","which","how","when","where","why","who","does","do",
    "según","segun","de","la","el","que","en","los","las","del","por","con","para","una","un",
    "se","es","al","como","y","no","más","mas","cuál","cual","cómo","como","cuándo","cuando",
    "dónde","donde","qué","quién","quien","son","su","sus","este","esta","cuánto","cuanto",
}


def _to_or_query(text: str) -> str:
    """Content tokens of the query, OR-joined for websearch_to_tsquery."""
    toks = re.findall(r"[0-9a-záéíóúñü][0-9a-záéíóúñü\-]*", text.lower())
    keep = [t for t in dict.fromkeys(toks) if len(t) > 2 and t not in _STOP]
    return " OR ".join(keep)


def lexical_search(conn, query: str, n: int, tiers: list[str] | None = None):
    """Return the top-n keyword matches as [(chunk_id, source_file, text), ...]."""
    or_query = _to_or_query(query)
    if not or_query:
        return []
    tier_filter, params = "", ()
    if tiers:
        tier_filter, params = "AND c.tier = ANY(%s)", (tiers,)
    sql = f"""
        SELECT c.chunk_id, d.source_file, c.text
        FROM chunks c JOIN documents d ON c.doc_id = d.doc_id,
             websearch_to_tsquery('simple', %s) AS q
        WHERE c.tsv @@ q {tier_filter}
        ORDER BY ts_rank_cd(c.tsv, q) DESC
        LIMIT %s
    """
    return conn.execute(sql, (or_query, *params, n)).fetchall()
