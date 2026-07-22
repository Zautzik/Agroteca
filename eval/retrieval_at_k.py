"""Phase-2 ship criterion: retrieval@k on the golden set.

For each answerable golden question we embed the question, pull the top-k chunks by
cosine similarity, and count it a HIT if a retrieved chunk contains the question's
verbatim `must_contain` string. This is deterministic, free, and LLM-free — the seed
of the full eval runner, and exactly what lets us compare chunk sizes / retrievers.

Matching is whitespace-insensitive because chunks are normalized text while
`must_contain` was captured from raw extraction.

Usage:
    uv run python eval/retrieval_at_k.py            # k=5
    uv run python eval/retrieval_at_k.py --k 10 --tiers open synthetic
"""
import argparse
import json
import re
from pathlib import Path

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.ingest.embed import embed_query

GOLDEN = settings.root / "eval" / "golden_set.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_answerable() -> list[dict]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            if q.get("answerable", True) and q.get("must_contain"):
                rows.append(q)
    return rows


def retrieval_at_k(k: int, tiers: list[str] | None) -> None:
    conn = store.connect()
    questions = load_answerable()
    hits_answer = hits_doc = 0
    misses = []

    tier_filter = ""
    params_tail: tuple = ()
    if tiers:
        tier_filter = "WHERE c.tier = ANY(%s)"
        params_tail = (tiers,)

    for q in questions:
        qv = embed_query(q["question"])
        sql = f"""
            SELECT d.source_file, c.text
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            {tier_filter}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        rows = conn.execute(sql, (*params_tail, qv, k)).fetchall()
        mc = _norm(q["must_contain"])
        answer_hit = any(sf == q["source_file"] and mc in _norm(txt) for sf, txt in rows)
        doc_hit = any(sf == q["source_file"] for sf, _ in rows)
        hits_answer += answer_hit
        hits_doc += doc_hit
        if not answer_hit:
            misses.append(q["id"])

    conn.close()
    n = len(questions)
    print(f"\n=== retrieval@{k} over {n} answerable questions "
          f"(tiers={tiers or 'ALL ingested'}) ===")
    print(f"  answer-chunk (right file + must_contain in top-{k}): {hits_answer}/{n} = {hits_answer/n:.2f}")
    print(f"  document     (right file present in top-{k})       : {hits_doc}/{n} = {hits_doc/n:.2f}")
    if misses:
        print(f"  answer-chunk misses: {', '.join(misses)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tiers", nargs="*", default=None)
    args = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    retrieval_at_k(args.k, args.tiers)
