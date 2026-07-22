"""Phase 3 payoff: dense vs lexical vs hybrid on the golden set.

Same dual metric as the Phase-2 baseline (answer-chunk + document), so numbers are
directly comparable to results/baseline.md. Also prints which questions hybrid fixed.

    uv run python eval/compare_retrievers.py --k 5
"""
import argparse
import json
import re
import sys

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.dense import dense_search
from agroteca.retrieve.hybrid import hybrid_search
from agroteca.retrieve.lexical import lexical_search

GOLDEN = settings.root / "eval" / "golden_set.jsonl"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_answerable() -> list[dict]:
    out = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            if q.get("answerable", True) and q.get("must_contain"):
                out.append(q)
    return out


def score(rows_fn, questions, k):
    ans = doc = 0
    misses = []
    for q in questions:
        rows = rows_fn(q["question"], k)
        mc = _norm(q["must_contain"])
        a = any(sf == q["source_file"] and mc in _norm(t) for _cid, sf, t in rows)
        d = any(sf == q["source_file"] for _cid, sf, t in rows)
        ans += a
        doc += d
        if not a:
            misses.append(q["id"])
    return ans, doc, misses


def main(k: int):
    conn = store.connect()
    qs = load_answerable()
    n = len(qs)
    methods = {
        "dense":   lambda query, kk: dense_search(conn, query, kk),
        "lexical": lambda query, kk: lexical_search(conn, query, kk),
        "hybrid":  lambda query, kk: hybrid_search(conn, query, kk, n=60),
    }
    print(f"\n=== retrieval@{k} over {n} answerable questions ===")
    print(f"{'method':8}  {'answer-chunk':16}  {'document':12}")
    results = {}
    for name, fn in methods.items():
        a, d, miss = score(fn, qs, k)
        results[name] = (a, d, miss)
        print(f"{name:8}  {f'{a}/{n} = {a/n:.2f}':16}  {f'{d}/{n} = {d/n:.2f}':12}")

    dmiss, hmiss = set(results["dense"][2]), set(results["hybrid"][2])
    print(f"\nhybrid FIXED (dense missed → hybrid hit): {sorted(dmiss - hmiss) or '—'}")
    print(f"hybrid regressed (dense hit → hybrid missed): {sorted(hmiss - dmiss) or '—'}")
    print(f"still missed by hybrid: {sorted(hmiss) or '—'}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(args.k)
