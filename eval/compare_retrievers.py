"""Retriever comparison on the golden set: dense vs lexical vs hybrid vs rerank.

Same dual metric as the Phase-2 baseline (answer-chunk + document), so numbers are
directly comparable to results/baseline.md. Prints what each stage fixed.

    uv run python eval/compare_retrievers.py --k 5
    uv run python eval/compare_retrievers.py --k 5 --open-only   # deploy corpus only (drops copyrighted 'local')
    uv run python eval/compare_retrievers.py --k 5 --latency     # + per-stage p50/p95 retrieval latency
"""
import argparse
import json
import re
import statistics
import sys
import time

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.dense import dense_search
from agroteca.retrieve.hybrid import hybrid_search
from agroteca.retrieve.lexical import lexical_search
from agroteca.retrieve.rerank import rerank_search

GOLDEN = settings.root / "eval" / "golden_set.jsonl"

# What a public deployment can legally serve: everything except the copyrighted 'local' tier.
DEPLOY_TIERS = ["open", "synthetic"]


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


def latency_pass(methods, questions):
    """Per-method retrieval latency over the golden set (p50/p95/mean/max, ms). Warms up first."""
    print(f"\n=== retrieval latency over {len(questions)} questions (ms) ===")
    print(f"{'method':8}  {'p50':>7}  {'p95':>7}  {'mean':>7}  {'max':>7}")
    for name, fn in methods.items():
        fn(questions[0]["question"], 5)  # warmup: exclude one-time lazy model load from timing
        ts = []
        for q in questions:
            s = time.perf_counter()
            fn(q["question"], 5)
            ts.append((time.perf_counter() - s) * 1000)
        ts.sort()
        p50 = statistics.median(ts)
        p95 = ts[max(0, round(0.95 * len(ts)) - 1)]
        print(f"{name:8}  {p50:>7.0f}  {p95:>7.0f}  {sum(ts)/len(ts):>7.0f}  {ts[-1]:>7.0f}")


def main(k, open_only, do_latency):
    conn = store.connect()
    qs = load_answerable()
    n = len(qs)
    tiers = DEPLOY_TIERS if open_only else None
    methods = {
        "dense":   lambda query, kk: dense_search(conn, query, kk, tiers),
        "lexical": lambda query, kk: lexical_search(conn, query, kk, tiers),
        "hybrid":  lambda query, kk: hybrid_search(conn, query, kk, n=60, tiers=tiers),
        "rerank":  lambda query, kk: rerank_search(conn, query, kk, tiers=tiers),
    }

    scope = (f"OPEN-ONLY deploy corpus ({'+'.join(DEPLOY_TIERS)}, no 'local')"
             if open_only else "FULL index (incl. copyrighted 'local')")
    print(f"\n=== retrieval@{k} over {n} answerable questions — {scope} ===")
    print(f"{'method':8}  {'answer-chunk':16}  {'document':12}")
    results = {}
    for name, fn in methods.items():
        a, d, miss = score(fn, qs, k)
        results[name] = (a, d, miss)
        print(f"{name:8}  {f'{a}/{n} = {a/n:.2f}':16}  {f'{d}/{n} = {d/n:.2f}':12}")

    dmiss = set(results["dense"][2])
    hmiss = set(results["hybrid"][2])
    rmiss = set(results["rerank"][2])
    print(f"\nhybrid FIXED over dense:    {sorted(dmiss - hmiss) or '-'}")
    print(f"rerank FIXED over hybrid:   {sorted(hmiss - rmiss) or '-'}")
    print(f"rerank regressed vs hybrid: {sorted(rmiss - hmiss) or '-'}")
    print(f"still missed by rerank:     {sorted(rmiss) or '-'}")

    if do_latency:
        latency_pass(methods, qs)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--open-only", action="store_true",
                    help="retrieve only from deployable tiers (open+synthetic), excluding copyrighted 'local'")
    ap.add_argument("--latency", action="store_true",
                    help="also report per-stage retrieval latency (p50/p95)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(args.k, args.open_only, args.latency)
