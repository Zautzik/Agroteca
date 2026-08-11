"""Retriever comparison on the golden set: dense vs lexical vs hybrid vs rerank.

Metrics come from eval/_scoring.py (the single definition), so this reproduces the same
answer@k / document@k the Phase-2 baseline reported and adds the ranking-quality metrics
(answer@1, MRR) that expose the reordering a reranker performs but answer@k cannot see.

    uv run python eval/compare_retrievers.py --k 5
    uv run python eval/compare_retrievers.py --k 5 --open-only   # deploy corpus only (drops copyrighted 'local')
    uv run python eval/compare_retrievers.py --k 5 --latency     # + per-stage p50/p95 retrieval latency
"""
import argparse
import statistics
import sys
import time

from _scoring import load_answerable, score
from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.dense import dense_search
from agroteca.retrieve.hybrid import hybrid_search
from agroteca.retrieve.lexical import lexical_search
from agroteca.retrieve.rerank import rerank_search

GOLDEN = settings.root / "eval" / "golden_set.jsonl"

# What a public deployment can legally serve: everything except the copyrighted 'local' tier.
DEPLOY_TIERS = ["open", "synthetic"]


def _sf_text(rows):
    """Retrievers return (chunk_id, source_file, text); the metrics need only (source_file, text)."""
    return [(sf, text) for _cid, sf, text in rows]


def latency_pass(methods, questions):
    """Per-method retrieval latency over the golden set (p50/p95/mean/max, ms). Warms up first."""
    print(f"\n=== retrieval latency over {len(questions)} questions (ms) ===")
    print(f"{'method':8}  {'p50':>7}  {'p95':>7}  {'mean':>7}  {'max':>7}")
    for name, fn in methods.items():
        fn(questions[0]["question"], 5)  # warmup: exclude the one-time lazy model load from timing
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
    pool = store.make_pool()
    pool.open()
    qs = load_answerable(GOLDEN)
    n = len(qs)
    tiers = DEPLOY_TIERS if open_only else None

    def run(search, query, kk, **kw):
        # A fresh pooled connection per call. The pool validates it on checkout, so a
        # connection the Docker proxy dropped during the reranker's long CPU-only gaps is
        # replaced transparently instead of failing the pass mid-run.
        with pool.connection() as conn:
            return _sf_text(search(conn, query, kk, tiers=tiers, **kw))

    methods = {
        "dense":   lambda q, kk: run(dense_search, q, kk),
        "lexical": lambda q, kk: run(lexical_search, q, kk),
        "hybrid":  lambda q, kk: run(hybrid_search, q, kk, n=60),
        "rerank":  lambda q, kk: run(rerank_search, q, kk),
    }

    scope = (f"OPEN-ONLY deploy corpus ({'+'.join(DEPLOY_TIERS)}, no 'local')"
             if open_only else "FULL index (incl. copyrighted 'local')")
    a_k, a_1, d_k, m_k = f"answer@{k}", "answer@1", f"document@{k}", f"mrr@{k}"
    print(f"\n=== retrieval@{k} over {n} answerable questions — {scope} ===")
    print(f"{'method':8}  {a_k:>9}  {a_1:>9}  {d_k:>11}  {m_k:>7}")
    results = {}
    for name, fn in methods.items():
        r = score(qs, fn, k)
        results[name] = r
        print(f"{name:8}  {r[a_k]:>9.2f}  {r[a_1]:>9.2f}  {r[d_k]:>11.2f}  {r[m_k]:>7.2f}")

    dmiss = set(results["dense"]["misses"])
    hmiss = set(results["hybrid"]["misses"])
    rmiss = set(results["rerank"]["misses"])
    print(f"\nhybrid FIXED over dense:    {sorted(dmiss - hmiss) or '-'}")
    print(f"rerank FIXED over hybrid:   {sorted(hmiss - rmiss) or '-'}")
    print(f"rerank regressed vs hybrid: {sorted(rmiss - hmiss) or '-'}")
    print(f"still missed by rerank:     {sorted(rmiss) or '-'}")

    if do_latency:
        latency_pass(methods, qs)
    pool.close()


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
