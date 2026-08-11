"""Would a bigger reranker candidate pool recover any of the recall-gap misses?

The reranker only reorders what hybrid fetched, so hybrid answer@N is the ceiling on
rerank answer@5 at candidates=N. Measuring that ceiling is cheap (no cross-encoder), and it
answers the question without the ~hour of CPU a full reranker sweep would cost:

  - answer not in hybrid top-60      -> true recall gap; no candidate count helps (embedder lever)
  - answer already in hybrid top-20  -> the reranker saw it and didn't float it; a precision limit
  - answer in hybrid rank 21..60     -> a bigger pool could let the reranker recover it (worth a run)

    uv run python eval/sweep_candidates.py
"""
import sys

from _scoring import answer_rank, load_answerable
from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.hybrid import hybrid_search

GOLDEN = settings.root / "eval" / "golden_set.jsonl"
CANDS = [20, 30, 40, 60]
RERANK_MISSES = {"q07", "q08", "q15", "q20", "q22"}  # what rerank@5 misses at candidates=20


def _hybrid_topn(pool, question, n):
    with pool.connection() as conn:
        return [(sf, text) for _cid, sf, text in hybrid_search(conn, question, n)]


def main():
    pool = store.make_pool()
    pool.open()
    qs = load_answerable(GOLDEN)
    by_id = {q["id"]: q for q in qs}
    n = len(qs)

    print(f"\n=== hybrid answer@N over {n} questions (ceiling on rerank answer@5 at candidates=N) ===")
    for N in CANDS:
        hits = sum(
            answer_rank(_hybrid_topn(pool, q["question"], N), q["source_file"], q["must_contain"]) is not None
            for q in qs
        )
        print(f"  candidates={N:<3}  hybrid answer@{N} = {hits}/{n} = {hits/n:.2f}")

    print("\n=== where each rerank@5 miss sits in hybrid's top-60 ===")
    for qid in sorted(RERANK_MISSES):
        q = by_id[qid]
        rank = answer_rank(_hybrid_topn(pool, q["question"], 60), q["source_file"], q["must_contain"])
        if rank is None:
            verdict = "not in top-60  -> true recall gap (embedder lever, not pool size)"
        elif rank <= 20:
            verdict = f"hybrid rank {rank}  -> already in the pool; a reranker precision limit, not pool size"
        else:
            verdict = f"hybrid rank {rank}  -> in 21..60; a bigger candidate pool could let rerank recover it"
        print(f"  {qid}: {verdict}")

    pool.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
