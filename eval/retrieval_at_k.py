"""Phase-2 ship criterion: retrieval@k on the golden set.

Embed each answerable question, pull the top-k chunks by cosine similarity, and score with
the shared metrics in eval/_scoring.py. Deterministic, free, and LLM-free — the seed of the
full comparison in compare_retrievers.py.

    uv run python eval/retrieval_at_k.py            # k=5
    uv run python eval/retrieval_at_k.py --k 10 --tiers open synthetic
"""
import argparse
import sys

from _scoring import load_answerable, score
from agroteca.config import settings
from agroteca.ingest import store
from agroteca.ingest.embed import embed_query

GOLDEN = settings.root / "eval" / "golden_set.jsonl"


def retrieval_at_k(k: int, tiers: list[str] | None) -> None:
    conn = store.connect()
    questions = load_answerable(GOLDEN)
    tier_filter, params_tail = ("", ())
    if tiers:
        tier_filter, params_tail = ("WHERE c.tier = ANY(%s)", (tiers,))

    def retrieve(question, kk):
        qv = embed_query(question)
        sql = f"""
            SELECT d.source_file, c.text
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            {tier_filter}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        return conn.execute(sql, (*params_tail, qv, kk)).fetchall()

    r = score(questions, retrieve, k)
    conn.close()

    print(f"\n=== retrieval@{k} over {r['n']} answerable questions "
          f"(tiers={tiers or 'ALL ingested'}) ===")
    print(f"  answer-chunk (right file + must_contain in top-{k}): {r[f'answer@{k}']:.2f}")
    print(f"  document     (right file present in top-{k})       : {r[f'document@{k}']:.2f}")
    print(f"  answer@1: {r['answer@1']:.2f}    mrr@{k}: {r[f'mrr@{k}']:.2f}")
    if r["misses"]:
        print(f"  answer-chunk misses: {', '.join(r['misses'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tiers", nargs="*", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    retrieval_at_k(args.k, args.tiers)
