"""Query-translation lever: do the cross-lingual misses recover if we retrieve AND rerank
in a *bilingual* query space?

The residual misses (q08/q20/q22) are Spanish questions whose answer phrase is English; they
survived MiniLM-512, MiniLM-128, and the e5-large upgrade. The experiment (docs/masterclass.md
4.13) diagnosed the bottleneck as cross-lingual alignment, not embedder quality — which puts the
lever on the QUERY side, not the model.

Method (per question):
  1. translate it to the other language (Ollama, cached in eval/query_translations.json);
  2. candidate pool = hybrid(original) UNION hybrid(translation), deduped by chunk_id;
  3. rerank that pool by the MAX cross-encoder score over {original, translation} — so an English
     answer chunk earns its score from the English query member, a Spanish chunk from the Spanish
     one. Symmetric, and lexical token-overlap in the translated language directly catches phrases
     like "clay seed balls" a Spanish query never shares.
Then compare the final answer@5 / miss set against the shipped rerank baseline, on the SAME
(shipped MiniLM) index.

Run: uv run python eval/query_translation.py
"""
import json
from pathlib import Path

import httpx
from ollama import Client

from _scoring import answer_at_k, answer_rank, document_hit, load_answerable, reciprocal_rank
from agroteca.config import settings
from agroteca.ingest import store
from agroteca.retrieve.hybrid import hybrid_search
from agroteca.retrieve.rerank import _reranker

GOLDEN = settings.root / "eval" / "golden_set.jsonl"
CACHE = settings.root / "eval" / "query_translations.json"
K = 5
CANDIDATES = settings.rerank_candidates
MISSES = {"q07", "q08", "q15", "q20", "q22"}  # the shipped cascade's residual misses

_client = Client(host=settings.gen_base_url, timeout=httpx.Timeout(180.0, connect=10.0))


def translate(question: str, lang: str) -> str:
    """ES->EN, EN->ES. Kept terse; the model just needs to echo the question in the other tongue."""
    target = "English" if lang == "es" else "Spanish"
    prompt = (f"Translate this agricultural question to {target}. "
              f"Output ONLY the translation, no quotes, no preamble.\n\n{question}")
    resp = _client.chat(model=settings.gen_model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_predict": 128})
    return resp["message"]["content"].strip().strip('"')


def translations_for(questions: list[dict]) -> dict[str, str]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    changed = False
    for q in questions:
        if q["id"] not in cache:
            cache[q["id"]] = translate(q["question"], q.get("lang", "es"))
            print(f"  translated {q['id']} ({q.get('lang')}): {cache[q['id']][:70]}")
            changed = True
    if changed:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def _sf_text(rows):
    return [(sf, text) for _cid, sf, text in rows]


def rerank_pool(pool, *queries, k=K):
    """Rerank a candidate pool by the MAX cross-encoder score over the given queries."""
    if not pool:
        return []
    texts = [r[2] for r in pool]
    best = [float("-inf")] * len(texts)
    for q in queries:
        for i, s in enumerate(_reranker().rerank(q, texts)):
            best[i] = max(best[i], s)
    ranked = sorted(zip(pool, best), key=lambda p: p[1], reverse=True)
    return [row for row, _ in ranked[:k]]


def main() -> None:
    qs = load_answerable(GOLDEN)
    print(f"translating {len(qs)} questions (cached in {CACHE.name})...")
    xlt = translations_for(qs)

    pool = store.make_pool()
    pool.open()

    def hyb(query):
        with pool.connection() as conn:   # fresh, self-healing conn per call (no conn held during rerank)
            return hybrid_search(conn, query, k=CANDIDATES)

    agg = {"base": dict(a5=0, a1=0, d5=0, rr=0.0, miss=[]),
           "xlt":  dict(a5=0, a1=0, d5=0, rr=0.0, miss=[])}
    moves = []
    for q in qs:
        sf, mc = q["source_file"], q["must_contain"]
        orig, trans = q["question"], xlt[q["id"]]

        po, pt = hyb(orig), hyb(trans)                             # each hybrid pool computed once
        base = rerank_pool(po, orig)                               # shipped: pool from orig, rerank on orig
        seen, union = {}, []
        for row in po + pt:                                        # bilingual union pool
            if row[0] not in seen:
                seen[row[0]] = 1; union.append(row)
        xr = rerank_pool(union, orig, trans)                       # rerank by max over {orig, trans}

        for tag, rows in (("base", base), ("xlt", xr)):
            rank = answer_rank(_sf_text(rows), sf, mc)
            a = agg[tag]
            a["a5"] += answer_at_k(rank, 5); a["a1"] += answer_at_k(rank, 1)
            a["d5"] += 1 if document_hit(_sf_text(rows), sf) else 0
            a["rr"] += reciprocal_rank(rank)
            if not answer_at_k(rank, 5):
                a["miss"].append(q["id"])

        rb = answer_rank(_sf_text(base), sf, mc)
        rx = answer_rank(_sf_text(xr), sf, mc)
        if rb != rx:
            moves.append((q["id"], rb, rx))
    pool.close()

    n = len(qs)
    print(f"\n=== query-translation lever over {n} answerable questions (shipped MiniLM index) ===")
    print(f"{'config':22}{'answer@5':>10}{'answer@1':>10}{'doc@5':>8}{'mrr@5':>8}")
    for tag, label in (("base", "shipped rerank"), ("xlt", "+ query translation")):
        a = agg[tag]
        print(f"{label:22}{a['a5']/n:>10.2f}{a['a1']/n:>10.2f}{a['d5']/n:>8.2f}{a['rr']/n:>8.2f}")

    print("\nrank changes (id: baseline_rank -> translated_rank; None = not in top-5):")
    for qid, rb, rx in moves:
        star = " *" if qid in MISSES else "  "
        print(f"  {qid}{star}  {rb} -> {rx}")
    print(f"\nshipped still-missed: {sorted(agg['base']['miss'])}")
    print(f"+xlt   still-missed: {sorted(agg['xlt']['miss'])}")
    recovered = sorted(set(agg['base']['miss']) - set(agg['xlt']['miss']))
    regressed = sorted(set(agg['xlt']['miss']) - set(agg['base']['miss']))
    print(f"RECOVERED by translation: {recovered or '-'}    |    regressed: {regressed or '-'}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
