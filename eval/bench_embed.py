"""Embedder throughput benchmark: MiniLM-384 (current baseline) vs multilingual-e5-large (upgrade).

The number behind config.py's embedder choice. Measures chunks/sec, ms/chunk, projected
full re-index time, and single-query embed latency for each model on THIS CPU, over real
chunk texts pulled from the live index (falls back to synthetic passages if the DB is down).

    uv run python eval/bench_embed.py

Measured 2026-08 on CPU: MiniLM-384 ~20.6 chunks/sec vs e5-large ~0.4 -> ~50x slower, an
~8-min full re-index vs ~7 hours. Re-run on your hardware; the ratio is the point, not the
absolutes (and note: this replaced an earlier "~10x" estimate that was off by 5x).
"""
import statistics
import time

from fastembed import TextEmbedding

try:
    from agroteca.config import settings
    from agroteca.ingest.store import connect
    _HAVE_DB = True
except Exception:
    _HAVE_DB = False

CORPUS_CHUNKS = 10_330  # live index size, for the full re-index projection
BATCH = 64              # matches settings.batch_size

MODELS = [
    ("MiniLM-384 (current)",     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False),
    ("e5-large-1024 (upgrade)",  "intfloat/multilingual-e5-large",                              True),
]


def get_texts(n: int = 160) -> list[str]:
    """Real chunk texts from the live index if reachable; else representative synthetic passages."""
    if _HAVE_DB:
        try:
            conn = connect()
            with conn.cursor() as cur:
                cur.execute("SELECT text FROM chunks ORDER BY random() LIMIT %s", (n,))
                rows = [r[0] for r in cur.fetchall()]
            conn.close()
            if rows:
                avg = sum(len(t) for t in rows) // len(rows)
                print(f"[data] {len(rows)} REAL chunk texts from the index (avg {avg} chars)")
                return rows
        except Exception as e:
            print(f"[data] DB unavailable ({e.__class__.__name__}); using synthetic passages")
    base = ("Soil organic matter improves water-holding capacity and cation exchange. "
            "For a cover crop of cereal rye seeded at 90 kg/ha in late autumn, terminate "
            "two to three weeks before cash-crop planting to manage the carbon-to-nitrogen "
            "ratio. Recommended stocking density for tilapia in a small-scale aquaponic "
            "system is 20 kg of fish per cubic metre of water. ") * 6
    texts = [base[:2000] + f" (sample {i})" for i in range(n)]
    print(f"[data] {len(texts)} synthetic passages (~{len(texts[0])} chars each)")
    return texts


def bench(label, model_name, is_e5, texts, runs=3):
    print(f"\n=== {label} ===")
    print(f"    loading {model_name} (downloads ONNX weights on first use)...")
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    passages = [f"passage: {t}" for t in texts] if is_e5 else texts
    vecs = list(model.embed(passages[:8], batch_size=BATCH))  # warmup
    print(f"    load+warmup: {time.perf_counter() - t0:.1f}s")

    times = []
    for _ in range(runs):
        s = time.perf_counter()
        vecs = list(model.embed(passages, batch_size=BATCH))
        times.append(time.perf_counter() - s)
    med = statistics.median(times)
    n = len(texts)
    cps = n / med
    dim = len(vecs[0])

    q = ("query: " if is_e5 else "") + "what is the recommended tilapia stocking density?"
    qts = []
    for _ in range(5):
        s = time.perf_counter()
        next(iter(model.embed([q])))
        qts.append((time.perf_counter() - s) * 1000)
    q_ms = statistics.median(qts)

    reindex_min = CORPUS_CHUNKS / cps / 60
    print(f"    dim={dim} | {n} chunks / {med:.2f}s -> {cps:.1f} chunks/sec ({med/n*1000:.1f} ms/chunk)")
    print(f"    projected full re-index ({CORPUS_CHUNKS:,} chunks): {reindex_min:.1f} min")
    print(f"    single-query embed latency: {q_ms:.0f} ms")
    return {"label": label, "dim": dim, "cps": cps, "reindex_min": reindex_min, "q_ms": q_ms}


def main():
    texts = get_texts()
    results = []
    for label, name, is_e5 in MODELS:
        try:
            results.append(bench(label, name, is_e5, texts))
        except Exception as e:
            print(f"    !! {label} failed: {e.__class__.__name__}: {e}")

    print("\n===== SUMMARY =====")
    print(f"{'model':26} {'dim':>5} {'chunks/sec':>11} {'re-index':>10} {'query ms':>9}")
    for r in results:
        print(f"{r['label']:26} {r['dim']:>5} {r['cps']:>11.1f} {r['reindex_min']:>8.1f}m {r['q_ms']:>8.0f}")
    if len(results) == 2:
        a, b = results  # MiniLM, e5
        print(f"\ne5-large is {a['cps']/b['cps']:.1f}x SLOWER to embed on this CPU "
              f"({a['cps']:.0f} vs {b['cps']:.0f} chunks/sec).")
        print(f"Full re-index: {b['reindex_min']:.0f} min (e5) vs {a['reindex_min']:.0f} min (MiniLM).")


if __name__ == "__main__":
    main()
