"""Mechanistic pilot: would a 512-context embedder recover the dense-blind golden answers?

The shipped embedder (MiniLM-L12) truncates at ~128 tokens, so ~half the golden answers have
their answer text PAST the window the dense vector ever reads (measured; see config.py's KNOWN
CEILING note). The obvious fix is a longer-context embedder, but proving it works normally costs
a full re-index (~7 h on CPU) + a re-run of the whole eval. This asks the *causal* question far
more cheaply, before spending those hours.

For each dense-blind answer chunk, measure how much the answer-bearing TAIL moves the
query<->chunk cosine under each embedder:

    delta = cos(query, full_chunk) - cos(query, chunk_truncated_to_the_~128-token_window)

MiniLM cannot see the tail, so its delta is ~0 by construction -- that is the CONTROL. If
e5-large shows a clearly positive delta on the same chunks, the longer context is *using* the
answer text MiniLM never reads: direct evidence the upgrade addresses the recall gap rather than
a hope pinned on a bigger model. The end-to-end number (answer@5 / MRR) still needs the
re-index; this isolates the mechanism first.

Run: uv run python eval/embedder_pilot.py
"""
import json
from pathlib import Path

import numpy as np
import psycopg

from _scoring import norm  # same dir; this script lives in eval/
from agroteca.config import settings

# fastembed's only 512-context multilingual options are e5-large (1024-dim) and mpnet-base
# (768-dim); there is no drop-in 384-dim longer-context model, so the real upgrade also forces
# a VECTOR(n) migration. e5-large is the documented target -> pilot with it.
CANDIDATE = "intfloat/multilingual-e5-large"
BASELINE = settings.embed_model  # the shipped MiniLM
WINDOW = settings.chunk_tokens // 4 * settings.chars_per_token  # ~128 tokens -> ~512 chars
MISSES = {"q07", "q08", "q20", "q22"}  # the reranker's remaining misses (from the eval)

GOLDEN = Path(__file__).with_name("golden_set.jsonl")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def answerable(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        if q.get("answerable", True) and q.get("must_contain"):
            out.append(q)
    return out


def answer_chunk(conn, source_file: str, must_contain: str) -> tuple[str, int] | None:
    """The indexed chunk carrying the answer, and where in it the phrase falls.

    When several chunks qualify (64-token overlap can duplicate a phrase near a boundary), pick
    the one where the phrase sits DEEPEST -- the hardest case for a truncating embedder, and the
    chunk a longer-context model would newly be able to read.
    """
    needle = norm(must_contain)
    rows = conn.execute(
        "SELECT c.text FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
        "WHERE d.source_file = %s",
        (source_file,),
    ).fetchall()
    best, best_off = None, -1
    for (text,) in rows:
        off = norm(text).find(needle)
        if off > best_off:
            best, best_off = text, off
    return (best, best_off) if best is not None else None


def main() -> None:
    from fastembed import TextEmbedding

    gold = answerable(GOLDEN)
    print(f"window ~= {WINDOW} chars (~128 tokens); {len(gold)} answerable golden questions\n")

    # Resolve each question to its deepest answer chunk + the truncated head MiniLM effectively
    # sees. Skip the two no-answer questions (no chunk to test) automatically via `answerable`.
    recs = []
    with psycopg.connect(settings.db_url, connect_timeout=10) as conn:
        for q in gold:
            hit = answer_chunk(conn, q["source_file"], q["must_contain"])
            if hit is None:
                print(f"  {q['id']}: answer phrase not found in index -- skipped")
                continue
            chunk, off = hit
            recs.append({
                "id": q["id"], "question": q["question"], "chunk": chunk,
                "head": chunk[:WINDOW], "off": off, "blind": off > WINDOW,
            })

    # Batch-embed once per model (query + full passage + truncated passage). e5 needs task
    # prefixes; MiniLM must not have them.
    e5 = TextEmbedding(model_name=CANDIDATE, threads=settings.ort_threads)
    mini = TextEmbedding(model_name=BASELINE, threads=settings.ort_threads)

    def embed(model, texts):
        return [np.asarray(v) for v in model.embed(texts)]

    e5_q = embed(e5, [f"query: {r['question']}" for r in recs])
    e5_full = embed(e5, [f"passage: {r['chunk']}" for r in recs])
    e5_head = embed(e5, [f"passage: {r['head']}" for r in recs])
    mn_q = embed(mini, [r["question"] for r in recs])
    mn_full = embed(mini, [r["chunk"] for r in recs])
    mn_head = embed(mini, [r["head"] for r in recs])

    hdr = f"{'id':<5}{'off':>6}{'blind':>7}   {'MiniLM full/head/d':<26}   {'e5 full/head/d':<26}"
    print(hdr)
    print("-" * len(hdr))
    blind_recs, d_e5, d_mini = [], [], []
    for i, r in enumerate(recs):
        mf, mh = cos(mn_q[i], mn_full[i]), cos(mn_q[i], mn_head[i])
        ef, eh = cos(e5_q[i], e5_full[i]), cos(e5_q[i], e5_head[i])
        dm, de = mf - mh, ef - eh
        star = " *" if r["id"] in MISSES else "  "
        flag = "yes" if r["blind"] else "no"
        print(f"{r['id']}{star}{r['off']:>5}{flag:>7}   "
              f"{mf:+.3f}/{mh:+.3f}/{dm:+.3f}      {ef:+.3f}/{eh:+.3f}/{de:+.3f}")
        if r["blind"]:
            blind_recs.append(r["id"]); d_e5.append(de); d_mini.append(dm)

    print("\n(* = the reranker's remaining misses; d = full - head, the answer tail's contribution)")
    if blind_recs:
        print(f"\ndense-blind answers (phrase past the window): {len(blind_recs)} -> {blind_recs}")
        print(f"  mean tail contribution  MiniLM d = {np.mean(d_mini):+.3f}   (control: ~0, it can't read the tail)")
        print(f"  mean tail contribution  e5-large d = {np.mean(d_e5):+.3f}   (positive => it uses the answer text MiniLM misses)")
        recovered = [b for b, de, dm in zip(blind_recs, d_e5, d_mini) if de > dm + 0.02]
        print(f"  e5 tail-lift exceeds MiniLM's by >0.02 on {len(recovered)}/{len(blind_recs)}: {recovered}")


if __name__ == "__main__":
    main()
