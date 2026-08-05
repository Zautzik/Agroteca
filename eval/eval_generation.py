"""Phase 5 — generation eval: abstention, citation, latency on the golden set.

Retrieval@k asked a mechanical question: "is the right chunk present?" (yes/no).
Generation is different. We measure three things that are cleanly mechanical:

  1. ABSTENTION  — on no-answer questions, does the model emit the EXACT abstain
     phrase instead of inventing an answer? (The phrase is canonical on purpose,
     so a machine can detect it exactly.)
  2. CITATION    — on answerable questions, does the answer carry a (....pdf)
     citation at all? We check the *format*, not the exact filename: the model
     paraphrases filenames (it wrote "agua de lluvia" for a file named "agua
     lluvia"), so exact-filename matching fails the same way exact-answer
     matching does.
  3. LATENCY     — wall-clock per call, reported as p50 / p95. Models load once,
     so this is honest per-question cost.

Faithfulness ("is every claim supported by the context?") is semantic and needs
an LLM judge / RAGAS -- a later layer. This script measures what's mechanical and
prints every answer so the rest can be read by eye.

    uv run python eval/eval_generation.py            # all questions
    uv run python eval/eval_generation.py --noanswer # just the abstention traps
    uv run python eval/eval_generation.py --limit 3  # first 3 (quick smoke)
"""
import argparse
import json
import re
import sys
import time

from agroteca.config import settings
from agroteca.generate import answer
from agroteca.ingest import store

GOLDEN = settings.root / "eval" / "golden_set.jsonl"
ABSTAIN = "no encuentro la respuesta en el contexto disponible"
CITES_PDF = re.compile(r"\([^)]*\.pdf\)", re.IGNORECASE)


def load_golden() -> list[dict]:
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main(only_noanswer: bool, limit: int | None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conn = store.connect()

    questions = load_golden()
    if only_noanswer:
        questions = [q for q in questions if not q.get("answerable", True)]
    if limit:
        questions = questions[:limit]

    abstain_ok = abstain_total = 0
    cite_ok = cite_total = 0
    latencies = []

    for q in questions:
        t0 = time.perf_counter()
        out = answer(conn, q["question"])
        dt = time.perf_counter() - t0
        latencies.append(dt)

        abstained = ABSTAIN in out.lower()

        if q.get("answerable", True):
            cite_total += 1
            cited = bool(CITES_PDF.search(out)) and not abstained
            cite_ok += cited
            tag = "WRONGLY-ABSTAINED" if abstained else ("cite-ok" if cited else "NO-CITE")
        else:
            abstain_total += 1
            abstain_ok += abstained
            tag = "abstain-ok" if abstained else "HALLUCINATED"

        print(f"[{q['id']}] {dt:5.1f}s  {tag}")
        print(f"    A: {out[:180]}")

    conn.close()

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95 = latencies[min(int(n * 0.95), n - 1)]
    print("\n=== Generation eval ===")
    if abstain_total:
        print(f"Abstention on no-answer:  {abstain_ok}/{abstain_total}")
    if cite_total:
        print(f"Citation on answerable:   {cite_ok}/{cite_total}")
    print(f"Latency  p50={p50:.1f}s  p95={p95:.1f}s  (n={n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noanswer", action="store_true", help="only the no-answer abstention traps")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    args = ap.parse_args()
    main(args.noanswer, args.limit)
