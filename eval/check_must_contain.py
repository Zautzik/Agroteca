"""Does each golden `must_contain` phrase identify its answer chunk uniquely?

The answer-chunk metric counts a hit when a retrieved chunk is the right file AND contains
the `must_contain` phrase. If that phrase also appears in a DIFFERENT chunk of the same file,
a hit could be granted for a chunk that isn't the true answer -- inflating the score. This
verifies each phrase lands in exactly one chunk of its source file:

  0 chunks  -> broken label (the phrase isn't in the indexed text; would always miss)
  1 chunk   -> clean; no inflation possible
  2 chunks  -> usually benign: chunk overlap (64 tokens) duplicates a phrase across neighbours
  >2        -> review: the phrase recurs elsewhere in the file

    uv run python eval/check_must_contain.py
"""
import sys

from _scoring import load_answerable, norm
from agroteca.config import settings
from agroteca.ingest import store

GOLDEN = settings.root / "eval" / "golden_set.jsonl"


def main():
    conn = store.connect()
    qs = load_answerable(GOLDEN)
    flagged = []
    print(f"\n=== must_contain uniqueness over {len(qs)} answerable questions ===")
    for q in qs:
        needle = norm(q["must_contain"])
        rows = conn.execute(
            "SELECT c.chunk_index, c.text FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
            "WHERE d.source_file = %s",
            (q["source_file"],),
        ).fetchall()
        # match in Python with the metric's own norm(), so this check can't diverge from it
        idxs = sorted(i for (i, text) in rows if needle in norm(text))
        adjacent = len(idxs) == 2 and idxs[1] - idxs[0] == 1
        mark = {0: "MISSING", 1: "unique"}.get(len(idxs), "adjacent(overlap)" if adjacent else "REVIEW")
        if len(idxs) != 1 and not adjacent:
            flagged.append((q["id"], len(idxs), q["source_file"]))
        print(f"  {q['id']:>4}  {len(idxs):>2} chunk(s) {str(idxs):14}  {mark:18}  {q['source_file']}")

    clean = len(qs) - len(flagged)
    print(f"\n{clean}/{len(qs)} phrases identify their chunk cleanly (unique, or adjacent overlap).")
    if flagged:
        print("REVIEW -- phrase appears in 0 or non-adjacent multiple chunks of its file:")
        for qid, m, sf in flagged:
            print(f"  {qid}: {m} matches in {sf}")
    else:
        print("No inflation risk: no must_contain phrase recurs in an unrelated chunk of its file.")
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
