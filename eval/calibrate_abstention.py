"""Phase 0 — can rerank scores tell a near miss from a total miss?

The proposed no-answer UX branches on a claim: when the system abstains, the score
distribution of what retrieval *did* surface says which failure we're in.

  near miss  -> the answer IS in the corpus, retrieval failed to rank it
  total miss -> the corpus genuinely doesn't cover the question

If that claim holds, the two populations separate on some statistic of the top-k
scores and we can pick a threshold from data. If it doesn't, we ship ONE honest
"not found" state and drop the distinction -- a cheap negative result that costs an
hour here instead of a UI built on a guess.

This runs the SAME path production serves (retrieve_ranked_async: translation +
bilingual hybrid + cross-encoder rerank, k=5), so the numbers are the ones the live
classifier would actually see. No generation -- retrieval only.

    uv run python eval/calibrate_abstention.py
    uv run python eval/calibrate_abstention.py --out eval/abstention_scores.csv

NOTE ON n: the golden set has only 2 genuine no-answer questions. Two points do not
make a threshold. Treat any separation here as directional evidence to be confirmed
once eval/harvest_feedback.py has collected real user abstentions -- which is the
whole reason that script exists.
"""
import argparse
import asyncio
import csv
import json
import statistics
import sys
import time

from agroteca.config import settings
from agroteca.generate import retrieve_ranked_async
from agroteca.ingest import store

GOLDEN = settings.root / "eval" / "golden_set.jsonl"

# Ground-truth groups, from measurements already on record:
#   TOTAL_MISS  -- the golden set's own `answerable: false` traps (q17, q19 here);
#                  both correctly abstained in the full generation eval.
#   NEAR_MISS   -- answerable, but the answer chunk never reached the top-5:
#                  "still missed by rerank" from eval/compare_retrievers.py.
#                  q22 corroborates independently (WRONGLY-ABSTAINED in generation).
# Everything else is the healthy band retrieval handles correctly.
NEAR_MISS = {"q07", "q15", "q20", "q22"}


def load_golden() -> list[dict]:
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def group_of(q: dict) -> str:
    if not q.get("answerable", True):
        return "total_miss"
    return "near_miss" if q["id"] in NEAR_MISS else "healthy"


async def _run(out_path: str | None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conn = await store.connect_async()
    rows_out = []

    for q in load_golden():
        t0 = time.perf_counter()
        rows, _meta, retrieval_ms, translated = await retrieve_ranked_async(conn, q["question"], k=5)
        dt = time.perf_counter() - t0
        scores = [float(s) for (_cid, _sf, _txt, s) in rows]

        rec = {
            "id": q["id"],
            "group": group_of(q),
            "n_rows": len(scores),
            # The statistics a runtime classifier could actually use -- it sees only
            # these five rows, so calibrate on exactly that, nothing richer.
            "top": round(scores[0], 4) if scores else None,
            "mean_top3": round(statistics.fmean(scores[:3]), 4) if scores else None,
            "mean_all": round(statistics.fmean(scores), 4) if scores else None,
            "spread": round(scores[0] - scores[-1], 4) if len(scores) > 1 else None,
            "translated": bool(translated),
            "retrieval_ms": retrieval_ms,
        }
        rows_out.append(rec)
        print(f"[{rec['id']:>4}] {rec['group']:<10} top={rec['top']:>8} "
              f"mean3={rec['mean_top3']:>8} spread={rec['spread']:>7}  ({dt:5.1f}s)")

    await conn.close()

    print("\n=== score distribution by group ===")
    print(f"{'group':<11} {'n':>3}  {'min top':>8} {'median top':>11} {'max top':>8}")
    groups: dict[str, list[float]] = {}
    for r in rows_out:
        if r["top"] is not None:
            groups.setdefault(r["group"], []).append(r["top"])
    for g in ("healthy", "near_miss", "total_miss"):
        v = sorted(groups.get(g, []))
        if not v:
            continue
        print(f"{g:<11} {len(v):>3}  {v[0]:>8.3f} {statistics.median(v):>11.3f} {v[-1]:>8.3f}")

    # The question this script exists to answer, stated plainly.
    healthy, near, total = (sorted(groups.get(g, [])) for g in ("healthy", "near_miss", "total_miss"))
    print("\n=== separation ===")
    if near and total:
        gap = min(near) - max(total)
        print(f"lowest near_miss top-score : {min(near):.3f}")
        print(f"highest total_miss top-score: {max(total):.3f}")
        if gap > 0:
            print(f"SEPARATED by {gap:.3f} -- a threshold near {(min(near) + max(total)) / 2:.3f} "
                  f"splits them on this sample.")
        else:
            print(f"OVERLAP of {-gap:.3f} -- top-score alone cannot classify. "
                  f"Check mean_top3/spread in the CSV before trying a composite rule; "
                  f"if none separate, ship a single unified no-answer state.")
    else:
        print("not enough labelled rows in one of the groups to compare.")
    if healthy and near:
        print(f"(healthy median {statistics.median(healthy):.3f} vs "
              f"near_miss median {statistics.median(near):.3f})")

    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {out_path}")


def main(out_path: str | None) -> None:
    if sys.platform == "win32":
        # psycopg's async mode can't run on Windows' default ProactorEventLoop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/abstention_scores.csv",
                    help="CSV of per-question score statistics (use '' to skip)")
    args = ap.parse_args()
    main(args.out or None)
