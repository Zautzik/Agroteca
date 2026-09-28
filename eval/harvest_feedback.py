"""Turn real user feedback into golden-set candidates.

`/feedback` has been appending votes to feedback.jsonl since Phase 8, and until now
NOTHING read the file. That is the open end of the loop: signal goes in, nothing comes
out, and the golden set stays frozen at the 22 questions one person thought of up front.

This script closes it. It reads the log, groups failures by what kind they are, and
emits golden-set candidates in the same JSONL shape eval/golden_set.jsonl already uses,
so a reviewed candidate is a copy-paste into the eval set.

Two failure classes matter, and they need opposite fixes:

  DOWNVOTE + answered   -> the system answered and the answer was wrong or unhelpful.
                           Candidate for `answerable: true` -- a retrieval or grounding
                           bug to chase.
  ABSTAINED             -> the system said it could not answer. Either the corpus really
                           lacks it (`answerable: false`, a legitimate new abstention
                           trap -- and the golden set has only TWO of those today), or
                           the answer is there and retrieval missed it (`answerable:
                           true`, a recall bug). The `retrieved` + `top_score` fields
                           recorded with each vote are what let a human tell those apart
                           without re-running the query.

Abstentions are the scarce resource. eval/calibrate_abstention.py is currently trying to
separate near-miss from total-miss on a sample of two; every abstention harvested here is
a data point that calibration needs.

    uv run python eval/harvest_feedback.py
    uv run python eval/harvest_feedback.py --abstained-only --out eval/candidates.jsonl
"""
import argparse
import collections
import json
import sys

from agroteca.config import settings

FEEDBACK_LOG = settings.root / "feedback.jsonl"
GOLDEN = settings.root / "eval" / "golden_set.jsonl"


def load_feedback() -> list[dict]:
    """Read the log, tolerating both old and new record shapes.

    Records written before the abstention fields existed simply lack those keys; they are
    still valid votes and must not be dropped, so every new field is read with a default."""
    if not FEEDBACK_LOG.exists():
        return []
    out = []
    for i, line in enumerate(FEEDBACK_LOG.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn line (crash mid-write) should not lose the rest of the log.
            print(f"  ! skipped unparseable line {i}", file=sys.stderr)
    return out


def existing_questions() -> set[str]:
    """Questions already in the golden set — harvest should never re-propose them.

    Matching is exact (case/whitespace-normalised only). A user asking the same thing in
    different words WILL be proposed again: "What stocking density does the FAO guide
    recommend?" does not match the golden set's "What is the recommended stocking density
    (fish per cubic meter) for tilapia...". Deliberately not fuzzy-matched -- a similarity
    threshold is one more number that would need calibrating, and the reviewer sees the
    candidate either way. The risk to watch is the reviewer's, not the script's: appending
    a near-duplicate would double-count one question in every metric derived from the set.
    """
    if not GOLDEN.exists():
        return set()
    qs = set()
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qs.add(json.loads(line)["question"].strip().lower())
    return qs


def classify(rec: dict) -> str | None:
    """What kind of signal is this record? None = not a failure worth harvesting."""
    if rec.get("abstained"):
        return "abstained"
    if rec.get("vote") == "down":
        return "bad_answer"
    return None  # upvotes confirm working behaviour; nothing to fix


def main(abstained_only: bool, out_path: str | None, min_count: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_feedback()
    if not records:
        print(f"No feedback yet at {FEEDBACK_LOG}.")
        print("Nothing to harvest — this is expected until the deployed app sees real use.")
        return

    already = existing_questions()
    # Group by question text: the same question failing repeatedly is a stronger candidate
    # than a one-off, and dedupes a user who retried the same thing five times.
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        kind = classify(rec)
        if kind is None or (abstained_only and kind != "abstained"):
            continue
        q = (rec.get("question") or "").strip()
        if q and q.lower() not in already:
            grouped[q].append(rec)

    votes = collections.Counter(r.get("vote") for r in records)
    n_abstain = sum(1 for r in records if r.get("abstained"))
    print(f"=== feedback log: {len(records)} records "
          f"({votes.get('up', 0)} up, {votes.get('down', 0)} down, {n_abstain} abstentions) ===")
    if n_abstain == 0 and any("abstained" not in r for r in records):
        print("  note: some records predate abstention logging, so 0 abstentions may just")
        print("        mean the UI wasn't capturing that branch yet.")

    candidates = []
    for q, recs in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        if len(recs) < min_count:
            continue
        kinds = {classify(r) for r in recs}
        scores = [r["top_score"] for r in recs if r.get("top_score") is not None]
        sources = collections.Counter(
            s.get("source") for r in recs for s in (r.get("retrieved") or []) if s.get("source")
        )
        candidates.append({
            "question": q,
            "seen": len(recs),
            "kinds": sorted(k for k in kinds if k),
            "top_score": round(max(scores), 4) if scores else None,
            "retrieved": [s for s, _ in sources.most_common(3)],
        })

    if not candidates:
        n_failures = sum(1 for r in records if classify(r) is not None)
        if n_failures == 0:
            print("\nNo failures logged — every record is an upvote. Nothing to harvest.")
        else:
            print(f"\n{n_failures} failure(s) logged, but none are new: already in the golden "
                  f"set, or below --min-count={min_count}.")
        return

    print(f"\n=== {len(candidates)} golden-set candidate(s), most-repeated first ===")
    for c in candidates:
        print(f"\n  [{'+'.join(c['kinds'])}] seen {c['seen']}x  top_score={c['top_score']}")
        print(f"    Q: {c['question'][:110]}")
        if c["retrieved"]:
            print(f"    retrieved: {', '.join(c['retrieved'])}")
        # The judgement a human has to make, stated per candidate rather than in the abstract.
        if "abstained" in c["kinds"]:
            print("    -> decide: corpus truly lacks this (answerable:false, a new abstention")
            print("       trap), or retrieval missed it (answerable:true, a recall bug)?")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for c in candidates:
                # golden_set.jsonl shape, with `answerable` left null ON PURPOSE: that is
                # exactly the call a human must make, and defaulting it would quietly
                # manufacture ground truth, which is the one thing an eval set cannot survive.
                fh.write(json.dumps({
                    "id": None, "question": c["question"], "answerable": None,
                    "_harvest": {"seen": c["seen"], "kinds": c["kinds"],
                                 "top_score": c["top_score"], "retrieved": c["retrieved"]},
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path} — review, set `id` and `answerable`, then append to "
              f"eval/golden_set.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstained-only", action="store_true",
                    help="only harvest abstentions (the scarce class calibration needs)")
    ap.add_argument("--out", default=None, help="write candidates as JSONL for review")
    ap.add_argument("--min-count", type=int, default=1,
                    help="only propose questions seen at least N times")
    args = ap.parse_args()
    main(args.abstained_only, args.out, args.min_count)
