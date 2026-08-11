"""Retrieval metrics — the single definition, imported by every eval.

The metric previously lived, duplicated, in each eval script, where it could drift or be
mislabeled independently; consolidating it here is the de-risking `validate_golden.py`
already applies to the data. Every figure derives from one primitive, `answer_rank` (the
1-based position of the answer chunk): set-membership metrics (answer@k) and ranking-quality
metrics (answer@1, MRR) both fall out of a rank — which matters because reordering is what a
reranker does, and answer@5 alone cannot see it.
"""
import json
import re
from pathlib import Path


def norm(s: str) -> str:
    """Collapse whitespace and lowercase, so a `must_contain` phrase matches despite the
    spacing and soft-hyphen line breaks that PDF extraction introduces."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_answerable(golden_path) -> list[dict]:
    """Golden questions with a checkable answer: `answerable` AND carrying a `must_contain`."""
    out = []
    for line in Path(golden_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            if q.get("answerable", True) and q.get("must_contain"):
                out.append(q)
    return out


def answer_rank(rows, source_file: str, must_contain: str) -> int | None:
    """1-based rank of the FIRST retrieved chunk that is the *answer chunk* — right file AND
    the `must_contain` phrase present — or None if no row qualifies.

    `rows`: an ordered (best-first) iterable of `(source_file, text)`.
    BOTH conditions matter: the right file alone is only a *document* hit, not an
    *answer-chunk* hit — and the phrase must land in a chunk *of that file*, so a decoy chunk
    from a different file that happens to contain the phrase does not count.
    """
    needle = norm(must_contain)
    for rank, (sf, text) in enumerate(rows, start=1):
        if sf == source_file and needle in norm(text):
            return rank
    return None


def document_hit(rows, source_file: str) -> bool:
    """Did the right document appear at any rank? (the weaker `doc@k` signal)"""
    return any(sf == source_file for sf, _text in rows)


def answer_at_k(rank: int | None, k: int) -> bool:
    """Is the answer chunk within the top k? Set membership — blind to order *inside* k."""
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    """1/rank — MRR's per-question term. Rewards floating the answer *higher* (what a
    reranker does), which `answer@k` cannot see. No hit -> 0.0."""
    return 1.0 / rank if rank else 0.0


def score(questions, retrieve, k: int = 5) -> dict:
    """Aggregate every metric over `questions` in ONE pass — the single place they're defined.

    `retrieve(question, k)` must return the top-k rows as `(source_file, text)`, best first.
    Add a metric here and it appears in every eval that imports this — one edit, no drift.
    """
    n = len(questions)
    hits1 = hitsk = docs = 0
    rr = 0.0
    misses = []
    for q in questions:
        rows = list(retrieve(q["question"], k))
        rank = answer_rank(rows, q["source_file"], q["must_contain"])
        hits1 += answer_at_k(rank, 1)
        hitsk += answer_at_k(rank, k)
        docs += document_hit(rows, q["source_file"])
        rr += reciprocal_rank(rank)
        if not answer_at_k(rank, k):
            misses.append(q["id"])
    return {
        "n": n,
        "answer@1": hits1 / n,
        f"answer@{k}": hitsk / n,
        f"document@{k}": docs / n,
        f"mrr@{k}": rr / n,
        "misses": misses,
    }
