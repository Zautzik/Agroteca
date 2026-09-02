"""HTTP-level, concurrent eval harness: fires eval/golden_set.jsonl at the LIVE FastAPI
/retrieve endpoint (not the in-process retrievers eval/retrieval_at_k.py and
eval/compare_retrievers.py call directly), so retrieval@k accuracy and latency are measured as
a real client would see them -- network + ASGI + DB-pool-checkout overhead included, not just
the retriever function's own cost.

Ground truth: the golden set carries no chunk-id ground truth -- chunk ids are DB-generated at
ingest time, unknowable when the golden set was authored. Like every other eval in this repo, a
hit is (right source_file) AND (the `must_contain` phrase present in that chunk's text), scored
with the exact same eval/_scoring.answer_rank the rest of the suite uses -- so the accuracy
number here is directly comparable to eval/retrieval_at_k.py and eval/compare_retrievers.py.
Only answerable questions carrying a `must_contain` are scored (q17/q19 are deliberately
unanswerable and excluded), matching eval/_scoring.load_answerable's convention.

Requires a running server: `uv run uvicorn agroteca.api:app --host 127.0.0.1 --port 8000`
(add `AGROTECA_DEVICE=cpu` first if there's no GPU on this machine).

    uv run python eval/eval_pipeline.py
    uv run python eval/eval_pipeline.py --base-url http://127.0.0.1:8001 --k 5 --concurrency 10
"""
import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, ValidationError

from _scoring import answer_at_k, answer_rank

GOLDEN = Path(__file__).resolve().parent / "golden_set.jsonl"


class GoldenQuestion(BaseModel):
    """One eval/golden_set.jsonl row. Every line is validated against this BEFORE any query is
    fired, so a malformed golden set aborts the whole run up front instead of surfacing as a
    confusing KeyError partway through a concurrent batch."""
    id: str
    lang: str
    question: str
    reference_answer: str
    source_file: str
    source_page: Optional[int] = None
    must_contain: Optional[str] = None
    answerable: bool = True
    note: Optional[str] = None


class RetrievedChunk(BaseModel):
    """One row of the /retrieve response's `sources` list -- mirrors agroteca.api.RetrievedChunk."""
    chunk_id: str
    source_file: str
    page: Optional[int] = None
    tier: str
    score: float
    text: str


class RetrieveResponse(BaseModel):
    """The /retrieve endpoint's full JSON body -- mirrors agroteca.api.RetrieveResponse."""
    sources: list[RetrievedChunk]
    retrieval_ms: int
    translated_query: Optional[str] = None


class QueryResult(BaseModel):
    """Per-question outcome: what got hit (if anything), and how long it took, end to end."""
    id: str
    latency_ms: float
    rank: Optional[int]
    hit: bool
    error: Optional[str] = None


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Parse + validate every line against GoldenQuestion before running anything."""
    questions: list[GoldenQuestion] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(GoldenQuestion.model_validate_json(line))
        except ValidationError as exc:
            errors.append(f"line {lineno}: {exc}")
    if errors:
        raise SystemExit(f"{path} failed schema validation:\n" + "\n".join(errors))
    return questions


async def fire_one(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                   q: GoldenQuestion, k: int) -> QueryResult:
    """Issue one /retrieve call, validate the response shape, and score it. Never raises --
    a request/validation failure becomes a QueryResult with `error` set (and counts as a miss),
    so one bad response can't take down the whole concurrent batch."""
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post("/retrieve", json={"question": q.question})
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000
            parsed = RetrieveResponse.model_validate(resp.json())
        except (httpx.HTTPError, ValidationError) as exc:
            return QueryResult(id=q.id, latency_ms=(time.perf_counter() - t0) * 1000,
                               rank=None, hit=False, error=f"{type(exc).__name__}: {exc}")

    rows = [(c.source_file, c.text) for c in parsed.sources[:k]]
    rank = answer_rank(rows, q.source_file, q.must_contain) if q.must_contain else None
    return QueryResult(id=q.id, latency_ms=latency_ms, rank=rank,
                       hit=answer_at_k(rank, k))


async def run(base_url: str, k: int, concurrency: int, timeout: float,
              limit: Optional[int] = None) -> list[QueryResult]:
    questions = load_golden(GOLDEN)
    scorable = [q for q in questions if q.answerable and q.must_contain]
    total_scorable = len(scorable)
    if limit is not None:
        scorable = scorable[:limit]
    print(f"Loaded and validated {len(questions)} golden questions "
          f"({total_scorable} answerable+scorable"
          f"{f'; running {len(scorable)} of them (--limit {limit})' if limit else ''}).")

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        # Warmup: one throwaway call so a cold model/connection doesn't skew p95 on the first
        # question drawn from the concurrent batch below.
        try:
            await client.post("/retrieve", json={"question": scorable[0].question})
        except httpx.HTTPError as exc:
            raise SystemExit(f"could not reach {base_url}/retrieve -- is the server running, and is "
                             f"--timeout generous enough? ({type(exc).__name__}: {exc})")

        results = await asyncio.gather(*[fire_one(client, sem, q, k) for q in scorable])
    return list(results)


def _p95(values: list[float]) -> float:
    ts = sorted(values)
    return ts[max(0, round(0.95 * len(ts)) - 1)]


def render_markdown(results: list[QueryResult], k: int, base_url: str, concurrency: int) -> str:
    n = len(results)
    hits = sum(r.hit for r in results)
    errors = [r for r in results if r.error]
    latencies = [r.latency_ms for r in results if r.error is None]
    accuracy = hits / n if n else 0.0

    lines = [
        f"## eval_pipeline — retrieval@{k} against `{base_url}` (concurrency={concurrency})",
        "",
        "| metric | value |",
        "|---|---|",
        f"| questions scored | {n} |",
        f"| retrieval@{k} accuracy | {accuracy:.1%} ({hits}/{n}) |",
        f"| p50 latency | {statistics.median(latencies):.0f} ms |" if latencies else "| p50 latency | n/a |",
        f"| p95 latency | {_p95(latencies):.0f} ms |" if latencies else "| p95 latency | n/a |",
        f"| mean latency | {statistics.mean(latencies):.0f} ms |" if latencies else "| mean latency | n/a |",
        f"| max latency | {max(latencies):.0f} ms |" if latencies else "| max latency | n/a |",
        f"| request errors | {len(errors)} |",
    ]
    misses = [r.id for r in results if not r.hit and not r.error]
    if misses:
        lines += ["", f"**Misses:** {', '.join(misses)}"]
    if errors:
        lines += ["", "**Errors:**"] + [f"- `{r.id}`: {r.error}" for r in errors]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10, help="max in-flight requests")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request timeout (s); AGROTECA_DEVICE=cpu measures ~500s/query for "
                         "the torch reranker (vs. ~25-30s under the old ONNX backend) -- default "
                         "is generous for that path. A GPU server needs far less.")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N answerable questions (fast smoke test)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = asyncio.run(run(args.base_url, args.k, args.concurrency, args.timeout, args.limit))
    print()
    print(render_markdown(results, args.k, args.base_url, args.concurrency))


if __name__ == "__main__":
    main()
