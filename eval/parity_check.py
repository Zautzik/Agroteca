"""Parity gate for the hosted profile: does the public demo run the SAME models as the eval?

The hosted profile swaps two runtimes -- the torch embedder for ONNX Runtime, the local
cross-encoder for Jina's API -- while keeping the models. That is only honest if the outputs
match, so this script measures it before anything is deployed:

  1. Embedding parity. Every ONNX export of the embedder in its Hugging Face repo is compared with
     the sentence-transformers reference on the golden questions and reference answers: cosine
     per text (min / mean) and file size. A variant passes at min cosine >= --min-cosine.
  2. Retrieval parity (--db): dense top-20 overlap, reference vs each passing variant, per question.
  3. Rerank parity (--db, needs AGROTECA_JINA_API_KEY): local cross-encoder vs Jina's API on the
     same hybrid pools -- Spearman rank correlation, top-5 agreement, and the score range (the UI's
     low-confidence cutoff assumes scores in [0, 1]).

Run on the machine that holds the index, with the local extra installed:
    uv sync --extra local
    uv run python eval/parity_check.py                 # 1 only (no DB, no keys)
    uv run python eval/parity_check.py --db            # 1 + 2 + 3 against AGROTECA_DB_URL
Writes results/parity.md. Pick AGROTECA_ONNX_MODEL_FILE from a passing variant only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agroteca.config import settings

GOLDEN = settings.root / "eval" / "golden_set.jsonl"
OUT = settings.root / "results" / "parity.md"


def _golden() -> tuple[list[str], list[str]]:
    """(questions, questions + reference answers)."""
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [r["question"] for r in rows]
    return questions, questions + [r["reference_answer"] for r in rows if r.get("reference_answer")]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def _onnx_variants() -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    info = HfApi().model_info(settings.embed_model, revision=settings.onnx_model_revision,
                              files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings
                  if s.rfilename.startswith("onnx/") and s.rfilename.endswith(".onnx"))


def _onnx_embed(filename: str, texts: list[str]) -> np.ndarray:
    from agroteca.ingest import onnx_embed
    settings.onnx_model_file = filename
    onnx_embed._runtime.cache_clear()
    return onnx_embed.embed_texts_onnx(texts)


def _spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def main(use_db: bool, min_cosine: float) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from sentence_transformers import SentenceTransformer
    questions, texts = _golden()
    ref_model = SentenceTransformer(settings.embed_model, device="cpu")
    ref = ref_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    print(f"reference: {settings.embed_model} (max_seq_length={ref_model.max_seq_length}), "
          f"{len(texts)} texts")
    if ref_model.max_seq_length != settings.embed_max_tokens:
        print(f"WARNING: reference max_seq_length {ref_model.max_seq_length} != "
              f"AGROTECA_EMBED_MAX_TOKENS {settings.embed_max_tokens}")

    lines = ["# Hosted-profile parity check", "",
             f"Embedder `{settings.embed_model}` @ `{settings.onnx_model_revision}`, "
             f"{len(texts)} texts (golden questions + reference answers).", "",
             "## 1. Embedding parity (ONNX variant vs sentence-transformers)", "",
             "| ONNX file | Size (MB) | Min cosine | Mean cosine | Pass |", "|---|---|---|---|---|"]
    passing = []
    for filename, size in _onnx_variants():
        try:
            got = _onnx_embed(filename, texts)
        except Exception as exc:                      # a variant this runtime can't load
            lines.append(f"| `{filename}` | {size / 1e6:.0f} | error: {type(exc).__name__} | | no |")
            print(f"{filename}: ERROR {exc}")
            continue
        c = _cos(ref, got)
        ok = float(c.min()) >= min_cosine
        if ok:
            passing.append(filename)
        lines.append(f"| `{filename}` | {size / 1e6:.0f} | {c.min():.5f} | {c.mean():.5f} | "
                     f"{'yes' if ok else 'no'} |")
        print(f"{filename}: {size / 1e6:.0f} MB  min={c.min():.5f}  mean={c.mean():.5f}  "
              f"{'PASS' if ok else 'fail'}")

    if use_db:
        from agroteca.ingest import store
        from agroteca.retrieve.hybrid import hybrid_search
        conn = store.connect()
        sql = ("SELECT chunk_id FROM chunks WHERE tier = ANY(%s) "
               "ORDER BY embedding <=> %s LIMIT 20")
        tiers = settings.tiers()
        lines += ["", "## 2. Dense retrieval parity (top-20 overlap with the reference)", "",
                  "| ONNX file | Mean overlap | Min overlap |", "|---|---|---|"]
        ref_ids = [[r[0] for r in conn.execute(sql, (tiers, ref_model.encode(q))).fetchall()]
                   for q in questions]
        for filename in passing:
            vecs = _onnx_embed(filename, questions)
            overlaps = []
            for q_ids, v in zip(ref_ids, vecs):
                got_ids = [r[0] for r in conn.execute(sql, (tiers, v)).fetchall()]
                overlaps.append(len(set(q_ids) & set(got_ids)) / max(len(q_ids), 1))
            lines.append(f"| `{filename}` | {np.mean(overlaps):.3f} | {min(overlaps):.3f} |")
            print(f"{filename}: dense top-20 overlap mean={np.mean(overlaps):.3f} min={min(overlaps):.3f}")

        if settings.jina_api_key:
            # Fetch every candidate pool BEFORE loading the local cross-encoder, then close the
            # connection: loading it and reranking on CPU take minutes, and a managed Postgres
            # (Neon) drops a connection left idle that long.
            pools = [[row[2] for row in hybrid_search(conn, q, k=settings.rerank_candidates,
                                                      tiers=tiers)]
                     for q in questions[:5]]
            conn.close()
            from agroteca.retrieve.rerank import _CrossEncoder, _JinaAPIReranker
            local = _CrossEncoder(settings.rerank_model, settings.resolved_device,
                                  settings.rerank_model_revision)
            api = _JinaAPIReranker(settings.jina_rerank_url, settings.jina_rerank_model,
                                   settings.jina_api_key,
                                   settings.jina_timeout)
            lines += ["", "## 3. Rerank parity (local cross-encoder vs Jina API, same pools)", "",
                      "| Question | Spearman | Top-5 agreement | Local range | API range |",
                      "|---|---|---|---|---|"]
            for i, (q, pool) in enumerate(zip(questions[:5], pools)):
                a, b = local.rerank(q, pool), api.rerank(q, pool)
                top_a = set(np.argsort(a)[::-1][:5]); top_b = set(np.argsort(b)[::-1][:5])
                lines.append(f"| q{i + 1} | {_spearman(a, b):.3f} | {len(top_a & top_b)}/5 | "
                             f"{min(a):.3f}–{max(a):.3f} | {min(b):.3f}–{max(b):.3f} |")
                print(f"q{i + 1}: spearman={_spearman(a, b):.3f} top5={len(top_a & top_b)}/5 "
                      f"local=[{min(a):.3f},{max(a):.3f}] api=[{min(b):.3f},{max(b):.3f}]")
        else:
            conn.close()
            lines += ["", "## 3. Rerank parity", "", "Skipped: AGROTECA_JINA_API_KEY not set."]

    lines += ["", f"Passing ONNX variants (min cosine >= {min_cosine}): "
              + (", ".join(f"`{p}`" for p in passing) or "**none**")]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true", help="also check dense and rerank parity")
    ap.add_argument("--min-cosine", type=float, default=0.999)
    args = ap.parse_args()
    main(args.db, args.min_cosine)
