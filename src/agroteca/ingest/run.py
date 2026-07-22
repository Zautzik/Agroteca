"""Ingestion orchestrator: manifests -> extract -> normalize -> chunk -> embed -> store.

Which tiers load is governed by config (LOCAL_MODE / include_distractor), so the
deploy build literally cannot ingest the copyrighted tier.

Usage:
    uv run python -m agroteca.ingest.run                 # ingest configured tiers
    uv run python -m agroteca.ingest.run --limit 2       # smoke test: first 2 docs
    AGROTECA_LOCAL_MODE=1 uv run python -m agroteca.ingest.run
"""
import argparse
import csv
import re
import sys

from agroteca.config import settings
from agroteca.ingest import store
from agroteca.ingest.chunk import chunk_document
from agroteca.ingest.embed import embed_passages
from agroteca.ingest.extract import extract_pages
from agroteca.ingest.normalize import normalize_page_text, strip_repeated_lines

MANIFESTS = {
    "open": "manifest.csv",
    "local": "manifest_local.csv",
    "synthetic": "manifest_synthetic.csv",
    "distractor": "manifest_distractor.csv",
}

# Small stopword set (ES+EN) for the text-layer gate — a doc that scores near zero
# after extraction is image-only and must not reach the chunker.
_STOP = {"de","la","el","que","en","los","las","del","por","con","para","una","un","se",
         "es","al","como","y","no","the","of","and","to","in","a","is","for","that","with",
         "as","are","on","be","by","this","it","or","from","an","at"}


def doc_id_for(filename: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", filename.lower()).strip("_")


def load_rows(tier: str) -> list[dict]:
    path = settings.data_dir / MANIFESTS[tier]
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "_tier": tier} for r in csv.DictReader(f)]


def source_path(row: dict):
    base = settings.synthetic_dir if row["_tier"] == "synthetic" else settings.raw_dir
    return base / row["filename"]


def has_text_layer(text: str) -> bool:
    toks = re.findall(r"[a-záéíóúñü]+", text.lower())
    if len(toks) < 30:
        return False
    return sum(t in _STOP for t in toks) / len(toks) >= 0.04


def assemble(path):
    """Return (full_text, page_spans) where page_spans = [(start, end, page), ...]."""
    if path.suffix.lower() == ".md":
        pages = [{"page": 1, "text": path.read_text(encoding="utf-8")}]
    else:
        pages = strip_repeated_lines(extract_pages(path))
    parts, spans, pos = [], [], 0
    for p in pages:
        norm = normalize_page_text(p["text"])
        if not norm:
            continue
        parts.append(norm)
        spans.append((pos, pos + len(norm), p["page"]))
        pos += len(norm) + 2  # for the "\n\n" join
    return "\n\n".join(parts), spans


def page_of(offset: int, spans) -> int | None:
    for s, e, pg in spans:
        if s <= offset < e:
            return pg
    return spans[-1][2] if spans else None


def ingest(limit: int | None = None) -> None:
    conn = store.connect()
    tiers = settings.tiers()
    print(f"tiers: {tiers} | model: {settings.embed_model} | chunk~{settings.chunk_tokens}tok")
    docs = chunks_total = skipped = 0

    rows = [r for tier in tiers for r in load_rows(tier)]
    if limit:
        rows = rows[:limit]

    for row in rows:
        path = source_path(row)
        name = row["filename"]
        if not path.exists():
            print(f"[missing] {name}")
            continue
        full, spans = assemble(path)
        if not has_text_layer(full):
            print(f"[no-text] {name} — needs OCR, skipped")
            skipped += 1
            continue

        pieces = chunk_document(full, settings.chunk_chars, settings.overlap_chars)
        embeddings = embed_passages([c.text for c in pieces])
        did = doc_id_for(name)

        store.wipe_chunks(conn, did)
        store.upsert_document(conn, {
            "doc_id": did, "source_file": name, "title": row.get("title"),
            "tier": row["_tier"], "lang": row.get("lang"), "topic": row.get("topic"),
            "url": row.get("url") or None,
        })
        chunk_rows = [{
            "chunk_id": f"{did}#{i:04d}", "doc_id": did, "chunk_index": i,
            "text": c.text, "embedding": emb, "tier": row["_tier"],
            "lang": row.get("lang"), "topic": row.get("topic"),
            "page": page_of(c.char_start, spans),
            "char_start": c.char_start, "char_end": c.char_end,
        } for i, (c, emb) in enumerate(zip(pieces, embeddings))]
        store.insert_chunks(conn, chunk_rows)
        conn.commit()

        docs += 1
        chunks_total += len(chunk_rows)
        print(f"[ok] {name[:52]:52} tier={row['_tier']:9} chunks={len(chunk_rows)}")

    conn.close()
    print(f"\n=== ingested {docs} docs, {chunks_total} chunks; skipped {skipped} (no text) ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ingest(limit=args.limit)
