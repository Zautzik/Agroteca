"""Validate eval/golden_set.jsonl against the corpus manifests.

Answerable golden questions may cite a source from either the open/deployable
tier (manifest.csv) or the copyrighted local-only tier (manifest_local.csv),
which is loaded only when the app runs locally. Distractor documents
(manifest_distractor.csv) are intentionally NOT valid answer sources.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Answerable sources are valid if present in the open, local-only, or synthetic tier.
ANSWERABLE_MANIFESTS = ["manifest.csv", "manifest_local.csv", "manifest_synthetic.csv"]
manifest_files = set()
for _m in ANSWERABLE_MANIFESTS:
    _p = ROOT / "data" / _m
    if _p.exists():
        with open(_p, newline="", encoding="utf-8") as f:
            manifest_files |= {row["filename"] for row in csv.DictReader(f)}

REQUIRED = {"id", "lang", "question", "reference_answer", "source_file", "source_page", "must_contain"}
seen_ids = set()

with open(ROOT / "eval" / "golden_set.jsonl", encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        if not line.strip():
            continue
        q = json.loads(line)
        missing = REQUIRED - q.keys()
        assert not missing, f"line {n}: missing keys {missing}"
        assert q["id"] not in seen_ids, f"line {n}: duplicate id {q['id']}"
        seen_ids.add(q["id"])
        answerable = q.get("answerable", True)
        if answerable:
            assert q["source_file"] in manifest_files, f"line {n}: {q['source_file']} not in manifest"
            assert q["must_contain"], f"line {n}: answerable question needs must_contain"

print(f"OK: {len(seen_ids)} questions validated against {len(manifest_files)} answerable manifest entries "
      f"(open + local-only + synthetic tiers)")
