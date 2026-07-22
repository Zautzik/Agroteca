"""Stage 2 — normalization. Each function fixes a specific failure we hit in Phase 1.

Order matters. Run `strip_repeated_lines` on the raw page list first (it needs the
original line breaks to spot running headers/footers), then run `normalize_page_text`
on each page's text.
"""
import re
from collections import Counter

_SOFT_HYPHEN = "­"

# Glyph-name digits, e.g. "The Market Gardener" encodes 30 as "/three.o/zero.o".
_GLYPH_DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break: 'ori-\\nginarias' -> 'originarias'."""
    text = text.replace(_SOFT_HYPHEN, "")
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_glyph_digits(text: str) -> str:
    """Map PDF glyph names like '/three.o' back to '3'."""
    return re.sub(r"/([a-z]+)\.o", lambda m: _GLYPH_DIGITS.get(m.group(1), m.group(0)), text)


def normalize_whitespace(text: str) -> str:
    """Collapse spaces; keep blank lines (paragraph breaks); single newline -> space."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)          # collapse runs of blank lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)     # a lone newline is a wrap, not a break
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_page_text(text: str) -> str:
    """Full per-page clean-up: dehyphenate -> fix glyphs -> normalize whitespace."""
    return normalize_whitespace(fix_glyph_digits(dehyphenate(text)))


def strip_repeated_lines(pages: list[dict], min_pages: int = 3, min_frac: float = 0.5) -> list[dict]:
    """Drop short lines that recur on many pages (running headers/footers/titles).

    A line is boilerplate if it appears on >= max(min_pages, min_frac*N) pages and
    is short (< 80 chars). Bare page numbers differ per page, so they're handled
    separately by dropping standalone 1-4 digit lines.
    """
    per_page_lines = [[ln.strip() for ln in p["text"].splitlines()] for p in pages]
    counts: Counter = Counter()
    for lines in per_page_lines:
        for ln in {l for l in lines if l}:
            counts[ln] += 1

    n = max(len(pages), 1)
    threshold = max(min_pages, int(min_frac * n))
    boiler = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 80}

    out = []
    for page, lines in zip(pages, per_page_lines):
        kept = [
            ln for ln in lines
            if ln not in boiler and not re.fullmatch(r"\d{1,4}", ln)  # drop bare page numbers
        ]
        out.append({**page, "text": "\n".join(kept)})
    return out
