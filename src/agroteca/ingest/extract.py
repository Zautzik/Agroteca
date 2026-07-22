"""Stage 1 — extraction. Pull text out of a PDF, one entry per page.

We use PyMuPDF (imported as `pymupdf`); it preserves reading order far better
than pypdf, which matters for multi-column agronomy books and legal documents.
"""
from pathlib import Path

import pymupdf


def extract_pages(path: Path) -> list[dict]:
    """Return [{'page': <1-based int>, 'text': <str>}, ...] for a PDF.

    Empty/near-empty pages are kept (an image-only page yields ''), so the
    caller's text-layer gate can still see and reject a no-text document.
    """
    pages: list[dict] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({"page": i, "text": page.get_text("text")})
    return pages
