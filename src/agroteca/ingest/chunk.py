"""Stage 3 — chunking. Recursive, boundary-aware, with overlap and exact offsets.

We split on the largest natural boundary that fits (paragraph -> sentence -> hard
cut), then pack whole units into chunks up to a target size, seeding each new chunk
with the trailing units of the previous one so a boundary-spanning fact survives.

Every chunk carries exact `char_start`/`char_end` into the document text, which
gives us honest provenance and lets `run.py` map a chunk back to its page.
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int


def _split_offsets(text: str, sep: str) -> list[tuple[int, int]]:
    """(start, end) spans of the pieces of `text` split on `sep`, offsets into `text`."""
    spans, pos = [], 0
    for piece in text.split(sep):
        spans.append((pos, pos + len(piece)))
        pos += len(piece) + len(sep)
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split text[start:end] into sentence spans (offsets into the full `text`)."""
    sub = text[start:end]
    spans = []
    for m in re.finditer(r".*?[.!?](?:\s+|$)|.+$", sub, re.S):
        if m.group(0).strip():
            spans.append((start + m.start(), start + m.end()))
    return spans or [(start, end)]


def _unit_spans(text: str, max_unit: int) -> list[tuple[int, int]]:
    """Atomic units: paragraphs, splitting oversized ones into sentences, then hard cuts."""
    spans: list[tuple[int, int]] = []
    for ps, pe in _split_offsets(text, "\n\n"):
        if not text[ps:pe].strip():
            continue
        if pe - ps <= max_unit:
            spans.append((ps, pe))
            continue
        for ss, se in _sentence_spans(text, ps, pe):
            if se - ss <= max_unit:
                spans.append((ss, se))
            else:  # a single very long sentence: hard cut
                k = ss
                while k < se:
                    spans.append((k, min(k + max_unit, se)))
                    k += max_unit
    return spans


def chunk_document(text: str, target_chars: int, overlap_chars: int) -> list[Chunk]:
    """Pack atomic units into overlapping chunks of ~target_chars."""
    units = _unit_spans(text, target_chars)
    chunks: list[Chunk] = []
    i, n = 0, len(units)

    while i < n:
        # Grow a window [i, j) until the next unit would overflow the target.
        j, cur_len = i, 0
        while j < n:
            span_len = units[j][1] - units[j][0] + (1 if j > i else 0)
            if cur_len + span_len > target_chars and j > i:
                break
            cur_len += span_len
            j += 1

        start, end = units[i][0], units[j - 1][1]
        chunks.append(Chunk(text=text[start:end], char_start=start, char_end=end))
        if j >= n:
            break

        # Overlap: rewind so the next window re-includes trailing units within the budget.
        new_i, back = j, 0
        k = j - 1
        while k > i:
            back += units[k][1] - units[k][0]
            if back > overlap_chars:
                break
            new_i = k
            k -= 1
        i = new_i if new_i > i else j  # always make progress

    return chunks
