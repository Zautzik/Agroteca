"""chunk_document drives page-level citations through char_start/char_end, so an off-by-one in
the overlap-rewind or the always-make-progress guard would corrupt a citation *without raising*.
These pin the invariants that guard against that silent failure.

Note: this chunker does NOT cover every character -- inter-unit separators (blank lines) fall
between chunks by design. The load-bearing guarantees are: offsets slice back exactly, no
*content* is dropped, overlap carries boundary units forward, and it always terminates.
"""
import re

from agroteca.ingest.chunk import chunk_document

PARAS = "\n\n".join(f"Paragraph {i} about soil and water and nitrogen-fixing crops." for i in range(12))


def test_offsets_slice_back_exactly():
    # THE citation invariant: char_start/char_end must reproduce the chunk's own text.
    for c in chunk_document(PARAS, target_chars=120, overlap_chars=30):
        assert PARAS[c.char_start:c.char_end] == c.text


def test_chunks_ordered_and_nondegenerate():
    chunks = chunk_document(PARAS, target_chars=120, overlap_chars=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.char_start < c.char_end          # never a zero-width chunk
        assert c.text.strip()                      # never an empty chunk
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)                # forward progress -- never rewinds past the last start


def test_no_content_is_dropped():
    # Every paragraph's content survives into some chunk -- the real "no unintended gap" guarantee.
    joined = " ".join(c.text for c in chunk_document(PARAS, target_chars=120, overlap_chars=30))
    for i in range(12):
        assert f"Paragraph {i} " in joined


def test_overlap_carries_boundary_units_forward():
    # Tiny units + an overlap budget larger than a unit => trailing units re-appear in the next
    # chunk, so a boundary-spanning fact is never orphaned between two chunks.
    text = "\n\n".join(f"Fact{i}." for i in range(20))
    chunks = chunk_document(text, target_chars=40, overlap_chars=20)
    assert len(chunks) > 1
    shared = set(re.findall(r"Fact\d+", chunks[0].text)) & set(re.findall(r"Fact\d+", chunks[1].text))
    assert shared, "overlap-rewind should re-include trailing units in the next chunk"


def test_terminates_on_pathological_input():
    # The hard-cut path (one token longer than target) and empty/blank inputs must terminate and
    # keep honest offsets -- no infinite loop, no crash.
    for text in ("x" * 5000, "", "   ", "one short line with no separators"):
        chunks = chunk_document(text, target_chars=100, overlap_chars=20)
        for c in chunks:
            assert text[c.char_start:c.char_end] == c.text
