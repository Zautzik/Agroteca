"""Query-side translation — retrieve in a bilingual space so a Spanish question can reach an
English answer chunk (and vice-versa).

Measured to recover a cross-lingual miss (q08) that no embedder swap could — the query-side fix
the embedder experiment pointed to (eval/query_translation.py; docs/masterclass.md 4.14). This
module owns ONLY the translation; the retrieval stays LLM-free (rerank_scored_bilingual takes the
translated string as an argument). It **fails soft**: any error returns None, and the caller falls
back to original-only retrieval — so translation is a bonus that can never make retrieval worse.
"""
from agroteca.config import settings
from agroteca.llm import make_llm

# Own backend instance (retrieval must not import from generate.py -> would be circular), built
# lazily with the shorter translate timeout so an unreachable LLM fails fast into the fallback.
_llm = None

_PROMPT = (
    "Translate this agricultural question to the other language: if it is in Spanish, give the "
    "English; if it is in English, give the Spanish. Preserve technical terms exactly. "
    "Output ONLY the translation — no quotes, no preamble.\n\n{q}"
)


def translate_query(question: str) -> str | None:
    """The question in the other of {ES, EN}, or None if translation is unavailable/empty.

    Language detection is left to the (multilingual) model rather than a brittle heuristic. On any
    failure this returns None on purpose — the caller treats translation as optional."""
    global _llm
    try:
        if _llm is None:
            _llm = make_llm(settings.translate_timeout)
        out = _llm.chat(None, _PROMPT.format(q=question), 128).strip().strip('"').strip()
        # Guard the degenerate cases: empty, or the model echoed the question unchanged.
        return out if out and out.lower() != question.strip().lower() else None
    except Exception:
        return None
