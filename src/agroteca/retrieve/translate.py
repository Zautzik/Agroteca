"""Query-side translation — retrieve in a bilingual space so a Spanish question can reach an
English answer chunk (and vice-versa).

Measured to recover a cross-lingual miss (q08) that no embedder swap could — the query-side fix
the embedder experiment pointed to (eval/query_translation.py; docs/masterclass.md 4.14). This
module owns ONLY the translation; the retrieval stays LLM-free (rerank_scored_bilingual takes the
translated string as an argument). It **fails soft**: any error returns None, and the caller falls
back to original-only retrieval — so translation is a bonus that can never make retrieval worse.
"""
import httpx
from ollama import Client

from agroteca.config import settings

# Own client (retrieval must not import from generate.py -> would be circular). Short connect
# timeout so an unreachable LLM fails fast into the original-only fallback.
_client = Client(host=settings.gen_base_url,
                 timeout=httpx.Timeout(settings.translate_timeout, connect=10.0))

_PROMPT = (
    "Translate this agricultural question to the other language: if it is in Spanish, give the "
    "English; if it is in English, give the Spanish. Preserve technical terms exactly. "
    "Output ONLY the translation — no quotes, no preamble.\n\n{q}"
)


def translate_query(question: str) -> str | None:
    """The question in the other of {ES, EN}, or None if translation is unavailable/empty.

    Language detection is left to the (multilingual) model rather than a brittle heuristic. On any
    failure this returns None on purpose — the caller treats translation as optional."""
    try:
        resp = _client.chat(
            model=settings.gen_model,
            messages=[{"role": "user", "content": _PROMPT.format(q=question)}],
            options={"num_predict": 128},
        )
        out = resp["message"]["content"].strip().strip('"').strip()
        # Guard the degenerate cases: empty, or the model echoed the question unchanged.
        return out if out and out.lower() != question.strip().lower() else None
    except Exception:
        return None
