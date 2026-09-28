"""Grounded off-ramps for a question the corpus could not answer.

When the model abstains, the UI currently dead-ends. The fix is to hand the reader
somewhere to go -- but every suggestion has to come from something retrieval actually
returned, never from the model's imagination. Inventing a plausible-sounding rephrasing
that also fails is worse than a clean "no": it costs the reader another full query
(minutes, on CPU) and spends the one thing this system sells, which is that it would
rather say it doesn't know than make something up.

So the raw material is what we already hold at abstention time: the reranked candidates,
with their scores, tiers and topics. Even a failed query lands *somewhere* in the corpus,
and where it landed is real information.

Topics are ranked by RELEVANCE TO THE FAILED QUERY, not by how many documents the corpus
happens to hold on each subject. Doc-count ranking sounds reasonable and is actively
misleading: it would answer a question about aphids with "Agricultural law (7 documents)"
purely because law has the most PDFs. Corpus-wide counts survive only as a fallback, for
when retrieval returned too little to rank at all.

Pure functions, no DB and no model -- the caller passes rows in, which keeps this unit
testable without standing up Postgres or loading a cross-encoder.
"""
from dataclasses import dataclass

from agroteca.topics import canonical

# Never advertised, whatever its topic says. The distractor tier exists to prove retrieval
# isn't fooled by an off-domain book; surfacing it would recommend the trap.
_NEVER_SUGGEST_TIERS = frozenset({"distractor"})

# Calibrated, not guessed -- eval/calibrate_abstention.py over the full golden set:
#
#   answered correctly (n=16)   top score 0.618 – 0.905   (median 0.812)
#   retrieval failed   (n=4)    top score 0.354 – 0.743
#   genuinely no answer (n=2)   top score 0.311 – 0.413
#
# Sorted, there is an empty band between 0.413 and 0.614, so 0.5 sits in open space with
# ~0.09 of margin either side. Below it: six-for-six genuine failures, zero good answers.
#
# What that measurement REFUSED to support is just as load-bearing. The original design
# wanted to tell a "near miss" (answer exists, retrieval missed it) from a "total miss"
# (corpus lacks it) -- those two populations OVERLAP (0.354 vs 0.413) and no threshold
# separates them, because the failure class is bimodal: one failed question scored 0.743,
# indistinguishable from a healthy answer, while another found nothing at all. So this
# classifier deliberately does NOT claim to know which kind of failure it is.
#
# The rule is also ONE-DIRECTIONAL. A low score reliably means nothing relevant surfaced.
# A high score means nothing at all -- two genuine failures sit at 0.614 and 0.743, inside
# the healthy band. Never phrase a high-score abstention to the reader as if retrieval
# succeeded.
NO_MATCH_BELOW = 0.5


def classify(candidates: list[tuple[object, object, float]]) -> str:
    """After an abstention: did retrieval surface anything convincing at all?

    'no_match'   -- top score below the calibrated band. Nothing relevant came back, and
                    whether that is a corpus gap or a retrieval failure is NOT knowable
                    from the score. Offer the corpus, not a diagnosis.
    'weak_match' -- retrieval returned confident-looking candidates the model still would
                    not answer from. Offer those documents so the reader can judge.

    Takes the same (topic, tier, score) rows as rank_topics so a caller passes one list.
    """
    scores = [s for *_rest, s in candidates]
    if not scores or max(scores) < NO_MATCH_BELOW:
        return "no_match"
    return "weak_match"


@dataclass(frozen=True)
class TopicChip:
    key: str
    label: str
    score: float | None      # best matching chunk's score; None if it came from fallback
    from_retrieval: bool     # True = related to what you asked; False = general corpus

    @property
    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "score": None if self.score is None else round(self.score, 4),
                "from_retrieval": self.from_retrieval}


def rank_topics(
    candidates: list[tuple[str | None, str | None, float]],
    lang: str = "es",
    servable_tiers: frozenset[str] | set[str] | None = None,
    limit: int = 6,
    fallback_counts: list[tuple[str, int]] | None = None,
) -> list[TopicChip]:
    """Rank corpus topics by how well they matched the query that just failed.

    candidates: (raw_topic, tier, score) per retrieved chunk, best first.
    servable_tiers: restrict to tiers this deployment can actually serve. None = no
        restriction. A public deploy must pass its servable set, or the list can offer a
        subject whose only documents are the copyrighted `local` tier -- advertising
        something the reader can never be shown.
    fallback_counts: (raw_topic, doc_count) corpus-wide, used ONLY to backfill when
        retrieval yielded too few distinct topics to be a useful off-ramp.

    A topic scores by its BEST matching chunk, not the sum over its chunks. Summing would
    reward whichever subject happens to own the most chunks in the pool -- a long document
    outranking a short, precise one -- which is the doc-count bias this function exists to
    avoid. Ties break toward the topic that appeared more often.
    """
    best: dict[str, float] = {}
    seen: dict[str, int] = {}
    labels: dict[str, str] = {}

    for raw, tier, score in candidates:
        if tier in _NEVER_SUGGEST_TIERS:
            continue
        if servable_tiers is not None and tier not in servable_tiers:
            continue
        c = canonical(raw)
        if c is None:              # unmapped or deliberately hidden -> never surfaced
            continue
        key = c["key"]
        labels[key] = c.get(lang) or c["en"]
        seen[key] = seen.get(key, 0) + 1
        if key not in best or score > best[key]:
            best[key] = score

    chips = [
        TopicChip(key=k, label=labels[k], score=best[k], from_retrieval=True)
        for k in sorted(best, key=lambda k: (-best[k], -seen[k], labels[k]))
    ]

    if len(chips) < limit and fallback_counts:
        # Re-aggregate by canonical key before ordering. The caller counts by RAW topic
        # (that is what the database column holds), so a merged subject arrives split
        # across several rows -- soil_fertility as suelos(5) + nutrition(2) + compost(1).
        # Ordering the raw rows would rank it on 5 instead of its real 8 and bury the
        # largest subject in the corpus below two smaller ones.
        merged: dict[str, int] = {}
        merged_label: dict[str, str] = {}
        for raw, count in fallback_counts:
            c = canonical(raw)
            if c is None:
                continue
            merged[c["key"]] = merged.get(c["key"], 0) + count
            merged_label[c["key"]] = c.get(lang) or c["en"]

        have = {c.key for c in chips}
        for key in sorted(merged, key=lambda k: (-merged[k], merged_label[k])):
            if len(chips) >= limit:
                break
            if key in have:
                continue
            have.add(key)
            chips.append(TopicChip(key=key, label=merged_label[key],
                                   score=None, from_retrieval=False))

    return chips[:limit]


def nearby_documents(
    candidates: list[tuple[str, str | None, float]],
    servable_tiers: frozenset[str] | set[str] | None = None,
    limit: int = 3,
) -> list[dict]:
    """The actual documents retrieval surfaced, deduped, best first.

    candidates: (source_file, tier, score) per retrieved chunk.

    This is the more concrete half of the off-ramp: naming the documents we *did* find
    lets a reader judge for themselves whether the corpus is in the right neighbourhood,
    rather than taking "not found" on faith.
    """
    best: dict[str, float] = {}
    for source, tier, score in candidates:
        if tier in _NEVER_SUGGEST_TIERS:
            continue
        if servable_tiers is not None and tier not in servable_tiers:
            continue
        if not source:
            continue
        if source not in best or score > best[source]:
            best[source] = score
    ordered = sorted(best, key=lambda s: -best[s])[:limit]
    return [{"source": s, "score": round(best[s], 4)} for s in ordered]
