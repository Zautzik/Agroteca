"""Canonical topic vocabulary — DRAFT, pending review.

The manifests carry a free-text `topic` per document (72/72 populated, 27 distinct
values). That raw vocabulary can't be shown to a user as-is for two reasons:

  1. It is bilingually fragmented. `agua`/`water`, `crops`/`cultivos` and
     `horticulture`/`horticultura` are the same subject under two labels, so a
     "what this corpus covers" list would show near-duplicates. Note the split does
     NOT track document language -- both `horticultura` docs are English books --
     so the ES/EN label pairs are arbitrary, not meaningful.
  2. One value is not a subject at all: `chemistry` is the planted distractor
     ("Organic Chemistry, 6th ed.", tier=distractor), which exists to prove
     retrieval isn't fooled by an off-domain book. Surfacing it would advertise a
     document the corpus deliberately does not want to answer from.

CANONICAL maps each raw manifest value to a stable key plus a label per UI language.
`key = None` means "never show this to a user".

Callers must ALSO filter by tier before building a topic list: `local` is the
copyrighted tier a public deployment cannot serve, so a topic whose only documents
are `local` must not be advertised on a deploy that excludes it. Doc counts below
are raw (all tiers) and are for review context only -- compute real counts from the
DB against the servable tier set.
"""

CANONICAL: dict[str, dict | None] = {
    # --- merged ES/EN duplicate pairs -------------------------------------
    # horticulture(3) + horticultura(2). Both `horticultura` docs are English
    # (Market Gardener, Square Foot Gardening) -- the label is arbitrary.
    "horticulture":      {"key": "horticulture", "es": "Horticultura",            "en": "Horticulture"},
    "horticultura":      {"key": "horticulture", "es": "Horticultura",            "en": "Horticulture"},
    # agua(2) + water(1) -- rainwater harvesting on both sides.
    "agua":              {"key": "water",        "es": "Agua",                    "en": "Water"},
    "water":             {"key": "water",        "es": "Agua",                    "en": "Water"},
    # crops(3: potato/maize/seed) + cultivos(1: alfalfa) -- both field crops.
    "crops":             {"key": "crops",        "es": "Cultivos",                "en": "Field crops"},
    "cultivos":          {"key": "crops",        "es": "Cultivos",                "en": "Field crops"},
    # (D) rotacion(1) folded in: one English document under a Spanish label, and a lone chip
    # promising a single document is a thin offer next to "Cultivos".
    "rotacion":          {"key": "crops",        "es": "Cultivos",                "en": "Field crops"},

    # --- merged families (judgment calls A and B, applied) -----------------
    # (A) plagas(1) + disease(1) + control-biologico(2). One family to a reader thinking
    # "something is eating my crop" -- and the `plagas` document is itself half about natural
    # enemies, which is exactly what control-biologico covers. Three single-ish chips became
    # one chip with four documents behind it.
    "plagas":            {"key": "pests_disease", "es": "Plagas y enfermedades",  "en": "Pests & disease"},
    "disease":           {"key": "pests_disease", "es": "Plagas y enfermedades",  "en": "Pests & disease"},
    "control-biologico": {"key": "pests_disease", "es": "Plagas y enfermedades",  "en": "Pests & disease"},
    # (B) suelos(5) + compost(1) + nutrition(2). One soil-fertility family; merging makes it
    # the largest topic in the servable corpus rather than a big ES cluster with two orphans.
    "suelos":            {"key": "soil_fertility", "es": "Suelos y fertilidad",   "en": "Soil & fertility"},
    "compost":           {"key": "soil_fertility", "es": "Suelos y fertilidad",   "en": "Soil & fertility"},
    "nutrition":         {"key": "soil_fertility", "es": "Suelos y fertilidad",   "en": "Soil & fertility"},

    # --- distinct subjects, localized only --------------------------------
    "permaculture":      {"key": "permaculture", "es": "Permacultura",            "en": "Permaculture"},
    "legal":             {"key": "legal",        "es": "Legislación agrícola",    "en": "Agricultural law"},
    "fruticultura":      {"key": "fruit",        "es": "Fruticultura",            "en": "Fruit growing"},
    "organic":           {"key": "organic",      "es": "Agricultura orgánica",    "en": "Organic farming"},
    "biodynamics":       {"key": "biodynamics",  "es": "Biodinámica",             "en": "Biodynamics"},
    "livestock":         {"key": "livestock",    "es": "Ganadería",               "en": "Livestock"},
    "economics":         {"key": "economics",    "es": "Economía agrícola",       "en": "Farm economics"},
    "agroecologia":      {"key": "agroecology",  "es": "Agroecología",            "en": "Agroecology"},
    "agroforestry":      {"key": "agroforestry", "es": "Agroforestería",          "en": "Agroforestry"},
    "aquaculture":       {"key": "aquaculture",  "es": "Acuicultura",             "en": "Aquaculture"},
    "mushrooms":         {"key": "mushrooms",    "es": "Hongos comestibles",      "en": "Mushrooms"},
    "agroclimatology":   {"key": "agroclimate",  "es": "Agroclimatología",        "en": "Agroclimatology"},
    "natural-farming":   {"key": "natural",      "es": "Agricultura natural",     "en": "Natural farming"},

    # --- never surfaced ----------------------------------------------------
    # The planted off-domain distractor. Not a subject this corpus answers from.
    "chemistry":         None,
}

# ---------------------------------------------------------------------------
# THE JUDGMENT CALLS, AS DECIDED
#
# What settled A, B and D was rendering the list rather than arguing about taxonomy.
# The topic list has exactly one job: it appears after someone has struck out, and it
# has to give them a productive next query. As originally drafted it ran to 23 chips,
# 7 of which led to a single document each. Nobody scans 23 chips. Merging is what
# makes the list usable at all, so the tie-breaker was always "which list would I
# rather be handed at that moment", not "which taxonomy is more correct".
#
# A. MERGED -- pests + disease + control-biologico -> `pests_disease`. Biological
#    control is a method a reader might seek by name, which is the argument for
#    keeping it separate; it lost because all three are one family to someone whose
#    crop is being eaten, and because the `plagas` document is itself substantially
#    about natural enemies. Four documents behind one chip beats three thin chips.
#
# B. MERGED -- suelos + compost + nutrition -> `soil_fertility`. This makes it the
#    largest topic in the servable corpus (8 docs). The worry was that `suelos`, the
#    biggest Spanish cluster, would be hidden behind an umbrella; in practice the
#    umbrella *surfaces* it, since the merged topic now leads the ranking.
#
# C. NOT MERGED -- organic / biodynamics / natural-farming / permaculture /
#    agroecology stay distinct. They are different schools, not synonyms, and
#    someone asking about biodynamics is not served by permaculture material.
#    Nothing in the data argued for collapsing them.
#
# D. MERGED -- rotacion -> `crops`. A single English document under a Spanish label.
#    Rotation planning is asked by name, which is the case for keeping it; it lost
#    because a chip promising exactly one document is a thin offer.
#
# All four are reversible in one line, with no re-index and no migration.
#
# The call that matters more than any of these: topics are ranked by RELEVANCE TO
# THE FAILED QUERY, not by document count (see agroteca.suggest.rank_topics).
# Doc-count ranking would answer a question about aphids with "Agricultural law",
# purely because law owns the most PDFs. That decision is what makes the merges
# above low-stakes.
# ---------------------------------------------------------------------------


def canonical(raw: str | None) -> dict | None:
    """Raw manifest topic -> {key, es, en}, or None if it must not be shown.

    An unknown value returns None rather than guessing: a topic added to a manifest
    without being added here is invisible to suggestions until someone labels it,
    which fails closed instead of leaking a raw slug into the UI.
    """
    if not raw:
        return None
    return CANONICAL.get(raw.strip().lower())


def label(raw: str | None, lang: str) -> str | None:
    """Localized display label for a raw topic, or None if it must not be shown."""
    c = canonical(raw)
    if c is None:
        return None
    return c.get(lang) or c.get("en")
