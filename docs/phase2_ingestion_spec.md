# Phase 2 — Ingestion Pipeline: Specification + Teaching Guide

This document is two things at once: a **build specification** you implement by hand, and a
**course** in how retrieval systems actually work. Read a section, understand *why*, then build
that piece and verify it before moving on. By the end you will have turned a folder of PDFs into
a searchable index, and you will be able to defend every decision to a senior engineer.

No working pipeline code is written for you here on purpose — you learn by building. What you get
instead is the *design*: the data shapes, the decisions with their trade-offs, the exact schema
and commands, and small illustrative snippets marked **(reference)**.

---

## 0. The mental model: what "ingestion" is and why it decides everything

A RAG system has two halves that run at different times:


- **Offline / indexing (this phase):** once, ahead of time, you read every document, cut it into
  small pieces, turn each piece into a form you can search by *meaning*, and store it. Think of it
  as building the index at the back of a book — but for a whole library, and searchable by idea,
  not just by exact word.
- **Online / querying (later phases):** every time a user asks a question, you search that index,
  pull the few most relevant pieces, and hand them to an LLM to write a grounded, cited answer.

Here is the single most important sentence in this whole project:

> **Retrieval is the bottleneck, not generation.** If the wrong pieces come back from the index,
> even the best LLM in the world will write a confident, wrong answer. So the *ceiling* on your
> system's quality is set here, in Phase 2. Generation can only work with what retrieval gives it.

That is why we spend real care on ingestion, and why "chunking is where RAG silently fails" is a
top-1% interview line. Most people rush this and never understand why their answers are mediocre.

**What Phase 2 produces:** a Postgres database containing many small, richly-labeled text *chunks*,
each with a vector that encodes its meaning and a keyword index for exact matches. Nothing answers
questions yet — but everything that answers questions later reads from here.

---

## 1. The unit of everything: the *chunk*

We do not store whole documents as retrievable units. Three reasons:

1. **Context limits.** You can only fit so much text into an LLM's prompt. Retrieving five focused
   paragraphs beats retrieving five 300-page books.
2. **Precision.** A question about "the width of a market-garden bed" should retrieve the one
   paragraph that answers it, not a whole book where that fact is drowned out.
3. **Citation.** To cite "*Reglamento 977, Art. 143*" you need to have stored text *at that
   granularity*, with the article/page recorded.

A **chunk** is a small, self-contained span of text (a few hundred words) plus metadata describing
where it came from. This is your core data structure. Design it deliberately:

### The chunk schema (the data contract)

| Field | Type | Why it exists |
|---|---|---|
| `chunk_id` | text (uuid or `docid#000`) | unique handle; lets the eval + citations point at exactly one chunk |
| `doc_id` | text | which document this came from (foreign key to a `documents` row) |
| `chunk_index` | int | position within the document (0,1,2,…) — for ordering and context expansion |
| `text` | text | the chunk's actual content (post-normalization) |
| `embedding` | vector(1024) | the meaning vector (Section 4) |
| `source_file` | text | filename, e.g. `Libro INIA N° 12.pdf` — for citation |
| `tier` | text | `open` / `local` / `synthetic` / `distractor` — controls what loads when (Section 6) |
| `lang` | text | `es`/`en` — for language-aware search and analysis |
| `topic` | text | from the manifest — enables metadata filtering |
| `page` | int (nullable) | page/locator — for citation |
| `char_start`,`char_end` | int | offsets in the source text — for debugging + provenance |

> **Teaching note — metadata is not decoration.** Half of what makes a RAG *good* is filtering on
> metadata: "only search the `open`+`synthetic` tiers when deployed," "prefer Spanish chunks for a
> Spanish query," "cite the page." A chunk without metadata is a chunk you can't govern, route, or
> cite. Your `DATA_GOVERNANCE.md` tiers become real here as the `tier` column.

---

## 2. Extraction & normalization — the stage where RAG silently dies

Before you can chunk, you must get clean text out of each PDF. You already discovered in Phase 1
that this is treacherous. Here is the disciplined version.

**Extractor: use PyMuPDF (`pymupdf`/`fitz`), not pypdf.** PyMuPDF preserves layout and reading
order far better, which matters for tables and multi-column pages. (pypdf was fine for *sampling*
text in Phase 1; for the real pipeline, upgrade.)

**Normalization steps, in order** — each fixes a specific failure you have already seen:

1. **De-hyphenate line-wrap breaks.** Scanned/old PDFs split words as `ori-\nginarias`. If you
   don't rejoin them, the chunk contains `ori ginarias`, which no query will match. Rule: a
   hyphen followed by a newline followed by a lowercase letter → delete the hyphen+newline.
   *(This is exactly why three of your golden `must_contain` strings broke in Phase 1.)*
2. **Normalize whitespace.** Collapse runs of spaces/newlines; convert single newlines inside a
   paragraph to spaces, but keep blank lines (paragraph boundaries) — the chunker needs them.
3. **Strip running headers/footers/page numbers.** Repeated lines (the book title on every page,
   bare page numbers) add noise to every chunk. Detect lines that recur on many pages and drop them.
4. **Handle glyph-encoded digits.** "The Market Gardener" encodes `30` as `/three.o/zero.o`. Either
   map these glyph names back to digits, or accept that numeric facts from that file won't extract
   (you already routed that file to the local tier, so it's low-stakes).
5. **OCR scanned pages.** For image-only government docs (INIA N°11, N°19), run
   `ocrmypdf -l spa+eng` first to add a real text layer. **Never synthesize** what you can't read
   (see `DATA_GOVERNANCE.md`): OCR recovers the *actual* text; invention would fabricate it.

**Verify this stage:** re-run your Phase-1 stopword-density gate on the *extracted+normalized* text
of each doc. Real prose is ≥8% stopwords; if a doc scores near zero after extraction, it's still
image-only and needs OCR. Do not let a no-text doc reach the chunker.

> **Teaching note.** The reason this stage "silently" fails is that nothing crashes — you get text,
> it just quietly doesn't match queries. The only way to catch it is to *look at the extracted text*
> and to measure retrieval later. Build the habit of eyeballing 2–3 chunks per document.

---

## 3. Chunking — the highest-leverage decision in the whole project

This is where you earn the "I measure, I don't guess" credential. Chunking has one central tension:

- **Chunks too small** → each is precise, but loses surrounding context; a fact split from its
  subject becomes unretrievable ("*it* is 30 inches wide" — *what* is?).
- **Chunks too large** → rich context, but the embedding (Section 4) must average the meaning of
  *everything* in the chunk, so a single fact's signal gets diluted and buried. Retrieval gets fuzzy.

The art is the middle. Here is the decision, with the alternatives so you understand the choice.

### Strategies (pick recursive as the baseline)

| Strategy | What it does | When |
|---|---|---|
| **Fixed-size** | cut every N characters/tokens, ignore structure | never (splits mid-word/sentence) |
| **Recursive character** ← **use this** | split on the biggest natural boundary that fits: paragraphs → lines → sentences → words | **default; robust, simple, good** |
| **Semantic** | embed sentences, cut where meaning shifts | later refinement; slower, more complex |
| **Structural** | split on the document's own headings/articles | **use for the legal docs** — chunk the RSA by *Artículo*, so you can cite "Art. 143" |

**Baseline parameters:** ~**512 tokens per chunk, ~64 tokens overlap.** Don't agonize — set this,
measure it, then adjust.

- **What's a token?** The unit an LLM/embedder reads — roughly ¾ of a word; ~4 characters of English
  (Spanish similar). So 512 tokens ≈ 350–400 words ≈ a healthy paragraph or two. You count tokens
  with the model's tokenizer, not by hand.
- **Why overlap?** So a fact sitting on a chunk boundary isn't cut in half. The last ~64 tokens of
  chunk *n* are repeated as the first ~64 of chunk *n+1*, so any sentence survives intact in at
  least one chunk.
- **Respect boundaries.** Prefer to break at paragraph/sentence ends (that's what "recursive" buys
  you). Never split mid-word. Carry `page` into each chunk's metadata as you go.

### The experiment you MUST run (this is the portfolio gold)

Do not pick 512 on faith. Chunk the corpus three ways — **256 / 512 / 1024 tokens** — and measure
each against your golden set (Section 7 shows how, with a lightweight retrieval check). Then write
the sentence every hiring manager wants to see:

> *"I tested 256/512/1024-token chunks; 512 gave the best retrieval@5 on the golden set, so I use 512."*

That one measured sentence separates you from everyone who guessed.

### Vanguard upgrade — Contextual Retrieval (do after the baseline works)

Chunks lose their context: a chunk reading "*it is applied to the surface*" doesn't say *what*.
**Contextual Retrieval** (an Anthropic technique) prepends a one-sentence, LLM-generated situating
line to each chunk *before embedding* — e.g. "*This passage from* El Nogal en Chile *describes
walnut cultivars offered in 1928:*". It measurably reduces failed retrievals and is a strong
talking point. Add it as a second pass once the plain pipeline is measured.

---

## 4. Embeddings — turning text into searchable *meaning*

### What an embedding actually is

An **embedding** is a list of numbers (a **vector**) that represents a piece of text's *meaning* as
a point in a high-dimensional space. The model is trained so that **texts with similar meaning land
near each other**, even when they share no words. "*perro*" and "*dog*" and "*canine companion*"
end up close together; "*tractor*" ends up far away. This is what lets you search by idea instead
of by exact keyword.

- **Dimensionality:** BGE-M3 outputs a **1024-number** vector per text. That's the `vector(1024)`
  column in your schema.
- **How "nearness" is measured — cosine similarity.** Ignore the vectors' lengths; compare their
  *direction*. Cosine similarity of 1.0 = same direction (same meaning), 0 = unrelated, −1 =
  opposite. In pgvector you'll use the cosine-distance operator `<=>` (distance = 1 − similarity),
  so *smaller is more similar*.

### Why THIS project forces a specific choice: multilingual

Your golden set deliberately contains **Spanish questions whose answers live in English documents**
(and vice-versa). A monolingual English embedder (e.g. `nomic-embed-text`) would place the Spanish
query and the English answer far apart and **fail every cross-lingual question**. You must use a
**multilingual** model that maps both languages into the *same* meaning space.

**Use BGE-M3** (`BAAI/bge-m3`). It is multilingual, strong, free to run locally, and — importantly
for Phase 3 — it emits **both**:
- a **dense** vector (semantic meaning, Section 4), and
- a **sparse** vector (learned keyword weights, basically a smart BM25).

That means your *hybrid* search in Phase 3 can come from **one model**, which is elegant and current.

> **Teaching note — dense vs sparse, the preview of hybrid.** Dense/semantic search is great at
> "meaning" but can *miss exact strings* — ask for variety code `WL-323` and a semantic model may
> return "alfalfa varieties in general." Sparse/lexical search nails exact tokens like `WL-323` but
> is blind to paraphrase. You need both. That's Phase 3. You set it up *here* by storing a dense
> vector **and** a keyword index for every chunk.

### Practical notes

- Run BGE-M3 locally via `FlagEmbedding` (`BGEM3FlagModel`) or `sentence-transformers`. Zero cost,
  consistent with the project's local-first philosophy.
- **Batch** your encoding (e.g. 32–64 chunks at a time) — encoding one-by-one is needlessly slow.
- **Normalize** embeddings to unit length (BGE-M3 does by default) so cosine behaves cleanly.
- Encode the *chunk text* (plus the contextual prefix if you added Section 3's upgrade).

---

## 5. The vector store — Postgres + pgvector

### Why a real database (not a pickle file or a bare FAISS index)

You *could* dump vectors into a file. Don't. A database gives you, in one place:

- **Metadata filtering** in the same query as the vector search ("nearest chunks *where tier in
  ('open','synthetic')*") — essential for your deploy-vs-local governance.
- **Keyword search** (Postgres full-text) for the sparse half of hybrid — no second system.
- **Persistence, transactions, SQL** you already understand, and a setup that reads as *production*.

**pgvector** is a Postgres extension that adds a `vector` column type and nearest-neighbor search.

### Stand it up with Docker (one command)

Create `docker-compose.yml` (reference):

```yaml
# (reference)
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: agroteca
      POSTGRES_DB: agroteca
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes: { pgdata: {} }
```
`docker compose up -d` and you have Postgres+pgvector on `localhost:5432`.

### Schema (DDL — this is the spec; you run it)

```sql
-- (reference) run once, e.g. in a migrations/001_init.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  doc_id      TEXT PRIMARY KEY,
  source_file TEXT NOT NULL,
  title       TEXT,
  tier        TEXT NOT NULL,      -- open | local | synthetic | distractor
  lang        TEXT,
  topic       TEXT,
  url         TEXT
);

CREATE TABLE chunks (
  chunk_id    TEXT PRIMARY KEY,
  doc_id      TEXT REFERENCES documents(doc_id),
  chunk_index INT  NOT NULL,
  text        TEXT NOT NULL,
  embedding   VECTOR(1024),       -- BGE-M3 dense
  tsv         TSVECTOR,           -- keyword index (sparse half of hybrid, Phase 3)
  page        INT,
  char_start  INT,
  char_end    INT
);

-- Approximate-nearest-neighbor index for fast cosine search:
CREATE INDEX chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword index (Spanish + English handled per-row at insert time):
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
```

> **Teaching note — what's an ANN index (HNSW)?** With tens of thousands of chunks, comparing a
> query vector to *every* chunk is slow. An **Approximate Nearest Neighbor** index (HNSW = a
> navigable graph of vectors) finds the closest ones in near-constant time by "walking" the graph,
> trading a tiny bit of accuracy for a huge speed gain. `IVFFlat` is the alternative (buckets); HNSW
> usually wins on recall/latency and needs no training step — use **HNSW**.

> **Teaching note — the `tsv` column.** `TSVECTOR` is Postgres's full-text representation. At insert,
> set it with `to_tsvector('spanish', text)` for ES chunks and `'english'` for EN chunks. This is
> what powers keyword search in Phase 3 (Postgres `ts_rank` approximates BM25; if you want *true*
> BM25 later, the `pg_search`/ParadeDB extension or BGE-M3's sparse vectors are the upgrade path).

---

## 6. The ingestion orchestrator — putting it together

One script, `src/agroteca/ingest/run.py` (you build it), that ties the stages into a pipeline.

**Flow (per run):**
1. **Decide which tiers to load** from `LOCAL_MODE` (see below). Read the matching manifest CSVs.
2. **For each document row:** locate the file (`data/raw/` for pdf tiers, `data/synthetic/` for the
   synthetic tier) → **extract** (§2) → **normalize** (§2) → **gate** (skip/flag if no text) →
   **chunk** (§3) → **embed** the chunks in batches (§4) → **insert** a `documents` row and its
   `chunks` rows (§5).
3. **Report:** documents ingested, chunks created, chunks/doc, and any skipped (no-text) files.

**Configuration in one place** — `src/agroteca/config.py`, a pydantic `Settings` object:
`db_url`, `embed_model="BAAI/bge-m3"`, `chunk_tokens=512`, `chunk_overlap=64`, `local_mode: bool`,
`batch_size=48`. Every knob you'll sweep lives here, nowhere else.

**`LOCAL_MODE` — the governance switch made real:**

| Mode | Tiers ingested | Why |
|---|---|---|
| deploy (`LOCAL_MODE=0`) | `open` + `synthetic` | only legally-deployable sources reach a public demo |
| local (`LOCAL_MODE=1`) | `open` + `synthetic` + `local` | on your farm machine you get the copyrighted books too |
| eval | above **+** `distractor` | distractors are loaded only to measure precision/abstention |

> This is where `DATA_GOVERNANCE.md` stops being a document and becomes running behavior. In an
> interview: "*the deploy build literally cannot ingest the copyrighted tier — it's a config gate,
> not a promise.*"

**Idempotency.** Make re-runs safe: compute a hash of each chunk's `(doc_id, chunk_index, text)`
and upsert, or wipe-and-reload per document. You will re-ingest many times while tuning chunk size;
a pipeline you're afraid to re-run is a broken pipeline.

---

## 7. Verify it works — the ship criterion (and the bridge to your eval)

**Smoke test (does anything come back?):** embed the string "*variedad de alfalfa WL-323*", run a
cosine search, print the top-5 chunks. You should see the INIA alfalfa chunk. (reference:)

```sql
-- (reference) :qvec is your query embedding as a vector literal
SELECT chunk_id, source_file, left(text, 120)
FROM chunks
WHERE tier IN ('open','synthetic')
ORDER BY embedding <=> :qvec      -- cosine distance, smaller = closer
LIMIT 5;
```

**The real verification — a retrieval check against your golden set.** This is the seed of your
Phase-1 eval runner, and it lets you run the chunk-size experiment *now*:

- For each **answerable** golden question, embed the `question`, retrieve top-k chunks, and check
  whether any retrieved chunk **contains that question's `must_contain` string**.
- The fraction that pass = **retrieval@k** (a real, honest metric). No LLM or RAGAS needed yet —
  it's deterministic and free, exactly the kind of check you should run in CI.
- Run it for the 256/512/1024 indexes → you now have the numbers that justify your chunk size.

**Ship criterion checklist for Phase 2:**
- [ ] `docker compose up` gives Postgres+pgvector.
- [ ] Every non-image manifest doc is extracted, normalized, chunked, embedded, and stored.
- [ ] No-text docs are detected and skipped/flagged (not silently stored empty).
- [ ] The smoke query returns sensible chunks.
- [ ] `retrieval@5` is computed on the golden set for at least the 512-token index (baseline number).
- [ ] `LOCAL_MODE` correctly changes which tiers are ingested.

When those are checked, Phase 2 is done and you have your **first real number** — the baseline your
whole blog post ("retrieval@5 went from X to Y") will build on.

---

## 8. Pitfalls & how to debug them

| Symptom | Likely cause | Fix |
|---|---|---|
| A doc produces 0 chunks | no text layer (image-only) | OCR it (§2) or exclude it |
| Chunks contain `word- word` fragments | de-hyphenation skipped | fix normalization (§2.1) |
| Cross-lingual questions never retrieve | monolingual embedder | confirm you're on **BGE-M3**, not an English model |
| Exact codes (`WL-323`) don't retrieve | that's expected for dense-only | it's what Phase 3 (hybrid) fixes — don't "fix" it here |
| Search is slow | no ANN index, or querying before `ANALYZE` | create the HNSW index; `ANALYZE chunks;` |
| Nonsense top results | embedding the wrong text, or dimension mismatch | assert vector length == 1024; log the exact text you embed |
| Re-run duplicates everything | no idempotency | upsert by hash or wipe-per-doc (§6) |

> **Debugging discipline:** when retrieval is bad, *print the chunks that came back*. 90% of RAG
> bugs are visible the moment you look at the retrieved text — it's hyphenated, it's a header/footer,
> it's the wrong language, or it's empty. Look before you theorize.

---

## 9. Build checklist — files, dependencies, commands

**Add dependencies** (via `uv add`): `pymupdf`, `sentence-transformers` (or `FlagEmbedding`),
`psycopg[binary]`, `pgvector`, `pydantic-settings`, and (optional, for a ready-made splitter)
`langchain-text-splitters`. Keep `ocrmypdf` as a CLI tool for the scanned docs.

**Files to create:**
```
docker-compose.yml
migrations/001_init.sql               # the DDL in §5
src/agroteca/config.py                # pydantic Settings
src/agroteca/ingest/extract.py        # PyMuPDF -> raw text per page
src/agroteca/ingest/normalize.py      # §2 steps (dehyphenate, whitespace, headers, glyphs)
src/agroteca/ingest/chunk.py          # recursive ~512/64; structural for legal docs
src/agroteca/ingest/embed.py          # BGE-M3, batched
src/agroteca/ingest/store.py          # insert documents + chunks (psycopg + pgvector)
src/agroteca/ingest/run.py            # orchestrator; reads manifests per LOCAL_MODE
eval/retrieval_at_k.py                # §7 golden-set retrieval check (baseline number)
```

**Run order:**
```
docker compose up -d
uv run psql ... -f migrations/001_init.sql      # or apply the DDL your way
uv run python -m agroteca.ingest.run            # ingest (deploy tiers)
LOCAL_MODE=1 uv run python -m agroteca.ingest.run
uv run python eval/retrieval_at_k.py            # your first real metric
```

Build one file at a time, in the order above, and run the smoke test after `store.py` works so you
see chunks land before you build the eval check. Small steps, each verified.

---

## 10. Glossary (keep this handy)

- **Chunk** — a small labeled span of a document; the unit you retrieve and cite.
- **Token** — the sub-word unit models read; ~4 characters / ~¾ of a word.
- **Embedding / vector** — a list of numbers encoding a text's meaning; similar meanings → nearby vectors.
- **Cosine similarity / distance** — compares vector *direction*; how "close in meaning" two texts are.
- **Dense vs sparse** — dense = semantic meaning (embeddings); sparse = keyword weights (BM25-like). Hybrid uses both.
- **ANN / HNSW** — approximate nearest-neighbor search; a fast index for finding the closest vectors.
- **pgvector** — Postgres extension adding a `vector` type + similarity search.
- **tsvector / BM25** — Postgres full-text representation / the classic keyword-relevance ranking.
- **Overlap** — shared tokens between adjacent chunks so boundary-spanning facts survive.
- **retrieval@k** — fraction of questions for which the right chunk appears in the top *k* results.
- **Contextual Retrieval** — prepending an LLM-written situating sentence to each chunk before embedding, to preserve context.

---

### How this phase connects to the story
Phase 1 built the ruler (the golden set). Phase 2 builds the thing being measured *and* takes the
first measurement (retrieval@5 on 512-token chunks). Phase 3 (hybrid + RRF) will move that number,
and you'll have proof. Everything from here is: change one thing, re-measure, keep what wins.
Next spec: **Phase 3 — Hybrid Retrieval & Reranking**, when you've shipped this.
