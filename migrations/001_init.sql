-- Agroteca schema. Apply once:
--   docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,     -- stable id derived from the filename
    source_file TEXT NOT NULL,        -- for citation
    title       TEXT,
    tier        TEXT NOT NULL,        -- open | local | synthetic | distractor
    lang        TEXT,
    topic       TEXT,
    url         TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,     -- e.g. "<doc_id>#0007"
    doc_id      TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    embedding   VECTOR(384),          -- must match settings.embed_dim (MiniLM-L12 = 384)
    tsv         TSVECTOR,             -- keyword index (Phase 3 hybrid / BM25-style)
    -- denormalized metadata so retrieval can filter without a join:
    tier        TEXT,
    lang        TEXT,
    topic       TEXT,
    page        INT,
    char_start  INT,
    char_end    INT
);

-- Approximate-nearest-neighbor index for fast cosine search:
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword index for the lexical half of hybrid search:
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Tier filtering is on the hot path (deploy vs local vs eval):
CREATE INDEX IF NOT EXISTS chunks_tier_idx ON chunks (tier);
