-- Phase 3: rebuild the keyword index with the language-agnostic 'simple' config.
-- 'simple' lowercases and tokenizes but does NOT stem or drop stopwords, so lexical
-- search matches exact tokens (variety codes, species, numbers) consistently across
-- Spanish and English. This is an in-place UPDATE — it never touches the embeddings.
--
--   docker exec -i agroteca-pgvector psql -U postgres -d agroteca < migrations/002_tsv_simple.sql

UPDATE chunks SET tsv = to_tsvector('simple', text);
