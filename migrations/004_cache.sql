-- Public demo (profile=hosted): a small key/value cache so repeated questions don't spend the
-- one-time Jina rerank allowance again (see agroteca.cache). Entries are rerank scores and query
-- translations, keyed by a SHA-256 of their exact inputs; the corpus is static, so they never
-- go stale. Safe to re-run. Not needed for the local profile.
--
--   uv run python scripts/apply_migrations.py 004_cache

CREATE TABLE IF NOT EXISTS kv_cache (
    key         TEXT        PRIMARY KEY,
    value       JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    hits        INT         NOT NULL DEFAULT 0
);
