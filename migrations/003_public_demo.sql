-- Public demo (profile=hosted): quota guards and feedback storage.
-- The free tiers behind the demo (Jina rerank tokens, Groq requests) must not be burnable by a
-- bot, so every question is counted before it is answered: per UTC day across all visitors, and
-- per hashed IP per hour. Raw IP addresses are never stored -- only a salted SHA-256.
-- Safe to re-run (IF NOT EXISTS). Not needed for the local profile.
--
--   psql "$AGROTECA_DB_URL" -f migrations/003_public_demo.sql

CREATE TABLE IF NOT EXISTS usage_daily (
    day       DATE PRIMARY KEY,
    requests  INT  NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_ip_hourly (
    ip_hash   TEXT        NOT NULL,
    hour      TIMESTAMPTZ NOT NULL,
    requests  INT         NOT NULL DEFAULT 0,
    PRIMARY KEY (ip_hash, hour)
);

-- The serverless filesystem is read-only, so thumbs-up/down votes live here instead of
-- feedback.jsonl when hosted.
CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL   PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    vote            TEXT        NOT NULL,
    question        TEXT        NOT NULL,
    answer_preview  TEXT        NOT NULL DEFAULT '',
    -- Retrieval context for abstentions (same fields as the local feedback.jsonl record).
    abstained       BOOLEAN     NOT NULL DEFAULT false,
    kind            TEXT,
    top_score       DOUBLE PRECISION,
    translated_query TEXT,
    retrieved       JSONB       NOT NULL DEFAULT '[]'::jsonb
);
