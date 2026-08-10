"""Central configuration. Every knob lives here so experiments change one place.

Values can be overridden with environment variables prefixed `AGROTECA_`
(e.g. `AGROTECA_CHUNK_TOKENS=256`) or a local `.env` file.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/agroteca/config.py -> parents[0]=agroteca, [1]=src, [2]=repo root
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGROTECA_", env_file=".env", extra="ignore")

    # --- paths ---
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    synthetic_dir: Path = ROOT / "data" / "synthetic"
    data_dir: Path = ROOT / "data"

    # --- database ---
    # host port 5433 to avoid colliding with an existing Postgres (e.g. Langfuse) on 5432
    db_url: str = "postgresql://postgres:agroteca@localhost:5433/agroteca"

    # --- embedding model ---
    # Baseline: MiniLM-L12 multilingual (384-dim). MEASURED ~50x faster on CPU than
    # e5-large (eval/bench_embed.py: 20.6 vs 0.4 chunks/sec) — an ~8-min full re-index
    # vs ~7 hours, and 19 vs 153 ms per query. Quality upgrade path (documented):
    # intfloat/multilingual-e5-large (1024-dim) or BGE-M3 — a config change + re-index.
    # NOTE: embed_dim MUST match the VECTOR(n) column in migrations/001_init.sql.
    # MiniLM does NOT need E5's "query:"/"passage:" prefixes.
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384
    batch_size: int = 64

    # --- reranker (cross-encoder, Phase 4) ---
    # multilingual cross-encoder; only runs on the small hybrid candidate pool.
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    rerank_candidates: int = 20   # how many hybrid results to re-score

    # --- chunking ---
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    chars_per_token: int = 4            # rough heuristic for char-based splitting

    # --- which tiers to ingest ---
    local_mode: bool = False           # False = deploy (open+synthetic); True adds copyrighted 'local'
    include_distractor: bool = False   # only for eval runs

    @property
    def chunk_chars(self) -> int:
        return self.chunk_tokens * self.chars_per_token

    @property
    def overlap_chars(self) -> int:
        return self.chunk_overlap * self.chars_per_token

    def tiers(self) -> list[str]:
        tiers = ["open", "synthetic"]
        if self.local_mode:
            tiers.append("local")
        if self.include_distractor:
            tiers.append("distractor")
        return tiers


settings = Settings()
