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
    db_pool_max: int = 8   # served-API pool ceiling (each streamed request holds a conn only for retrieval)

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
    ort_threads: int | None = None  # ONNX intra-op threads for embed+rerank; None = all cores, lower to bound concurrency

    # --- generation (LLM) ---
    # Local Ollama by default; deploy swaps to a hosted/smaller model by setting
    # AGROTECA_GEN_MODEL / AGROTECA_GEN_BASE_URL — no source edits (7 min on CPU is unservable).
    gen_model: str = "qwen2.5:3b"
    gen_base_url: str = "http://localhost:11434"   # Ollama host
    gen_timeout: float = 300.0                     # read timeout (s): generous for slow CPU first-token; a hosted model can lower it
    gen_num_predict: int = 1024                    # cap output tokens (runaway-generation guard)

    # --- query translation (serving-side cross-lingual lever) ---
    # Retrieve in a BILINGUAL query space: translate the question to the other language and rerank
    # the union pool by max cross-encoder score. MEASURED to recover a cross-lingual miss (q08,
    # rank None->1) that no embedder swap could -- answer@5 0.75->0.80, zero regressions
    # (eval/query_translation.py). Fails soft to original-only retrieval, so it can never do worse
    # than the baseline. Costs one LLM translation call per query (the app already runs an LLM).
    translate_queries: bool = True
    translate_timeout: float = 60.0

    # --- chunking ---
    # KNOWN CEILING: the MiniLM embedder truncates at ~128 tokens (max_seq_length), so the
    # dense vector encodes only the first ~128 tokens of each 512-token chunk -- the tail is
    # present for lexical FTS and the reranker's text, but not in the semantic vector.
    # Confirmed with a cosine test (chunks sharing their first ~128 tokens but with different
    # tails embed identically, cosine 1.0). Measured blast radius: ~half the golden answers
    # (10/19) have their answer text PAST this window, and 4 of those are the reranker's
    # remaining misses -- so this is a root cause of the recall gap, not a curiosity.
    # A documented cap on dense recall; the fix is a
    # longer-context embedder (e5-large 512 / BGE-M3 8192) or smaller chunks -- a measured
    # re-index experiment, not a hot patch, since it re-runs the whole eval.
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
