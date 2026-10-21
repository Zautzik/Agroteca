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
    # Pinned: this model loads with trust_remote_code=True (it ships a custom architecture),
    # which executes Python from the HF repo. Pinning the commit means a later push to that repo
    # can't silently change what runs here -- bump deliberately, after reviewing the diff.
    rerank_model_revision: str = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
    rerank_candidates: int = 20   # how many hybrid results to re-score

    # --- deployment profile ---
    # "local": the measured reference stack -- sentence-transformers + the torch cross-encoder on
    # `device`, generation through Ollama. "hosted": the $0 public demo -- the SAME MiniLM model run
    # through ONNX Runtime (no torch in the bundle), the SAME jina-reranker-v2 served by Jina's API,
    # and generation through an OpenAI-compatible endpoint (Groq). Same models on purpose: the
    # numbers in eval/results.csv stay comparable, and a hosted eval row is measured, not assumed.
    profile: str = "local"        # "local" | "hosted"

    # hosted: query embedding (ONNX). The model file is fetched once per cold instance from the
    # embed model's own Hugging Face repo, pinned to a revision, into a writable cache dir.
    onnx_model_file: str = "onnx/model.onnx"   # path inside the HF repo; a smaller variant is a
                                               # deliberate swap only after eval/parity_check.py passes
    onnx_model_revision: str = "main"          # pin to a commit sha for a reproducible deploy
    onnx_cache_dir: Path = Path("/tmp/agroteca-onnx")
    embed_max_tokens: int = 128                # MiniLM's max_seq_length -- must match the index

    # hosted: rerank via Jina's API (same model as `rerank_model`).
    jina_api_key: str = ""
    jina_rerank_url: str = "https://api.jina.ai/v1/rerank"
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    jina_timeout: float = 30.0
    # Jina's free key also caps tokens per minute. On a 429/503 the call waits (Retry-After, else
    # 2s, 4s) and retries this many times before giving up with a friendly "busy" error.
    jina_max_retries: int = 2

    # hosted: generation via an OpenAI-compatible chat-completions API (Groq by default).
    gen_api_key: str = ""
    openai_base_url: str = "https://api.groq.com/openai/v1"
    # Groq retired llama-3.3-70b-versatile (Aug 2026) and recommends gpt-oss-120b. It is a
    # reasoning model: its hidden reasoning counts against max_tokens, so effort is kept low.
    # Set openai_reasoning_effort="" for a non-reasoning model (the field would be rejected).
    openai_model: str = "openai/gpt-oss-120b"
    openai_reasoning_effort: str = "low"

    # hosted: serverless-friendly DB pool, and protection for the free-tier quotas.
    db_pool_min: int = 5
    db_prepare_threshold: int = 5          # psycopg's default; hosted turns it off (see below)
    daily_request_cap: int = 0             # 0 = no cap (local). Hosted: questions per UTC day, all users
    per_ip_hourly_cap: int = 0             # 0 = no cap (local). Hosted: questions per IP per hour
    ip_hash_salt: str = ""                 # hosted: secret salt so stored IP hashes can't be reversed
    cache_enabled: bool | None = None      # None = on when hosted (saves Jina/Groq quota), off locally

    @property
    def hosted(self) -> bool:
        return self.profile == "hosted"

    @property
    def prepare_threshold(self) -> int | None:
        """None disables server-side prepared statements, which a transaction-mode pooler
        (PgBouncer, e.g. Neon's pooled endpoint used by the hosted profile) can't track."""
        return None if self.hosted else self.db_prepare_threshold

    # --- inference device (embedder + reranker; the generation LLM runs in Ollama, separately) ---
    # "cuda" by default: this is a GPU-serving deployment, and warmup.load_and_warm() refuses to
    # silently fall back to CPU if AGROTECA_DEVICE=cuda but no GPU is visible (fail loud, not slow).
    # "cpu" forces CPU explicitly (local iteration on a GPU-less machine). "auto" resolves to
    # cuda-if-available-else-cpu at warmup time -- an explicit opt OUT of the fail-loud guarantee,
    # for "same code, $0 locally, real GPU in prod" workflows that don't need that guarantee.
    device: str = "cuda"
    warmup_batch_size: int = 8   # dummy batch size for the startup warmup forward pass

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    # --- generation (LLM) ---
    # Local Ollama by default; deploy swaps to a hosted/smaller model by setting
    # AGROTECA_GEN_MODEL / AGROTECA_GEN_BASE_URL — no source edits (7 min on CPU is unservable).
    gen_model: str = "qwen2.5:3b"
    gen_base_url: str = "http://localhost:11434"   # Ollama host
    # read timeout (s), i.e. max gap between streamed chunks -- not a total-request timeout.
    # 300s was calibrated against the OLD ONNX-backed retrieval story; it's too short now that
    # CPU prefill on the ~2800-token RAG prompt alone can exceed it (measured: a real generation
    # request hit this exact timeout mid-stream on CPU, aborting a request that was working, not
    # hung). 1800s matches the CPU-tier costs measured elsewhere in this pipeline post-GPU-pivot.
    # A hosted/GPU model can lower this back down -- it's a config knob, not a code change.
    gen_timeout: float = 1800.0
    gen_num_predict: int = 1024                    # cap output tokens (runaway-generation guard)

    # --- query translation (serving-side cross-lingual lever) ---
    # Retrieve in a BILINGUAL query space: translate the question to the other language and rerank
    # the union pool by max cross-encoder score. MEASURED to recover a cross-lingual miss (q08,
    # rank None->1) that no embedder swap could -- answer@5 0.75->0.80, zero regressions
    # (eval/query_translation.py). Fails soft to original-only retrieval, so it can never do worse
    # than the baseline. Costs one LLM translation call per query (the app already runs an LLM).
    translate_queries: bool = True
    translate_timeout: float = 60.0
    # Output cap for the translation call. A plain model needs ~40 tokens; a reasoning model
    # spends part of the budget on reasoning before the answer, so the cap leaves room for that.
    translate_max_tokens: int = 512

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
    chars_per_token: int = 4            # rough, English-biased heuristic for char-based splitting;
                                        # ES prose and cultivar codes tokenize differently, so a
                                        # "512-token" chunk may be ~400-600 real tokens. Spot-check
                                        # against a new embedder's max_seq_length before trusting it.

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
