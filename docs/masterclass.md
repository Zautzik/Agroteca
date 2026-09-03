# Agroteca — Masterclass, Skills Inventory & Roadmap

> A living study document **and** a from-first-principles study guide: what I've built, what I've
> *learned*, where the project stands, and how to explain it in an interview (Parts 1–5) — plus the
> **full technical foundation underneath it**: the mathematics derived, the system design, and every
> library choice with its rejected alternatives (Parts 6–9). Every **one-breath answer** is phrased
> the way I'd say it out loud to an interviewer; every formula is one you can reproduce at a whiteboard.
>
> _Status snapshot: end-to-end complete and pushed — governed corpus → measured retrieval
> cascade → grounded/cited/abstaining generation → a **production-hardened**, **containerized**
> streaming service → a bilingual, **dual-theme** web app, over a **unit-tested, single-source-of-truth**
> eval. Beyond the build I ran the **embedder upgrade as a complete three-stage experiment** — a
> mechanistic pilot with a control, a cheap re-chunk probe, and a full e5-large re-index — and let it
> **disprove my own hypothesis**. Then I rebuilt serving on `psycopg_pool.AsyncConnectionPool` +
> FastAPI dependency injection, moved the embedder/reranker onto `torch`/CUDA with an explicit
> fail-fast warmup gate, and — the story worth telling — chased a 20× CPU slowdown into a **second
> falsified hypothesis** (thread oversubscription, not the real cause) before finding the true one by
> reading the reranker's own source instead of guessing. Then that work went in front of an actual
> person, who found two real bugs my own tests hadn't: `/ask/stream` sent the client zero bytes —
> not even HTTP headers — for the entire retrieval phase (fixed by moving retrieval inside the
> streaming generator so headers flush immediately, which also required moving generation onto
> `ollama.AsyncClient`), and a 300s read timeout calibrated before the CPU cost was known, now too
> short for real prefill time. Both closed, plus real, backend-driven progress feedback (not a
> fabricated percentage) replacing a purely decorative loading animation. Every claim below is
> backed by a committed artifact. Only the public URL — a hosting-cost decision, not a code
> problem — remains, and it now also requires a GPU host, a cost this document names rather than
> hides._

---

## Part 1 — What I've developed

Depth is marked honestly: 🟢 interview-solid · 🟡 emerging, needs a few more reps.

### 1A. Technical / hard skills

| Skill | What I can now do | Depth |
|---|---|---|
| **RAG architecture** | Explain the offline-index / online-query split and why "retrieval is the ceiling on answer quality" | 🟢 |
| **Chunking** | Chunk size vs overlap as a recall/precision knob — *and* the embedder's `max_seq_length` as a hard ceiling (I caught my 512-token chunks truncating to ~128 in the dense vector) | 🟢 |
| **Embeddings / bi-encoders** | Explain cosine similarity, multilingual models, why one vector per chunk enables ANN search | 🟢 |
| **Dense retrieval** | pgvector `<=>` cosine + HNSW; know its blind spot (misses exact codes like `WL-323`) | 🟢 |
| **Lexical retrieval** | Postgres FTS: `tsvector`/`tsquery`, `ts_rank_cd`, `'simple'` config, GIN index; blind spot (paraphrase) | 🟡 |
| **Hybrid + RRF** | Fuse two rankings with `1/(k+rank)`, k=60 — rank-fusion vs score-fusion, and *why* rank-fusion | 🟢 |
| **Cross-encoder reranking** | The cascade (fast recall → slow precision); bi- vs cross-encoder; run only on a small pool | 🟢 |
| **Grounded generation** | ground / cite / abstain; rules in the **system** role for injection resistance | 🟢 |
| **Eval-first methodology** | Golden set *before* system; retrieval@k dual metric; measure-then-decide; single-variable attribution | 🟢 |
| **SQL fundamentals** | SELECT/FROM/JOIN/WHERE/ORDER BY/LIMIT, aliases, parameterized queries, injection safety | 🟡 |
| **Python** | comprehensions, `*`-unpacking, `sorted(key=,reverse=)`, `lru_cache`, scope, regex, f-strings, `argparse`, `if __name__=="__main__"` | 🟡 |
| **Postgres / pgvector / Docker** | VECTOR(384), HNSW + GIN, container on host port 5433, volumes | 🟡 |
| **GPU inference (PyTorch/CUDA)** | `sentence-transformers`/`transformers` on `torch`, explicit `.to(device)`, `torch.inference_mode()`, a `trust_remote_code` model pinned to a reviewed commit hash | 🟢 |
| **Async Python / concurrency** | `asyncio`, `AsyncConnectionPool`, FastAPI `Depends`-based DI, `run_in_executor` for CPU-bound work off the event loop, a Windows event-loop-policy gotcha found by running it | 🟢 |
| **Data governance** | Four-tier corpus, idea/expression copyright line, citation integrity, `LOCAL_MODE` | 🟢 |
| **Git workflow** | Selective staging (never `git add .`), narrative commit messages | 🟢 |
| **Debugging discipline** | isolate-and-inspect with `repr()`; disk-vs-editor; `-u` buffering; pipe exit-code trap; falsifying my own performance hypothesis by reading library source before rewriting it | 🟢 |
| **API serving** | FastAPI + NDJSON token streaming; an async, health-checked connection pool; input guardrails; startup model warm-up with a fail-fast device gate | 🟢 |
| **Testing** | pytest; unit-pinning load-bearing assumptions (cross-encoder sort direction, RRF math, OR-tokenizer, metric derivation) | 🟢 |
| **Front-end craft** | a self-contained streaming SPA; a token-driven dual-theme (OKLCH) design system with a typographic hierarchy | 🟡 |
| **Experiment design** | staged experiments (cheapest decisive test first); a *control* in a cosine A/B; accepting a falsified hypothesis and reading the architecture under a flat metric; isolating a fake variable (threading) from the real one (benchmark input length) with a controlled A/B | 🟢 |
| **Containerization / MLOps** | multi-stage `uv` + CUDA Docker build, non-root, health-checked; weights baked in at build time (immutable, offline-startable); a real multi-service `docker compose up`; governance enforced at the image layer; idempotent, resumable ingest for long re-indexes | 🟡 |

### 1B. Cognitive habits (the senior-maker skills)

These matter *more* than the table above. An interviewer can teach a junior `tsvector`; they can't easily teach these.

| Habit | The move | Where I did it |
|---|---|---|
| **Measure, don't eyeball** | Trust the aggregate over the golden set, never one query | Phase 4: one reranked query looked noisy, aggregate was 0.75 |
| **Distrust your own metric** | A "failure" can be a mislabeled ground truth — inspect it | **q18**: caught a wrong golden label instead of "fixing" the model |
| **Single-variable attribution** | Change one thing, measure, attribute | 7B→3B: abstention held, so the *model* wasn't the variable |
| **Isolate & inspect** | When output ≠ expectation, `repr()` the exact value in isolation | The PONG mystery → unsaved buffer, not a logic bug |
| **Know what *not* to build** | The best fix is sometimes no fix | Didn't add a relevance gate — it would've suppressed a correct answer |
| **Predict-then-verify** | State a bet, then let reality grade it | Predicted "abstention is hardest"; reality: 2/2, and the one "miss" was a label bug |
| **Tradeoff literacy** | Name the price of every choice out loud | local vs API (latency), 7B vs 3B (speed vs "IQ"), k vs coverage |
| **Run it to find what reading can't** | Some failures are emergent — stand the service up and hit it | Idle-dropped pool connection, a strangling timeout, ONNX cold-start — all found live |

### 1C. War stories (STAR-ready — memorize 3–4)

1. **The AND-vs-OR lexical bug** — `websearch_to_tsquery` ANDs terms by default, so full-question queries matched *nothing* (lexical = 0.00). Fixed with an OR-of-content-terms query. _Lesson: read the tool's default behavior, don't assume._
2. **The q18 golden mislabel** — the eval flagged a hallucination; inspecting the retrieved chunks showed the model was *correct* and the ground-truth label was wrong. _Lesson: your eval is only as good as its labels._
3. **The measure-then-decide embedder choice** — benchmarked MiniLM-384 vs e5-large on CPU (`eval/bench_embed.py`): a **50× throughput gap** (20.6 vs 0.4 chunks/sec), so I shipped the fast baseline and documented e5-large / BGE-M3 as the measured upgrade path. _Lesson: pick models with a number, not a vibe — and when the number contradicts your guess, believe the number._
4. **Save-before-run / disk-vs-editor** — running code silently executed the *saved* file, not the editor buffer; `repr()` in isolation exposed it. _Lesson: what runs is what's on disk, not what you're looking at._
5. **The bugs that only appear when you run it** — I stood the FastAPI server up and it surfaced three failures no unit test shows: a pooled DB connection the Docker proxy severed while *idle* (fixed with validate-on-checkout), a timeout that strangled *slow-but-correct* CPU generation (split into a short connect-timeout and a generous read-timeout), and an ONNX Runtime cold-start that made the *first* request 3× slower (fixed by running one warm-up inference at startup: **>90 s → 27 s**). _Lesson: the gap between "runs on my machine" and "serves strangers" is exactly this class of emergent bug — the only tool that crosses it is standing the thing up and hitting it._
6. **The metric that hid the win** — reranking read as a modest `answer@5` gain (0.45 → 0.75). Then I added `answer@1` and MRR — the metrics an *ordering* step actually moves — and the real story surfaced: **0.15 → 0.75** and **0.25 → 0.75**. It doesn't merely fetch the answer, it pins it to rank 1. _Lesson: measure what your component actually does, or a set-membership metric will flatter or hide it._
7. **The embedder that only read the first paragraph** — `paraphrase-multilingual-MiniLM-L12-v2` truncates at ~128 tokens, but I was chunking at 512. I caught it with a cosine test: two chunks sharing their first ~128 tokens but with *completely different* tails (tilapia vs. astrophysics) embedded **identically** (cosine 1.0000) — so ~75% of every chunk was invisible to the dense vector. Then I quantified the blast radius: **~half my golden answers (10 of 19) have their answer text *past* that ~128-token window** — dense can only reach those chunks by their opening, not the answer — and **four of them (q07, q08, q20, q22) are precisely the reranker's remaining misses**. So truncation is a *root cause* of the recall gap the candidate sweep surfaced, not a curiosity. The fix is a longer-context embedder or smaller chunks; both re-run the whole eval, so rather than thrash the numbers I ran it as a *staged* experiment — **story #8**. _Lesson: know your embedder's context window — never assume a model reads the whole thing you hand it._

8. **The embedder experiment I ran to completion — and the hypothesis it disproved.** The truncation catch (#7) predicted a fix: swap the 128-token MiniLM for a longer-context, stronger multilingual embedder. I didn't just assert it — I ran it as a staged experiment, cheapest decisive test first.
   - **Pilot (mechanism, minutes).** A controlled cosine A/B: for each dense-blind answer chunk, `Δ = cos(query, full_chunk) − cos(query, head-truncated-to-128-tok)`. MiniLM's Δ came back **+0.001** — many answers *exactly* `0.000`, because chopping tokens it never encoded is a no-op. That flat-zero **control** *proves* the truncation. e5-large's Δ was **+0.030**, positive on every blind chunk — it reads the tail. Mechanism confirmed with a control, no 7-hour re-index needed.
   - **2a — cheap outcome probe (~30 min).** Re-chunk to 128 tokens with the *same* model (throwaway DB) to kill truncation. The result was counterintuitive: dense `document@5` rose **0.65 → 0.90** (the model can finally *see* the answer), but the final reranked `answer@5` **fell 0.75 → 0.60**. Smaller chunks = 4.4× more of them, diluting answer-chunk precision and starving the reranker's context. **Chunk size is a precision/recall tradeoff, and 512 was already near the sweet spot** — the "obvious fix" made the product worse.
   - **2b — the real upgrade (~7 h re-index, resumable).** e5-large at 512-tok, full-index re-embed, driven in small restartable batches by [`scripts/reindex_e5.ps1`](../scripts/reindex_e5.ps1) — each invocation embeds the next few documents and reports progress, so the whole 7-hour job survives a machine sleep or a Ctrl-C between batches. **Dense retrieval more than doubled** (`answer@5` 0.35 → 0.75, MRR 0.33 → 0.72) — e5's dense *alone* equals MiniLM's entire reranked cascade. Yet the **final** number barely moved (0.75 → 0.80): the reranker had been *masking* MiniLM's weak dense, so a stronger retriever relocates the work earlier rather than lifting the ceiling.
   - **The disproven prediction (the best part).** I predicted e5 would recover the **cross-lingual** misses (q08/q20/q22 — Spanish question, English answer phrase). **It didn't.** All three survived MiniLM-512, MiniLM-128, *and* e5-large; e5 recovered a *different*, same-language question instead. So those misses aren't an embedder-quality problem at all — they need a different lever (query translation), which the experiment *redirected* me toward. _Lesson: run the expensive experiment, let it disprove you, and read the architectural truth under a flat headline — a stronger embedder bought reranker-independence, not a higher final score._

9. **The 20× slowdown that wasn't the threading bug I diagnosed.** After moving the reranker off ONNX onto `torch`/CUDA, a call that used to take ~25 s started taking 500+ s on CPU. My first hypothesis — thread-pool oversubscription from `asyncio.to_thread` (each worker thread lazily re-initializing PyTorch's own internal thread pool, up to a dozen-plus pools competing for twelve cores) — was plausible, well-documented, and *wrong*. I built the fix (a single dedicated worker thread for every torch call) and the number didn't move. Rather than ship a fix that hadn't fixed anything, I read the reranker's own cached source and found its batching was already correct — one tokenize-and-forward-pass call over the whole candidate pool, exactly what I'd have "fixed" it to do. The real cause was embarrassingly upstream: my benchmark used short synthetic strings, not the ~500-character real chunks the live system reranks. Rerunning the *identical*, unthreaded, single-call benchmark against real chunk text reproduced the full 500 s on its own — same code, same machine, different input. The gap was eager-mode PyTorch on unoptimized weights doing real work on real-length input, versus the old backend's quantized, graph-optimized ONNX runtime — a real cost, not a bug, and one I'd been hiding from myself with an unrepresentative benchmark. _Lesson: a plausible hypothesis with a plausible fix isn't a confirmed diagnosis until the fix moves the number — and the fastest way to find the real cause is to read the source, not rewrite it._

10. **The bug that only a real user found.** I'd verified the async serving rewrite thoroughly — curl, pytest, a full live round trip. Then someone else opened the app and hit an immediate "network error" my own testing never had. Root cause: `/ask/stream` awaited the *entire* retrieval phase before constructing the response, so the client got zero bytes -- not even HTTP headers -- for however long CPU retrieval took (minutes). Browsers apply a far stricter time-to-first-byte timeout than they do to an already-open, slow stream; my own patient terminal-watching had never sat in that failure window long enough to trip it. Fix: move retrieval (including the pooled DB connection) *inside* the response generator so headers flush immediately, which forced generation onto `ollama.AsyncClient` too since the whole handler was now one coroutine. Verified directly: a 10s curl now shows `200 OK` with full headers in a fraction of a second, then legitimately times out waiting on the still-slow body -- proof the mechanism, not just the symptom, was fixed. The same user then hit a *second*, independent bug the same way -- Ollama's 300s read timeout, calibrated before the CPU cost of the new backend was known, firing mid-generation on real prefill time. I raised it to match what's actually measured now instead of what was assumed when the number was first chosen. A third, smaller catch fell out of the same session: `uvicorn --reload` logged "Reloading..." after a config edit, and I trusted it -- two hours later the *original*, un-restarted worker was still serving requests with the stale config. _Lesson: your own testing is not the same test as someone else using the thing -- and a log line claiming success is a claim, not a fact; check the process itself._

> **One-breath answer to "what did you learn building this?"** — *"That the engineering is 20% of it — the other 80% is measuring honestly, distrusting my own metrics, and knowing which problems not to solve. I caught a wrong ground-truth label that would've made me 'fix' a model that was already right, caught myself about to ship a threading fix for what was really an unrepresentative benchmark, and then had a real user find two live bugs in the exact code I'd already called verified -- because my own testing and someone else's first click are not the same test."*

---

## Part 2 — Where we stand

| Phase | Deliverable | Status | Measured result |
|---|---|---|---|
| 1 | Golden set + 4-tier governed corpus + validator | ✅ | 22 Q (20 answerable + 2 no-answer), validator green |
| 2 | Ingestion → pgvector + FTS | ✅ | 10,330 chunks; dense baseline **0.35@5** |
| 3 | Hybrid (dense+lexical+RRF) | ✅ | **0.45@5** (lexical alone 0.50 — a real RRF nuance) |
| 4 | Cross-encoder reranking | ✅ | **0.75@5**, doc@5 **0.90**; +6 fixed, 0 regressions |
| 5 | Grounded / cited / abstaining generation | ✅ | page-precise citations; 3B ≈ 7B on abstention, ~3× faster |
| 6 | Serving: FastAPI + NDJSON evidence streaming | ✅ | streams a cited answer, sources-first, end-to-end |
| 7 | Bilingual, dual-theme web app + transparency drawer | ✅ | answer *or* honest abstention, evidence one click away |
| 8 | Production hardening (config-driven models · pooled + health-checked DB · guardrails · startup warm-up) | ✅ | verified on a live server; first request **>90 s → 27 s** |
| — | Eval-code hardening: single-source `_scoring` + 9 unit tests | ✅ | assumptions pinned (reranker direction, RRF, tokenizer, MRR) |
| — | **Embedder experiment** (mechanistic pilot → re-chunk probe → full e5-large re-index) | ✅ | e5 *dense* **0.35 → 0.75**; final 0.75 → 0.80; hypothesis **falsified** (cross-lingual misses stubborn) |
| — | **Containerized** (two-stage `uv`, non-root, health-checked) + **resumable ingest** (`--resume`) | ✅ | governance holds at the image layer; long re-index now restartable in batches |
| — | **Async serving** — `psycopg_pool.AsyncConnectionPool` + FastAPI DI, retrieval released before generation | ✅ | verified live; found and fixed a Windows-only `asyncio` event-loop incompatibility |
| — | **GPU inference pivot** — `torch`/CUDA embedder + reranker, fail-fast device warmup, weights baked into the image | ✅ | GPU path built and code-verified; CPU path measured at **~500 s/rerank** — a named, honest cost |
| — | **`eval/eval_pipeline.py`** — concurrent, HTTP-level eval harness against a new retrieval-only `/retrieve` route | ✅ | pydantic-validated golden set; live-verified against a real server |
| — | **`docker compose up`** — the app wired into compose alongside the DB, not just documented | ✅ | config-validated; a claim in a doc is a promise a stranger will test |
| — | **Found by a real user, fixed same-session**: `/ask/stream` sent zero bytes (not even headers) until retrieval finished; a stale 300s Ollama read-timeout | ✅ | root-caused via the actual response-send order, not assumed; both closed and verified live |
| — | **Real-time progress feedback** — `/ask/stream` reports backend-driven phase transitions (translating / reranking / generating); the UI's decorative growth animation became an actual progress path | ✅ | live-verified: a real question, the correct stone lighting up within seconds, screenshotted |
| 9 | Public URL + write-up | ⏳ | now also a GPU-hosting decision, not just a cost one — named explicitly, not hidden |

**Headline number, memorized:** retrieval@5 **0.35 → 0.45 → 0.75** (dense → hybrid → reranked). The sharper cut — the *ordering* the headline hides: **answer@1 `0.15 → 0.75`** and **MRR@5 `0.25 → 0.75`**. Whenever the answer is in the pool, the reranker pins it to **rank 1**. Deployed (open-tier only, no copyrighted docs): **0.65 answer@5 / 0.75 doc@5** — the honest number a stranger gets.

**The embedder upgrade, measured (not guessed):** a full e5-large re-index (§4.13) lifts *dense* answer@5 **0.35 → 0.75** — e5's dense *alone* ≈ the shipped reranked cascade — but the *final* only **0.75 → 0.80**, because the reranker was already masking the weak dense. I **kept MiniLM shipped**: a noise-level final gain doesn't justify e5's ~8× slower query embed and a `VECTOR(1024)` migration. The value was the *finding*, not the swap.

**Generation — the honest read (qwen2.5:3b, k=5):** grounds, **cites to the page** (`Book.pdf, p. 37`), and **abstains** on genuine no-answer via a canonical, machine-checkable phrase. Citation is scored against a deliberately strict regex that *under*-counts — the model cites in each source's own idiom (title, author-year, filename), so the interview move is to name that as a **measurement** limitation, not a model failure. Retrieval latency is production-grade (bi-encoder p95 ≤ 250 ms); the cross-encoder is now GPU-scoped for real speed — on CPU it measures **~500 s/query** (war story #9), up from the old ONNX backend's ~25 s, the honest cost of an eager-mode `torch` model on unoptimized weights. A GPU host is no longer an optional speed lever; it's the deploy requirement, and the config makes it a one-env-var swap once you have one.

**Stack:** Postgres 16 + pgvector (Docker, host 5433) · `sentence-transformers` MiniLM-L12 (384-dim, `torch`/CUDA) · `jinaai/jina-reranker-v2-base-multilingual` reranker (`transformers`, `trust_remote_code`, pinned to a reviewed commit) · `qwen2.5:3b` via Ollama (config-swappable host/model) · FastAPI + `psycopg_pool.AsyncConnectionPool` (async DI, validate-on-checkout) · a self-contained OKLCH / tri-font streaming web app · pytest · `uv`/hatchling · Docker Compose (multi-service, GPU-scheduled).

---

## Part 3 — What's left to build

**Shipped** (all committed and pushed): the serving layer (FastAPI + NDJSON evidence streaming), the bilingual dual-theme web app, the full production-hardening pass, and a unit-tested single-source eval. What used to be "what's left" is now the body of the project.

**The one genuinely remaining step:** a public **URL** — and it is *not* a code problem, though the GPU pivot (§4.15) sharpened what kind of decision it is. The app is config-swappable to a hosted model and serves the open tier only; going live now needs a GPU box, or hosted reranker + LLM endpoints, not just "a faster CPU box" — I measured the CPU path at ~500 s/query for reranking alone (§4.15) and would not present that as production-viable. Naming that plainly is the honest version of "not a code problem."

**Honest headroom — the strongest "what would you do next" answers, each measured rather than guessed:**
- **Query-side translation — done, and it *validated* the diagnosis (§4.14).** I ran it: translate the question to the other language, then retrieve + rerank in the bilingual space. It recovered **q08** (rank ∞ → 1) — a cross-lingual miss that survived MiniLM-512, MiniLM-128, *and* e5-large — with **zero regressions**, reaching the same 0.80 as the e5 upgrade at a fraction of the cost. q20/q22 didn't recover because the local 3B *translator* garbled agronomy terms (`siembra`→"tillage", `turba`→"turf"), so the residual is **translation quality**, a named next lever, not architecture. Shipped behind a config flag with soft fallback.
- **Embedder upgrade — done, not headroom.** I ran it end to end (§4.13). e5-large doubles dense recall, but the reranker already captures most of that at the final stage, so I kept fast MiniLM shipped and documented the tradeoff (8× slower query, `VECTOR(1024)` migration) rather than pay it for a noise-level final gain. *The candidate sweep and the truncation catch both pointed at the embedder; the experiment showed the truer bottleneck was the reranker compensating — and the residual gap is cross-lingual.*
- **RAGAS** faithfulness / context precision-recall — the semantic layer a substring metric can't reach.
- **A citation-format one-shot** — pin the model to one citation idiom so the *measurement* stops undercounting a model that already cites faithfully.
- **Latency** — a GPU or hosted reranker collapses the ~25 s rerank and the CPU generation to sub-second, one env var, no code change.
- **Chunk-anchored ground truth** — *considered and rejected*: brittle to re-chunking, and it would false-negative legitimately-recurring answers. Knowing what **not** to build is the point.

---

## Part 4 — The Masterclass

Explaining the **whole system** as one coherent narrative. Deliver this and you can carry a 45-minute technical interview.

### 4.1 — The one-sentence pitch
> *"Agroteca is an eval-first, bilingual RAG system over agricultural documents: it answers agronomy questions strictly from a governed corpus, with citations, and abstains when it can't — and every component was measured against a golden set built before the system existed."*

### 4.2 — The mental model: two machines, not one

```
        OFFLINE (build the index, once)            ONLINE (answer a query, every time)
   ┌─────────────────────────────────────┐   ┌──────────────────────────────────────────┐
   │ PDFs → extract → normalize → chunk → │   │  question → embed → ┌ dense  ┐            │
   │ embed → store (pgvector + tsvector) ─┼───┼──────────────────── ┤ +RRF+  ├→ rerank →  │
   │                                      │   │                     └ lexical┘   top-k    │
   │                                      │   │                                     ↓      │
   │                                      │   │              LLM: ground / cite / abstain  │
   └─────────────────────────────────────┘   └──────────────────────────────────────────┘
```

**Why it matters:** embedding 10,330 chunks is expensive and done *once*; a query must be fast.

### 4.3 — The retrieval cascade, stage by stage

Each stage exists to fix the previous stage's specific weakness:

1. **Dense (semantic)** — embed the query, nearest chunks by cosine. *Strength:* meaning/paraphrase, cross-lingual. *Weakness:* fumbles exact tokens (`WL-323`). → **0.35**
2. **Lexical (keyword)** — Postgres FTS matches exact terms. *Strength:* codes/names. *Weakness:* blind to paraphrase. *(Bug: it ANDs terms → matched nothing → fixed with OR-of-terms.)*
3. **Hybrid + RRF** — run both, fuse by **rank** (`1/(k+rank)`), not raw score (cosine vs `ts_rank_cd` aren't comparable). → **0.45** *(nuance: hybrid < lexical@5 here — a real RRF subtlety).*
4. **Cross-encoder rerank** — top ~20 candidates re-scored by a model that reads *(query, chunk) together*. Slow but precise → runs only on the small pool. → **0.75**, doc@5 **0.90**.
5. **Generation** — feed top-k to a local LLM with three rules: **GROUND / CITE / ABSTAIN**.

> **The single most important sentence:** *"Each retrieval stage is a targeted fix for the previous stage's blind spot — dense misses exact codes, lexical misses paraphrase, hybrid gets both, the reranker sharpens precision. And I can prove each moved the number: 0.35 → 0.45 → 0.75."*

### 4.4 — Why "recall then precision" (the cascade principle)

```
   cheap + wide (bi-encoders)  →  expensive + sharp (cross-encoder)
   10,330 chunks → top 20          20 pairs → top 5
   "don't lose the answer"         "put the answer on top"
```
Never run the expensive model on the whole corpus — only on the 20 candidates the cheap models narrowed to. That's how real systems scale.

### 4.5 — The eval-first spine (the project's thesis)

Most RAG demos: build → eyeball 3 questions → ship. **This inverts it:** the golden set (22 hand-verified Q/A, stratified: factual, exact-term, synthesis, no-answer, cross-lingual) was built **before** the retriever, so every change is *measured, not guessed*.

Dual metric: **answer-chunk@k** (did the *exact answer chunk* surface?) vs **document@k** (did the *right document* surface?). The gap tells you chunking problem vs retrieval problem.

> **One-breath answer to "how do you know it works?"** — *"I don't eyeball it — I score it against a golden set I built before the system, with a dual metric that separates 'found the right doc' from 'found the exact answer chunk.' That's how I know reranking fixed six questions with zero regressions."*

### 4.6 — Grounded generation: the three disciplines

```
GROUND  → answer only from the retrieved chunks (no outside knowledge)
CITE    → (source_file.pdf) after the claim
ABSTAIN → exact canonical phrase when the corpus can't answer
```

Two senior details:
- **Rules live in the `system` role**, separate from retrieved context — a prompt-injection payload hiding inside a document chunk can't easily override the abstention rule.
- **The abstain phrase is *canonical* on purpose** — an exact, machine-detectable string, so abstention is a measurable metric, not a vibe.

### 4.7 — The model choice, defended

> *"RAG generation is a grounding-and-obedience task, not a reasoning task — retrieval already located the answer. So I used a 3B local model, and I checked it: 7B and 3B abstained identically on the no-answer cases, at ~3× lower latency. Honest caveat — my no-answer set is small, so I call that directional and would grow it before treating it as settled. I kept it local for zero cost, offline operation, and to keep copyrighted source chunks on-device."*

### 4.8 — The governance layer (the differentiator)

Four tiers, each controlling *where a document may appear*: **open** (deployable), **local-only** (copyrighted, never shipped), **synthetic** (grounded original notes, cited as synthetic), **distractor** (eval-only hard-negatives). Two principles decide entry: the **idea/expression** line (facts are free, expression isn't) and **citation integrity** (never attribute synthesized text to a real source).

> **One-breath answer to "how do you handle copyright/licensing?"** — *"Provenance is first-class — every document lives in exactly one tier, and the tier controls whether it can be served publicly. Copyrighted material runs only in local mode and is never in a public deployment."*

### 4.9 — Interview lightning round (rapid-fire)

- **"Why hybrid search?"** → *Dense misses exact codes, lexical misses paraphrase; fusion gets both.*
- **"Why a reranker?"** → *Bi-encoders give good recall but imperfect precision — a cross-encoder reads the pair together and sharpens the top, cheaply, on a small pool.*
- **"Why RRF over score fusion?"** → *Cosine and `ts_rank_cd` scores aren't on the same scale; fusing by rank sidesteps that.*
- **"Why local models?"** → *Zero inference cost, offline for the farm use case, and copyrighted chunks never leave the machine.*
- **"How do you stop hallucination?"** → *Ground to context, cite every claim, a canonical abstention phrase I can measure — plus rules in the system role to resist injection from the corpus itself.*
- **"Biggest lesson?"** → *A metric failure can be a wrong label; I caught one and avoided a fix that would've suppressed a correct answer.*

### 4.10 — The serving layer, and the gap it closes

The thing an interviewer is really probing: can you take a working model and *serve strangers*?

- **Config-driven models** — the LLM, host, timeouts, pool size, and inference device are settings, not source; a deploy swaps the unservable local CPU model for a hosted GPU one with an env var. The single change that makes it deployable at all.
- **An async, health-checked connection pool** — `psycopg_pool.AsyncConnectionPool` (`min_size=5, max_size=20, timeout=30.0`), injected via FastAPI's `Depends()` so routes never touch the pool object; each connection validates on checkout, so an idle-dropped connection is replaced transparently rather than 500-ing a request. The stream path's dependency releases its connection *before* the minutes-long generation — enforced by FastAPI's own dependency-teardown ordering, not a manual `finally` — so an answer never holds the pool hostage.
- **A dedicated retrieval-only route** — `/retrieve` shares one retrieval implementation with `/ask/stream` but skips generation entirely, so an eval harness or a monitoring probe can measure retrieval latency without paying for a multi-minute LLM call it doesn't need.
- **Explicit, fail-fast model warmup** — both models load and run one real forward pass *inside the serving process itself*, before it accepts connections; `AGROTECA_DEVICE=cuda` refuses to start if no GPU is visible rather than silently degrading. (`AGROTECA_DEVICE=auto` opts out of that guarantee, deliberately, for local testing.)
- **Guardrails on the public surface** — bounded input length (a 1 MB payload can't reach the FTS regex), `Literal`-validated feedback, and both models warmed at startup so concurrent first requests can't each load the 1.1 GB reranker.
- **A timeout that tells slow from broken** — fail fast if the model is *unreachable* (short connect timeout), stay patient if it's merely *slow* (generous read timeout). A flat timeout punishes the honest slow answer.

> **One-breath:** *"The gap between a demo and a service is the failures that only appear under real traffic — a stale pooled connection, a timeout that can't tell slow from wedged, a cold-start on the first request, an event loop that silently can't run my async driver on Windows. I stood the server up, hit it, read the tracebacks, and fixed each one — then re-verified against the running process."*

### 4.11 — The eval as engineered software, not a script

The eval is the project's thesis, so I hardened the *measurement code* to the same standard as the system:

- **One source of truth** — every metric derives from a single primitive, `answer_rank` (the 1-based position of the answer chunk); `answer@1`, `answer@k`, `document@k`, and MRR all fall out of it, so a new metric is a one-file change that can't drift across the eval scripts.
- **The assumptions are unit-tested** — nine pytest cases pin the load-bearing, silent-failure-prone logic: that a higher cross-encoder score truly means more relevant (the `reverse=True` bet — measured +1.25 vs −3.72), the RRF formula and its first-seen-row rule, the OR-tokenizer (including exact-code preservation), and the rank-derived metrics.
- **The labels are stress-tested** — a uniqueness check confirms each `must_contain` phrase identifies its answer chunk; when it flagged one, I *inspected* and found the recurrence benign (the answer is genuinely stated three times) and recorded that, rather than degrade a correct label.
- **The free lever is measured before the expensive one** — before reaching for a heavier embedder I measured hybrid `answer@N` (the reranker's recall ceiling) across pool sizes; it's flat, and four of five remaining misses aren't in the top-60 at all. So the embedder is the *measured* next lever — a bigger pool is not.

> **One-breath:** *"My eval is the thesis, so I engineered it like the system: one definition of every metric, unit tests pinning the assumptions a refactor could silently flip, and a uniqueness check on my own ground truth. When the check flagged a label, I inspected before changing it — a flag is a prompt to look, not a verdict."*

### 4.12 — The interface: transparency as the product, type as the persuasion

A grounded RAG that *shows* its grounding is a more trustworthy object than one that asserts it. So the UI makes a slow, honest system feel alive and reveals its evidence:

- **Streaming reduces anxiety, not latency** — the wall-clock is identical; watching an answer assemble is a different experience from a dead spinner. Perceived-vs-actual latency is the whole game in a slow-but-honest system.
- **Every answer carries a receipt** — a one-click drawer exposes the exact chunks, cross-encoder scores, governance tiers, page numbers, and a retrieval-vs-generation latency split. The opposite of a confident black box.
- **"I don't know" has a face** — abstention renders as a distinct empty-basket state, never a fabricated guess.
- **A design system, not a stylesheet** — two perceptual **OKLCH** themes over one token layer, plus a typographic hierarchy that reserves a monospace face for every *data* surface, so a reader feels precision before parsing a word. Delivered as a token remap over existing markup — zero changes to the streaming, the drawer, or the logic.

> **One-breath:** *"Most AI demos hide the machinery and ask for trust; I did the reverse — every answer opens a drawer with the exact evidence, scores, tiers, and page-level citations. Transparency is the product, and the polish is typographic, not decorative: the data wears a monospace face, so the interface reads as precise."*

![The evidence drawer, open: each source shown with a governance-tier badge, cross-encoder relevance score, page-level citation, and snippet.](img/hero-drawer.jpg)

### 4.13 — The embedder experiment: measure, predict, run, be wrong well

The retrieval headline is one embedder deep. To test whether a better one was the next lever, I ran a three-stage experiment and let it grade my prediction — the whole arc lives in war story #8; here is the *systems* reading.

| final cascade | answer@5 | answer@1 | doc@5 | mrr@5 |
|---|---|---|---|---|
| MiniLM-384, 512-tok (shipped) | 0.75 | **0.75** | **0.90** | 0.75 |
| MiniLM-384, 128-tok (kill truncation) | 0.60 | 0.45 | 0.90 | 0.52 |
| **e5-large-1024, 512-tok (stronger model)** | **0.80** | 0.70 | 0.85 | 0.75 |

Three findings, in ascending order of how senior they sound:

1. **The dense layer, not the final layer, holds the truth.** e5's *dense* `answer@5` went **0.35 → 0.75** — a +0.40 swing that is not noise on any sample size. The *final* number moved +0.05, because the cross-encoder reranker had been *compensating* for MiniLM's weak dense. **A stronger retriever relocates quality out of the expensive reranker and into cheap dense search** — the same accuracy at lower serving complexity (you could shrink or drop the reranker). That architecture win is invisible in the flat headline; you only see it if you read the per-stage rows.
2. **Chunk size is a tradeoff, not a bug.** Eliminating truncation (128-tok) *raised* document recall but *lowered* answer precision — 512 was already near-optimal for this cascade. The obvious fix degraded the product; only the measurement caught it.
3. **My hypothesis was falsified — and that's the credential.** I predicted e5 would fix the cross-lingual misses; it fixed a *different* one and left all three cross-lingual cases missed. Those are a representation-alignment-across-languages problem compounded by exact-answer-phrase specificity — a bottleneck no embedder swap has moved. A clean negative that redirects the next experiment (query-side translation) instead of another blind model swap.

> **One-breath:** *"I benchmarked the embedder upgrade end to end. Dense retrieval doubled — but the final number barely moved, because my reranker was already masking the weak dense; a better retriever just did the reranker's job earlier. And the misses I predicted it would fix, it didn't — they're cross-lingual, a deeper problem. So the real result isn't 'a better model,' it's 'the reranker was compensating, and my recall gap needs translation, not vectors.' I'd rather report that than a number that flattered a wrong story."*

> **The methodology worth naming out loud:** *cheapest decisive test first.* A mechanistic pilot with a control (minutes) told me the mechanism was real; a same-model re-chunk probe (30 min) told me the *outcome* of removing it; only then did I spend the 7-hour re-index — and I made that re-index **resumable** so a machine-sleep couldn't cost the run. Staged experiments spend compute in proportion to what's already been de-risked.

### 4.14 — The query-translation lever: the diagnosis, confirmed

The embedder experiment (§4.13) left a hypothesis it couldn't test itself: if the residual misses are *cross-lingual*, the fix is on the query side, not the model. So I ran that too — translate each question to the other language, retrieve `hybrid(original) ∪ hybrid(translation)`, and rerank the union by the **max** cross-encoder score over both queries (an English answer chunk earns its score from the English query member, a Spanish chunk from the Spanish one).

| config (shipped MiniLM index) | answer@5 | answer@1 | mrr@5 |
|---|---|---|---|
| shipped rerank | 0.75 | 0.75 | 0.75 |
| **+ query translation** | **0.80** | **0.80** | **0.80** |

**`q08` went from unranked to rank 1** — a cross-lingual miss that had survived MiniLM-512, MiniLM-128, *and* e5-large. The query-side lever moved what no embedder could, with **zero regressions** — the diagnosis *confirmed by experiment*, not by argument. And it reaches the same 0.80 as the e5 upgrade **at a fraction of the cost**: no re-index, no `VECTOR(1024)` migration, no ~8× query latency — just a translation call the app's LLM already affords. (They're complementary: e5 recovered a *same-language* miss, translation a *cross-lingual* one; stacked, they'd likely reach ~0.85.)

The honest bound: q20/q22 *didn't* recover — because the local 3B **translator** mistranslated domain terms (`siembra`→"tillage", `turba`→"turf"). The residual is **translation quality**, a named and fixable cause (a stronger translator, or a tiny agronomy glossary), not an architectural wall.

> **One-breath:** *"The embedder experiment told me the misses were cross-lingual, so I tested the query-side fix: translate the question, retrieve and rerank bilingually. It recovered a miss no embedder swap could, at a fraction of the cost — the diagnosis confirmed. The two it didn't recover failed on the translator's vocabulary, not the architecture. That's a full scientific loop: measure, get disproven, follow the disproof to the real lever, and confirm it."*

### 4.15 — The GPU-pivot performance investigation: a second falsified hypothesis

Where §4.13 is "I predicted a fix and the experiment disproved it," this is its mirror: "I diagnosed a bug, built the fix, and the experiment disproved *the diagnosis*." Both are the same discipline — believe the measurement over the story — applied to a different kind of claim: not "will this model change help," but "is this the actual bug."

**The symptom.** After moving the embedder and reranker from `fastembed` (ONNX, ONNX-optimized weights) to `sentence-transformers`/`transformers` (raw PyTorch, for real GPU inference), a single 20-candidate rerank call — ~25 s under the old backend — started taking 500+ s on CPU, sometimes not returning inside a 10-minute window at all.

**Hypothesis 1 (plausible, wrong): thread-pool oversubscription.** `asyncio.to_thread` draws from a default executor of up to ~16 worker threads; PyTorch's CPU backend lazily initializes its own OpenMP thread pool the first time it runs on a given OS thread. Different requests landing on different worker threads could mean a dozen-plus competing thread pools fighting over twelve physical cores — a real, documented PyTorch failure mode, and a clean story that fit the symptom. I built the fix: one dedicated worker thread for every torch call, so the pool initializes exactly once.

**The falsification.** The fix shipped, the number didn't move. Still ~500 s. A hypothesis that survives the fix it predicted isn't confirmed — it's wrong, and the honest move is to say so rather than ship the patch because it *sounds* right.

**Hypothesis 2 (confirmed by reading the source, not guessing again): the benchmark was lying.** Before writing a second theory, I read the reranker's own cached implementation — not my mental model of it, the actual code — and found its batching was already correct: tokenize the whole candidate pool together, one forward pass. There was nothing to fix at that layer. So I isolated the one variable I hadn't controlled: my own test used short synthetic strings (~15 words); the live system reranks full ~500-character document chunks. Rerunning the *identical*, single-threaded, unmodified call against real chunk text reproduced the full 500 s on its own — same code, same machine, only the input changed.

| what changed | rerank(20 candidates) |
|---|---|
| synthetic ~15-word strings (my original benchmark) | 10.6 s |
| real ~500-char chunks, same call, same machine | 511.3 s |

**The real cause.** Eager-mode `torch` inference over realistic-length input, on unoptimized (unquantized, non-graph-compiled) weights, versus the old backend's ONNX Runtime doing the same job with quantization and CPU-specific graph optimization baked in. Not a threading bug, not an unbatched loop — a genuine ~20× cost difference between two inference backends, invisible in my own benchmark until I fed it realistic data.

**What shipped instead of a fake fix:** the single-worker-thread executor stayed (a legitimate hardening, just not *the* fix), `AGROTECA_DEVICE=auto` for CPU-local/GPU-prod parity without contradicting the fail-fast startup guarantee, and the CPU-vs-GPU cost was written into the deploy docs as a named requirement rather than an implied speed option.

> **One-breath:** *"A rerank call went from 25 seconds to 500. My first theory — thread-pool oversubscription — was plausible and well-documented, so I built the fix. The number didn't move, which meant the theory was wrong, not that I hadn't tried hard enough. I read the model's actual source instead of guessing again, confirmed its batching was already correct, and found the real cause by controlling one variable at a time: my benchmark used toy-length text, production uses real chunks. Same code, real input, same 500 seconds. The lesson isn't 'I found a bug' — it's that a fix which doesn't move the number is information, not failure, and the fastest way to the real answer is to read the source before you rewrite it."*

### 4.16 — The zero-bytes bug: what my own testing couldn't find

§4.15 was a bug I found by testing myself, harder. This one I didn't find at all — a real user did, on their first try, after I'd already called the serving rewrite verified. The gap between those two facts is the whole lesson.

**The bug.** `/ask/stream`'s route handler `await`ed the entire retrieval phase — hybrid search, cross-encoder rerank — *before* constructing the `StreamingResponse`. Starlette sends the ASGI response's status and headers as soon as that object exists, before it ever touches the body. So with retrieval running first, the client received *nothing* until retrieval finished: not slow bytes, not even a status line. On CPU that's now several minutes of literal silence on the wire. Browsers apply a materially stricter "time to first byte" timeout than they do to an already-open stream that's merely slow to continue — and that's the exact, measured line my own testing (patient, terminal-based, never actually timed against a browser's real threshold) had never crossed.

**Why my own verification missed it.** I'd tested the identical code path — curl, pytest, a full round trip that produced a correct answer. All of those either don't enforce a TTFB-style timeout at all (curl, absent an explicit one) or happened to land inside the window before it fired. The bug was real and reachable the entire time; nothing about my test suite made it invisible, except that none of my tests were a browser sitting through the actual production-representative wait.

**The fix.** Move retrieval — including the pooled DB connection checkout — *inside* the response's async generator, so headers flush the instant Starlette gets the response object, before any of the slow work starts. That required the generator itself to become `async def` (since it now needs `await` for retrieval), which meant the generation half could no longer block the event loop with a synchronous Ollama call — so generation moved onto `ollama.AsyncClient`, run alongside (not replacing) the sync client the CLI/eval paths still use correctly on their own thread.

**Verified the mechanism, not just the absence of the symptom:** a 10-second curl now shows `HTTP/1.1 200 OK` with full headers arriving in well under a second, then correctly times out waiting on the still-slow body — proof the fix changed *when headers are sent*, the actual defect, not just that one manual retry happened not to fail.

**The second, independent bug the same session surfaced.** After the fix shipped, the same user hit a *different* "network error" further into the wait. The instinct that mattered: don't assume it's the first bug again. The traceback said `httpx.ReadTimeout` inside the generation call specifically — Ollama's own 300s read timeout, firing mid-stream on CPU prefill that legitimately now exceeds it. The number wasn't wrong when it was chosen; the CPU cost it was measured against had moved since. Raised to 1800s to match what's actually observed now.

**A third, quieter catch from the same debugging window:** editing a config file, uvicorn's `--reload` logged "Reloading...", and I trusted the log line. Checking the actual worker process's start time two hours later showed the *original*, un-restarted process, still serving every request with the stale config. The log reported an intention, not a completed fact.

> **One-breath:** *"I'd verified the async serving rewrite — curl, tests, a full live round trip. A real user still found a bug in the first ten seconds of trying it, because their browser enforced a timeout my own tools never triggered. The fix was structural: send headers before doing the slow work, not after, which meant the whole request handler had to become properly async end to end. Then the same user found a second, unrelated bug the same way, and I caught myself almost blaming it on the first fix instead of reading the actual traceback. The lesson isn't that I write bad tests — it's that my own testing and someone else's first click are fundamentally different tests, and only one of them is the one that ships."*

---

## Part 5 — Interview Question Bank

Every answer is grounded in *this* project, phrased to say out loud in one or two breaths. **★** marks questions your own results or war stories make especially strong. Practice the ★ ones until they're reflex.

### 5.1 — RAG fundamentals

- **What is RAG, and why not just fine-tune?** → *Retrieval grounds answers in a source corpus at query time. It's cheap to update (add a document, no retraining), it gives citations, and it curbs hallucination. Fine-tuning bakes knowledge into weights — expensive to update and no citations.*
- **Walk me through your pipeline end to end.** → *Offline: PDFs → extract → normalize → chunk → embed → store in pgvector + a full-text index. Online: question → embed → dense + lexical retrieval → fuse with RRF → cross-encoder rerank → top-k → LLM that grounds, cites, or abstains.*
- **★ What's the hardest part of RAG?** → *Retrieval — it's the ceiling on answer quality. The LLM can't answer from context it never received, so I spent most of the effort making retrieval measurable and good: 0.35 → 0.45 → 0.75.*
- **Where does RAG fail?** → *Retrieval misses (answer never reaches the model), bad chunking (answer split across a boundary), and the model over-trusting a topically-related-but-wrong chunk. My eval is built to expose each.*

### 5.2 — Chunking & embeddings

- **How did you chunk, and why that size?** → *Recursive, ~512 tokens with 64 overlap. Size is a recall/precision knob — too big dilutes relevance, too small drops context — and overlap keeps an answer from being cut in half at a boundary.*
- **★ Bi-encoder vs cross-encoder?** → *A bi-encoder embeds the query and each chunk separately, so vectors are precomputed and search is fast — that's retrieval. A cross-encoder reads the (query, chunk) pair together, which is far more accurate but too slow for the whole corpus — so I run it only on the top ~20, as a reranker.*
- **Why a multilingual embedder?** → *The golden set has cross-lingual pairs — a Spanish question against an English source and vice-versa. An English-only model would fail those outright.*
- **What is cosine similarity and why use it?** → *The angle between two vectors; it's magnitude-invariant, which is what you want for semantic closeness regardless of text length.*
- **Why MiniLM / 384 dims?** → *Chosen by measured CPU throughput, not hype. I benchmarked it (`eval/bench_embed.py`): MiniLM-384 does 20.6 chunks/sec vs e5-large's 0.4 — **50× faster**, an ~8-minute full re-index vs ~7 hours, and 19 ms vs 153 ms per query. MiniLM-L12 hit the speed/quality sweet spot for local, PyTorch-free inference; e5-large is the documented upgrade for when I have a GPU or overnight CPU time.* (An earlier draft of this card guessed "~1.2 chunks/sec" — the real measurement is 0.4. Measure, don't estimate.)
- **★ You suspected the embedder was your bottleneck — how did you find out, and were you right?** → *I ran a staged experiment (§4.13). A cosine pilot with a control proved the mechanism (128-token truncation), a cheap same-model re-chunk measured the outcome of removing it, and a full e5-large re-index tested the real upgrade. e5's dense retrieval doubled (0.35 → 0.75) — but the final reranked number barely moved (0.75 → 0.80), because the reranker was already masking the weak dense. And I was **wrong** about the misses: I predicted e5 would fix the cross-lingual ones; it didn't. So I kept MiniLM shipped and let the experiment redirect me toward query translation. The value was the finding, not the swap.*

### 5.3 — Retrieval, hybrid & RRF

- **★ Why hybrid search?** → *Dense retrieval understands paraphrase but fumbles exact tokens like a cultivar code `WL-323`; lexical nails exact tokens but is blind to paraphrase. Fusing them gets both.*
- **Why RRF, and why fuse by rank instead of score?** → *Cosine distance and `ts_rank_cd` aren't on the same scale, so you can't just add the scores. RRF fuses by position — `1/(k+rank)` — which sidesteps the scale mismatch.*
- **Why k=60 in RRF?** → *It's a smoothing constant from the RRF paper — it dampens the pull of the very top ranks so an item that's mid-ranked in both lists still fuses well.*
- **What's HNSW?** → *An approximate-nearest-neighbor graph index — it trades exactness for speed so vector search scales past a brute-force scan.*
- **What's the GIN index for?** → *It's the inverted index behind Postgres full-text search — fast keyword lookup over the `tsvector` column.*

### 5.4 — Reranking

- **★ Why add a reranker?** → *Bi-encoder retrieval gives good recall but imperfect precision — the answer is often in the pool but not on top. A cross-encoder reads each pair together and floats the true answer up. It took me from 0.45 to 0.75 and fixed six questions with zero regressions.*
- **Why only 20 candidates?** → *The cross-encoder is expensive per pair, so it runs on the pool the cheap retrievers already narrowed to — never the whole corpus. That's the cascade: cheap-and-wide for recall, expensive-and-sharp for precision.*
- **How do you know reranking actually helped and didn't break anything?** → *I diffed the per-question misses before and after: +6 fixed, 0 regressed. Aggregate over the golden set, never a single query.*

### 5.5 — Evaluation (your strongest area)

- **★ How do you know the system works?** → *I don't eyeball it. I score it against a golden set of 22 hand-verified Q/A pairs I built before the retriever, with a dual metric.*
- **★ Why build the eval first?** → *It's the single most load-bearing decision — it turns every later change from a guess into a measurement. Most RAG demos build the pipeline and eyeball three questions; I inverted that.*
- **What's retrieval@k?** → *The fraction of questions where the answer is in the top-k results.*
- **Why a dual metric?** → *answer-chunk@k asks "did the exact answer chunk surface?"; document@k asks "did the right document surface?" The gap between them separates a chunking problem from a retrieval problem.*
- **★ How do you evaluate generation, versus retrieval?** → *Retrieval is a clean membership check. Generation isn't — the model paraphrases and varies run-to-run, so exact-string scoring fails. I measure what's mechanical (abstention via a canonical phrase; citation via format) and leave faithfulness to an LLM-judge layer like RAGAS.*
- **What's RAGAS?** → *LLM-judged RAG metrics — faithfulness (is every claim supported by the retrieved context?), plus context precision and recall.*
- **How do you measure abstention?** → *The model must emit an exact canonical phrase when it can't answer, so abstention becomes machine-detectable instead of a vibe.*
- **★ Tell me about a time your eval lied to you.** → *It flagged a "hallucination" on a fish-density question. I inspected the retrieved chunks and the model had grounded correctly in an FAO aquaponics document — the ground-truth label was wrong. I'd been one step from adding a filter that would've suppressed a correct answer. Your metric is only as good as its labels.*
- **★ My citation metric undercounted — why?** → *The model cited faithfully in whatever style each source used — filename, title, or author-year — because it echoes the citation already in the chunk. My regex only caught the filename format. The bug was my measurement's rigidity, not the model.*

### 5.6 — Generation, grounding & hallucination

- **★ How do you stop hallucination?** → *Three disciplines: ground every statement to the retrieved context, cite the source after each claim, and abstain with a fixed phrase when the context can't answer.*
- **★ Why is abstention the hard part?** → *LLMs are trained to be helpful and agreeable, so "I don't know" fights their grain. That's why I made the abstain phrase canonical and measured it — the model held the line on the no-answer cases I have, though I'm honest that's a small sample I'd grow (and pair with RAGAS faithfulness) before calling it settled.*
- **★ How do you defend against prompt injection?** → *The retrieved chunks are untrusted — a document could contain "ignore your instructions." So the ground/cite/abstain rules live in the system role, which the model weights above user-and-content text, keeping a malicious chunk from overriding them.*
- **Why a local LLM instead of an API?** → *Zero inference cost, it works offline for the farm use case, and copyrighted source chunks never leave the machine. The tradeoff is CPU latency, which I measured rather than hand-waved.*
- **★ 7B or 3B — how did you choose?** → *By measurement. Both abstained identically on my no-answer cases (a small sample, honestly) at ~3× lower latency on 3B, because RAG generation is a grounding-and-obedience task, not a reasoning task — retrieval already located the answer.*
- **How does non-determinism affect your eval?** → *Same input gives different phrasing run-to-run, since I don't fix a seed — another reason exact-string scoring of generated answers is doomed and you need semantic judgment.*

### 5.7 — Vector DB & infrastructure

- **★ Why Postgres + pgvector instead of Pinecone/Weaviate?** → *One store does metadata filtering, keyword search, and vector search together; it's production-credible; and I enforce the governance tiers in SQL. No extra service to run.*
- **How do you enforce the corpus tiers?** → *A `tier` column and a `WHERE tier = ANY(%s)` filter — governance lives in the query, so a public deployment simply never selects the local-only tier.*
- **Why Docker, and why port 5433?** → *Reproducible Postgres+pgvector in one command; I mapped host 5433 because 5432 was already taken by another service (Langfuse).*
- **SQL injection — how do you avoid it?** → *Interpolate structure, parameterize values. Table/column names can be built into the query string; anything user-derived goes through a bound parameter, never string-formatting.*

### 5.8 — Data governance & licensing

- **★ How do you handle copyright and licensing?** → *Provenance is first-class. Every document lives in exactly one of four tiers, and the tier controls where it can appear — copyrighted material runs only in local mode and is never in a public deployment.*
- **What's the idea/expression distinction?** → *Facts and ideas aren't copyrightable; the specific expression is. So the system can use the facts from a copyrighted book without serving its text.*
- **Your "synthetic" tier — isn't that fabrication?** → *No — they're original, human-verified concept notes grounded in open sources, and they're cited *as* synthetic. The rule is citation integrity: never attribute synthesized text to a real source.*
- **What are the distractor documents for?** → *Eval-only hard-negatives — deliberately off-topic content to test that the system doesn't retrieve and answer from irrelevant material.*

### 5.9 — Production, latency & scaling

- **What's your latency, and how would you improve it?** → *Generation is CPU-bound on context prefill — a few minutes per question locally. Levers, in order: a GPU, a smaller/quantized model, fewer chunks, and caching. I'd measure each rather than assume.*
- **How would this scale to millions of documents?** → *Retrieval already scales — HNSW is approximate-nearest-neighbor, and the reranker only ever sees a 20-candidate pool. Embedding is offline and batched. The online path barely changes.*
- **How would you serve and deploy it?** → *A FastAPI endpoint wrapping the answer function, streaming tokens for responsiveness, with a measured p95. Deploy the open tier only — governance stays enforced in the query.*
- **★ Why an async connection pool instead of a sync one with a threadpool?** → *A served ASGI app already runs on an event loop; a sync pool call would block a worker thread per in-flight DB operation. `AsyncConnectionPool` awaits a free connection instead, so concurrent requests queue on the pool's own scheduler — cheaper, and it composes cleanly with FastAPI's dependency injection so a route never manages checkout/checkin itself.*
- **★ You moved to GPU inference — why does the app refuse to start on a misconfigured host instead of just running slower on CPU?** → *Because a "GPU microservice" that silently serves on CPU is a worse failure mode than refusing to start — it looks healthy while violating its own latency contract. `require_cuda()` checks at startup, not at request time, so the failure is loud, immediate, and in the deploy logs, not discovered by a confused user three minutes into a hung request.*
- **★ You measured a 20× CPU slowdown after a backend swap — walk me through how you found the real cause.** → *My first hypothesis was thread-pool oversubscription — plausible, documented, and I built the fix. The number didn't move, which told me the theory was wrong. Instead of guessing again, I read the reranker's actual source and confirmed its batching was already correct. Then I isolated the one uncontrolled variable: my benchmark used short synthetic text, not the real ~500-character chunks production reranks. Rerunning the identical call against real chunk text reproduced the slowdown on its own — the gap was unoptimized eager-mode inference on realistic input, not a bug. The fix that didn't work was still useful: it told me where the bug wasn't.*
- **★ You said the serving rewrite was verified — then a user found a bug immediately. What happened?** → *My testing was real but incomplete in a specific way: curl and my own round-trip tests never enforced a strict time-to-first-byte timeout, so they never tripped the actual defect — the route awaited all of retrieval before constructing the response, so the client got zero bytes, not even headers, for the whole CPU-bound wait. A browser does enforce that timeout, and a real user's first request hit it in seconds. The fix was structural — send headers before doing slow work, not after — and I verified the mechanism directly with a short curl showing headers now arrive in under a second. The honest lesson: my own testing and someone else's first click are different tests, and I'd rather find that out from a teammate than from a user in production.*
- **★ How do you decide a timeout value, and how do you know when it's wrong?** → *Calibrate it against a measured cost, not a guess, and re-check it when that cost changes. A 300s Ollama read-timeout was fine when generation was faster; after a backend swap made retrieval and prefill both slower, the same number started killing legitimately-working requests. I found it from the actual traceback, not by assuming a repeat of an earlier bug, and raised it to match what's measured now — the discipline is treating a timeout as a claim about system performance, which needs updating exactly when the system's performance does.*

### 5.10 — Judgment, debugging & behavioral (STAR)

- **★ Tell me about a bug you're proud of finding.** → *Lexical search scored 0.00. `websearch_to_tsquery` ANDs terms by default, so a full-sentence question matched no chunk that contained every word. I switched to an OR of the content terms. Lesson: read the tool's default behavior before trusting it.*
- **★ Tell me about a time you were wrong.** → *I diagnosed a generation failure as a retrieval problem and was about to add a relevance gate. I inspected first — the retrieval was right, the model was right, and the golden-set label was wrong. I revised the diagnosis and fixed the label instead.*
- **How do you debug?** → *Isolate and inspect over theorize. When output didn't match my code, I stopped guessing and printed the function's raw return with `repr()` in isolation — the mismatch was an unsaved editor buffer, not a logic bug.*
- **What does this project demonstrate about you?** → *Judgment and measurement discipline, not line count — I can defend every design decision with a number and I know which problems not to solve.*
- **★ What's your latency?** → *Measured per stage (`eval/compare_retrievers.py --latency`): bi-encoder retrieval is production-grade — dense p95 106 ms, hybrid p95 249 ms, all under a quarter-second. The cross-encoder reranker is the bottleneck at ~25 s p50 on CPU — but that's the free/local-inference tax, not the architecture: it re-scores only the ~20-candidate pool and drops to sub-second on a GPU or a hosted reranker with no code change. If a demo needs speed I'd offer a hybrid-only "fast mode" (p95 ≤ 250 ms) at the 0.45-vs-0.75 answer@5 trade-off I can quantify.*
- **★ What number does a public deploy actually get?** → *Lower than 0.75, and I measured it. ~30% of my index is copyrighted (`local` tier, 11 docs) and can't be served, so on the deployable open+synthetic corpus the cascade scores 0.65 answer@5 / 0.75 doc@5 — two golden questions (q06, q12) lose their only source. I report the deployed number, not just the lab number.*
- **What would you do next?** → *RAGAS faithfulness scoring, a stronger embedder (e5-large/BGE-M3) to close the recall-gap misses, a smarter citation metric, the FastAPI + streaming serving layer, and a deployable frontend.*

### 5.11 — The tricky / gotcha questions (be ready)

- **★ Your hybrid (0.45) scored *lower* than lexical alone (0.50) at k=5 — explain that.** → *First, honesty about the statistics: at n=20 that answer@5 gap is a single question — within noise, and I say so. The evidence I actually trust is `doc@5`, where lexical beats hybrid 0.85 to 0.65 — a four-document gap. Equal-weight RRF was diluting the stronger lexical retriever's recall into the pool. That's a real effect, and exactly why I didn't stop at hybrid — the cross-encoder reranker is what decisively won, taking it to 0.75. I report the dip, and I distinguish noise from signal.*
- **Your model said "0,8 litro" but the reference said "1 litro" — is that wrong?** → *Both are correct — the chunk gives ~1 litre before losses and 0.8 after an 80% capture efficiency, and the model reported the more complete figure. It shows why substring-matching a generated answer against a reference is the wrong tool; you need faithfulness judgment.*
- **You only have 22 golden questions — isn't that too few?** → *It's small but hand-verified and stratified across five question types, which makes it a sharp diagnostic. It's a floor I'd grow — but a measured 22 beats an eyeballed zero, and it already caught real bugs.*
- **Isn't a 3B model too weak for a "real" system?** → *For a grounding-and-obedience task, no — 3B matched 7B on abstention on my (admittedly small) no-answer set. Capability is only worth paying for where the task actually uses it, and here retrieval already did the reasoning.*

### 5.12 — Engineering-hygiene lessons from the build (say these if asked about workflow)

- **"What runs is what's saved."** — A run silently executed the file on disk, not my unsaved editor buffer. Save before you run; verify with `repr()` in isolation, not by re-reading the tab.
- **A pipe hides the real exit code.** — `ollama pull … | tail -3` reported success because that's `tail`'s exit code, not the download's, which had actually failed on a DNS blip. Use `set -o pipefail`, or don't pipe when you need the true status.
- **Redirected output is buffered.** — `print()` to a file is block-buffered, so a long job's progress looks frozen; run Python with `-u` for live output.
- **Transient failures are transient.** — A DNS "no such host" mid-download was a blip; a clean retry resumed and succeeded. Confirm the real state (`ollama list`) before concluding the model's missing.
- **A long job must outlive its launcher.** — A 7-hour re-index launched from the editor session died when the machine slept; the process was a child of that session. The durable fix is *idempotency*: each doc's delete+insert commits atomically, so a `--resume` flag skips finished docs and the re-index runs in small, restartable batches. Design the long job to survive interruption *before* you start it.
- **CPU generation is prefill-bound, not just slow-per-token.** — A warm model still stalled the app: the ~2,800-token RAG prompt (five chunks) takes minutes to *prefill* on CPU before the first token, and abandoned retries queue up serially. The web server was healthy the entire time — the "network error" the UI showed was the model, not the service. Distinguish the layer that's slow from the layer that looks broken.

### 5.13 — Serving, testing & the craft layer

- **★ How did you take this from a script to a service?** → *Config-driven models (deploy swaps to a hosted model with an env var), a health-checked connection pool, input guardrails, and a startup warm-up. And I found the real bugs by running it: an idle-dropped pooled connection, a timeout that couldn't tell slow from wedged, and an ONNX cold-start — none of which a unit test surfaces.*
- **★ Why a connection pool with a health check?** → *A served API shouldn't open a fresh connection per request. But pooled connections go stale — Docker's proxy dropped mine while idle — so the pool validates each on checkout and replaces a dead one. Fail-fast when the DB is genuinely down, self-heal when a connection merely went stale.*
- **★ You added a reranker — how do you know it improved *ordering*, not just membership?** → *That's precisely the trap I avoided. `answer@5` is set membership — blind to a reorder within the top five. So I added `answer@1` and MRR, and the reranker jumped 0.15 → 0.75 and 0.25 → 0.75. It doesn't just fetch the answer, it puts it first.*
- **★ How do you know your eval *code* is correct?** → *I unit-test its assumptions. The load-bearing ones are invisible by inspection — a flipped sort comparator wouldn't crash, it'd silently invert ranking — so nine pytest cases pin the cross-encoder direction, the RRF math, the tokenizer, and the metric derivation.*
- **How do you keep the metric from drifting across scripts?** → *One definition. Every metric derives from a single `answer_rank` primitive in a module both eval drivers import; adding MRR was a one-file change, and the arc reproduced exactly through it — the proof the refactor was equivalent.*
- **Tell me about the front end.** → *A self-contained streaming SPA. Streaming cuts *perceived* latency, not wall-clock; every answer opens an evidence drawer; abstention has its own empty-basket state. The visual layer is a token-driven dual-theme OKLCH design system with a monospace face on every data surface — applied as a pure token remap, so it touched no logic.*
- **★ How would you ship this?** → *It's containerized — a two-stage `uv` build off a committed lockfile, a slim non-root runtime, a `/health` healthcheck, models cached in a volume. One image serves the API and, with an override command, runs the ingest. And the copyrighted tier can't ship: the Dockerfile never COPYs `data/raw`, so governance holds at the image layer, not just in config.*
- **★ Your re-index is ~7 hours — how do you make that survivable?** → *Idempotent, resumable ingest. Each document's delete+insert commits atomically, so a `--resume` flag skips finished docs and `--limit` runs the next N — the re-index becomes small restartable batches instead of one fragile uninterrupted pass. I built that after a session-bound run died to a machine sleep: the lesson was to make a long job outlive its launcher.*
- **How does the deploy stay legally clean at the UI?** → *The served corpus is open + synthetic only (`LOCAL_MODE` off never ingests the copyrighted tier), and the drawer badges every source with its governance tier — provenance is visible, not merely enforced.*
- **★ What would make you say this system is production-ready vs. a portfolio piece?** → *It's the middle ground, honestly: the retrieval layer and the serving hardening are production-grade and verified live; the CPU model is not, by design, but the config makes the swap to a hosted one a single env var. I'd name exactly that in an interview rather than overclaim — the honesty is the credibility.*

---

## Part 6 — The Mathematics, from first principles

> Parts 1–5 explain *what* the system does; Part 6 explains *why the math works*. Master this and you can defend any component at a whiteboard. Notation is plain-text and reproducible: `A·B` is a dot product, `‖A‖` a vector norm, `Σ` a sum.

### 6.1 — The embedding space: what a vector *is*

An embedding model is a function `f: text → ℝ^d` (`d = 384` for MiniLM, `1024` for e5-large). It's trained so that **semantically similar texts land at nearby points** — meaning *becomes geometry*. Directions in the space encode features; distance encodes similarity.

- **How that space gets built — contrastive learning.** Training pulls *positive* pairs (a question and its true passage) together and pushes *negatives* apart, minimizing a loss like InfoNCE: `L = −log[ exp(sim(q,p⁺)/τ) / Σ exp(sim(q,pᵢ)/τ) ]`. That objective is *why* paraphrases cluster and why a bi-encoder can retrieve by meaning.
- **Dimensionality is a real knob.** More dims = more representational capacity, but more storage (`d × 4 bytes/vector`) and slower distance math. 384 vs 1024 isn't free — I *measured* the tradeoff (§4.13): e5's 1024-dim doubled dense recall but cost 8× query latency.

### 6.2 — Cosine similarity, derived

The similarity metric behind dense retrieval:

```
cos(θ) = (A · B) / (‖A‖ · ‖B‖)
       = Σ aᵢbᵢ  /  (√Σaᵢ²  ·  √Σbᵢ²)
```

- **Range** `[−1, 1]`; `1` = same direction, `0` = orthogonal (unrelated).
- **Why cosine, not Euclidean or raw dot product?** Cosine is **magnitude-invariant** — it measures *direction only*. A one-sentence chunk and a full-page chunk about the same fact should score high; length shouldn't matter. Raw dot product would reward long vectors; Euclidean would penalize length differences.
- **The unifying identity:** for **unit-normalized** vectors, `‖A−B‖² = 2 − 2·cos(θ)`, so cosine, dot product, and Euclidean *rank identically* once you normalize. Many pipelines normalize and use the cheaper dot product.
- **In this project:** pgvector's `<=>` operator is **cosine *distance*** = `1 − cos(θ)` (smaller = closer). The HNSW index is built `vector_cosine_ops` so the graph is organized for exactly this metric.

### 6.3 — Inside the embedder: tokens, attention, pooling, and the truncation ceiling

1. **Tokenization** — text → subword tokens (WordPiece/BPE). A code like `WL-323` may fragment awkwardly, which is *why dense retrieval fumbles exact tokens* and lexical is needed.
2. **Self-attention** — the transformer core: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`. Every token's representation becomes a weighted blend of *all* tokens → context-awareness. It's `O(n²)` in sequence length `n`, which is **why a `max_seq_length` exists at all**.
3. **The truncation ceiling (the war-story math).** MiniLM's `max_seq_length = 128`. Tokens past position 128 are **dropped before the forward pass** — the model literally never computes on them. That's why two chunks sharing their first 128 tokens embed *identically* (cosine 1.0) regardless of their tails, and why ~half my answers were invisible to the dense vector (§ war story #7).
4. **Pooling** — the transformer emits one vector *per token*; you need one *per chunk*. **Mean pooling** averages them (MiniLM's current default — the exact `UserWarning` we saw); **CLS pooling** takes the special `[CLS]` token's vector. Mean pooling is why the whole (visible) sequence contributes.

### 6.4 — Bi-encoder vs cross-encoder: the complexity argument that *forces* the cascade

| | Bi-encoder (retrieval) | Cross-encoder (rerank) |
|---|---|---|
| Input | query and doc **separately** | `(query, doc)` **concatenated** |
| Cost | 1 forward for the query + `O(N·d)` cheap dot products | **`N` full forwards**, one per pair |
| Precompute? | ✅ docs embedded offline | ❌ query-dependent, can't cache |
| Accuracy | good recall, imperfect precision | high — full cross-attention between query & doc tokens |

The reranker reads the query and the chunk *together*, so every query token can attend to every doc token — far more discriminating than comparing two independently-made vectors. But that costs a full transformer forward **per candidate**, so you can only afford it on a small pool. **The math dictates the architecture:** bi-encoders over 10,330 chunks (cheap recall) → cross-encoder over 20 (expensive precision).

### 6.5 — Approximate nearest neighbor & HNSW

- **Exact NN** compares the query to all `N` vectors → `O(N·d)` per query. Fine at 10k, fatal at millions.
- **HNSW (Hierarchical Navigable Small World)** trades a little recall for huge speed:
  - A **multi-layer graph**. Top layers: few nodes, long-range links. Bottom layer: every node, short links. (Think a skip-list in graph form.)
  - **Search**: enter at the top, greedily hop toward the query, descend layer by layer → **~`O(log N)`**.
  - **Params**: `M` (edges per node — graph degree), `ef_construction` (build quality), `ef_search` (search breadth — the live **recall↔speed** knob).
- **Alternatives (and why HNSW):** *IVFFlat* clusters vectors and probes only the nearest cells (`O(N/nlist · probes)`) — less memory but needs a training step and gives worse recall at equal speed; *LSH* hashes near vectors into buckets (simple, weaker recall); *flat/brute* is exact but `O(N·d)`. pgvector offers HNSW and IVFFlat — **HNSW wins on recall/speed with no training step**, which is why the index is `USING hnsw`.

### 6.6 — Lexical retrieval: tf-idf → BM25 → Postgres FTS

- **tf-idf**: term weight = `tf × idf`, where `idf = log(N/df)` — reward terms frequent *in the doc* but rare *in the corpus*.
- **BM25**: tf-idf plus **term-frequency saturation** (`tf/(tf+k₁)` — diminishing returns) and **length normalization**. The modern lexical default.
- **Postgres FTS (what I actually use)**: `tsvector` (doc → sorted lexemes) matched against a `tsquery` via `@@`, ranked by **`ts_rank_cd`** (cover density — rewards query terms appearing *close together*). I use the **`'simple'` config**: no stemming, no stopwords → language-agnostic exact-token matching, correct for a mixed ES+EN corpus with codes like `WL-323`.
- **The AND/OR subtlety (war story #1):** `websearch_to_tsquery` **ANDs** terms, so a full-sentence question required *every* word in one chunk → matched nothing (lexical = 0.00). Fix: build an **OR of the content terms**.

### 6.7 — Rank fusion: Reciprocal Rank Fusion (RRF) derived

**The problem:** dense scores are cosine `∈ [0,1]`; lexical scores are `ts_rank_cd ∈ [0,∞)`. **Different scales — you can't just add them.**

**RRF's move:** throw away the scores, keep only the *ranks*:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))        # summed over each ranker r
```

- **Scale-free** by construction — only positions matter, so the cosine-vs-`ts_rank_cd` mismatch vanishes.
- **`k = 60`** is a **smoothing constant** (from Cormack et al., 2009). The `1/(k+rank)` shape means rank-1 scores `1/61`, rank-2 `1/62`… — small gaps at the top, so an item **both** lists rank moderately well beats an item only *one* list loves. Larger `k` = flatter weighting.
- **Alternatives (and why RRF):** *CombSUM/CombMNZ* sum normalized scores — but need a normalization scheme and are outlier-sensitive; *weighted linear fusion* is tunable but needs training data to set weights. **RRF is parameter-light and robust**, the reason it's the default.
- **The honest nuance I measured:** equal-weight RRF *diluted* the stronger lexical recall (`doc@5` 0.85 → 0.65) — a real fusion artifact, and exactly why I didn't stop at hybrid; the reranker is what decisively won.

### 6.8 — Evaluation metrics, formally

Let `rank` = the 1-based position of the answer chunk in the results (`∞` if absent). Everything derives from this one primitive (the `_scoring.answer_rank` design).

```
answer@k = (1/|Q|) · Σ  1[ rank ≤ k ]          # set membership: is it in the top-k?
RR       = 1 / rank        (0 if absent)        # reciprocal rank of one query
MRR      = (1/|Q|) · Σ RR                        # mean reciprocal rank
```

- **`answer@k` is set-membership** — binary, and *blind to ordering within the top-k*.
- **`MRR` rewards height**: rank-1 → `1.0`, rank-2 → `0.5`, rank-3 → `0.33`. This is **why MRR caught the reranker win**: a reordering that moves the answer from rank 3→1 doesn't change `answer@5` at all, but sends `answer@1` and MRR from `0.15/0.25 → 0.75`.
- **`document@k` vs `answer@k`**: the gap diagnoses a **chunking** problem (right doc, wrong chunk in top-k) vs a **retrieval** problem (wrong doc entirely).
- **`nDCG`** (the IR standard for *graded* relevance) discounts by `1/log₂(rank+1)`; I use *binary* relevance, so MRR/answer@k suffice — but naming nDCG shows you know the general tool.

### 6.9 — The reranker's score, and small-sample statistics

- The cross-encoder emits a **logit** — higher = more relevant. I sort `reverse=True`, and that assumption is **unit-tested** (a relevant pair scored `+1.25`, an irrelevant one `−3.72`): a flipped comparator wouldn't crash, it would *silently* invert ranking.
- **Statistical honesty (n = 20):** one question is `0.05` of `answer@5`. A Wald 95% CI on a proportion `p̂ = 0.75, n = 20` is roughly `±0.19` — **wide**. So I treat single-question cross-config differences as **directional**, trust multi-question gaps (like a 4-doc `doc@5` swing) as **signal**, and say so out loud. *Reporting the confidence interval you can't shrink is the statistics credential.*

---

## Part 7 — System Design & Architecture

> How the pieces fit, why the boundaries fall where they do, and how it scales. This is the "design the system on a whiteboard" section.

### 7.1 — The two-phase architecture (the load-bearing decision)

Everything hangs on splitting **offline indexing** from **online querying**:

- **Offline (build the index, rarely):** expensive, batched, *idempotent*. Embedding 10,330 chunks is minutes-to-hours of compute — you pay it once and amortize.
- **Online (answer a query, every time):** cheap, low-latency, per-request. Retrieval is sub-second because the expensive work already happened.

If you blur this line — e.g., embed documents at query time — you've built something that can't scale. Keeping it sharp is *why* the system can claim to scale to millions of docs with barely a change to the online path.

### 7.2 — The ingestion pipeline (offline), and why it's idempotent

`manifest → extract (PyMuPDF) → normalize (de-hyphenate, strip repeated headers) → chunk (512-tok / 64 overlap) → embed (batched) → store (pgvector + tsvector)`

- **Idempotency by design:** each document's `wipe_chunks + insert` runs in **one atomic transaction**, so a doc is all-or-nothing. That single property is what makes the `--resume` flag *sound* — it can trust that a doc with any chunks is fully done, so a 7-hour re-index becomes small restartable batches (the resumable-ingest work).
- **Governance is a pipeline gate:** `settings.tiers()` decides which manifests load; with `LOCAL_MODE=0` the copyrighted tier is *never read*, so a public index *physically cannot contain it*.

### 7.3 — Data model & indexing (the schema decisions)

- **Two tables:** `documents` (one row per source — `tier`, `source_file`, `title`) ⟷ `chunks` (one row per chunk — `text`, `embedding VECTOR(n)`, `tsv`, plus **denormalized** `tier/lang/topic/page`).
- **Why denormalize tier onto chunks?** So retrieval can filter (`WHERE tier = ANY(%s)`) on the **hot path without a JOIN**. A deliberate normalization/latency tradeoff.
- **Three index types, one per access pattern:** `HNSW` on `embedding` (vector search), `GIN` on `tsv` (keyword search), `btree` on `tier` (governance filter).
- **The coupling to remember:** `VECTOR(n)` *must* equal `embed_dim`. Swapping MiniLM(384)→e5(1024) is therefore a **schema migration**, not a config flip — the friction I measured in §4.13.

### 7.4 — The serving layer (online), request by request

`/stats`, `/retrieve`: `validate → Depends(get_conn) checks out a pooled connection → work → dependency RELEASES conn → response`. `/ask/stream` is different on purpose (see below): `StreamingResponse(gen())` returns *immediately* → headers flush → `gen()` itself checks out a connection, retrieves, releases it, then streams generation — connection checkout happens *inside* the response body, not before it.

- **FastAPI + uvicorn (ASGI):** async gives real streaming; pydantic gives validation at the edge.
- **`psycopg_pool.AsyncConnectionPool`, injected via DI:** `min_size=5, max_size=20, timeout=30.0`, opened once in the app's `lifespan`. `/stats` and `/retrieve` take a connection through a `Depends(get_conn)` generator dependency — a route asks for a connection, it never touches the pool object. Checkout doesn't block a worker thread (the pool awaits a free connection), and each connection is validated on checkout, **self-healing** one the Docker proxy dropped while idle. *Fail fast if the DB is truly down; self-heal if a connection merely went stale.*
- **`/ask/stream` deliberately does NOT use `Depends(get_conn)`.** The original design did — retrieval ran in the route handler, using the injected connection, and only *then* was `StreamingResponse` constructed and returned. That's a real bug, found live: Starlette sends the ASGI response's headers as soon as the response object exists, *before* it iterates the body — so with retrieval running first, the client got zero bytes, not even headers, for the whole CPU-bound retrieval phase (minutes). Browsers apply a much stricter time-to-first-byte timeout than they do to an already-open, slow stream, and that's what was firing. Fix: `request.app.state.pool.connection()` is checked out *manually*, inside the response's own async generator — so `StreamingResponse` returns instantly (headers flush immediately), and the connection is still released before the connection-free generation phase, just via an explicit `async with` inside the generator instead of FastAPI's dependency teardown.
- **Generation moved onto `ollama.AsyncClient`** as a consequence: once `/ask/stream`'s generator needed `await` for retrieval, it had to become `async def`, and an async generator can't contain a blocking sync call without stalling the event loop for every other concurrent request. The CLI/eval paths still use the sync client correctly, on their own thread.
- **Real, backend-driven progress feedback:** `/ask/stream` emits `{"type":"progress","stage":...}` frames at real phase boundaries (translating / reranking / generating), not a fabricated percentage — the reranker's own batching means there's no honest finer-grained signal without breaking the single forward pass that makes it fast.
- **A retrieval-only route:** `/retrieve` shares the same retrieval implementation as `/ask/stream` but never calls the LLM, so an eval harness or a monitor can measure retrieval latency in isolation instead of it being buried inside generation time.
- **A real, Windows-specific gotcha, found by running it:** `asyncio`'s default event loop on Windows (`ProactorEventLoop`) cannot run the async DB driver at all — not a performance issue, a hard incompatibility. The fix is a platform-conditional event-loop policy, applied before the server starts.
- **Streaming protocol — NDJSON:** newline-delimited JSON over one HTTP response: a `meta` frame (sources) first, then `token` frames, then `done`. *Why not SSE?* SSE works but adds event-framing ceremony a `fetch`-stream client doesn't need. *Why not WebSocket?* It's bidirectional; this is one-way server-push — WebSocket is overkill.

### 7.5 — Concurrency, scaling, and the true bottleneck

- **Where it blocks:** embed + rerank are CPU/GPU-bound `torch` work, so both run on a single dedicated worker thread (`agroteca.torch_runtime`) rather than `asyncio.to_thread`'s ad-hoc pool — a deliberate choice against thread-pool oversubscription, though the measured 20× CPU slowdown that prompted it turned out to have a different root cause (§4.15): unoptimized inference on realistic-length input, not threading.
- **Scaling reads:** the serving process is **stateless** → scale horizontally behind a load balancer; the shared state is the DB + vector index.
- **Scaling to millions of docs:** retrieval already scales — HNSW is `~log N`, the reranker *always* sees a fixed 20-pool, and embedding is offline + batched. **The online path barely changes** — that's the architecture's scalability claim, and it's honest.
- **The real bottleneck is now the reranker itself, not just generation** — and it's device-bound: on GPU, sub-second; on CPU, ~500 s for a 20-candidate pool, because the `torch`/`transformers` backend (needed for real GPU inference) trades away the ONNX backend's quantization and graph optimization. Generation is separately **prefill-bound** on CPU: a ~2,800-token RAG prompt must be processed before the first token. Levers, measured not guessed: GPU for both (one env var), fewer chunks, a smaller/quantized model, response caching.

### 7.6 — Config-driven design (12-factor) & the deploy swap

Every knob lives in one typed `Settings` object, overridable by `AGROTECA_*` env vars. The unservable local-CPU model becomes a hosted GPU model with **one env var, zero code change** — config strictly separated from code, so the *same artifact* deploys across environments.

### 7.7 — Resilience: each guard maps to a failure I *observed by running it*

| Guard | Failure it closes |
|---|---|
| Pool check-on-checkout | idle-dropped connection → 500 |
| Split connect/read timeouts | a slow-but-correct CPU answer killed as if "wedged" |
| Startup warm-up inference, run in-process | cold-start made the *first* request slower than every later one |
| `require_cuda()` fail-fast at startup | a "GPU microservice" silently serving on CPU without anyone noticing |
| Bounded input length | a 1 MB payload reaching the FTS regex |
| Windows event-loop policy | the async DB driver silently unable to run at all under the default loop |
| Retrieval moved inside the response generator | headers not flushing until retrieval finished → a browser's stricter TTFB timeout firing before the backend ever responded (found by a real user, not my own tests) |
| `gen_timeout` raised to match measured CPU cost | a legitimately-working generation aborted because prefill now legitimately exceeds the old 300s assumption |

### 7.8 — Containerization & governance at the image layer

A **multi-stage `uv` + CUDA build**: a dependency layer keyed on the committed `uv.lock` → a dedicated `model-bake` stage that downloads and warms both models at *build* time (so a load-time bug surfaces in CI, not in production) → a slim, **non-root**, health-checked runtime with those weights baked directly into the image. `HF_HUB_OFFLINE=1` at runtime means the container has no network dependency for its own inference, only for reaching Postgres and the LLM endpoint — both externally configured. The image's `ENTRYPOINT` runs the same warmup as a fail-fast GPU preflight before `CMD` starts. One image **serves *and* ingests**, and `data/raw` is *never* `COPY`ed — so the copyrighted tier can't ship even by accident. **Governance holds at the image boundary, not just in runtime config.**

The app is also wired into `docker-compose.yml` alongside the database (GPU-scheduled via `deploy.resources.reservations.devices`), so `docker compose up -d` is a real, single-command way to stand up the full stack — checked with `docker compose config` before being documented as a fact, not just asserted as one.

---

## Part 8 — Python, Libraries & the Choices Behind Them

> Every dependency was a *choice* with alternatives. This is the defense of each — the pros/cons an interviewer probes with *"why X and not Y?"* The pattern to notice: **most choices are downstream of two hard constraints — run locally (governance: copyrighted chunks never leave the machine) and run on CPU (cost).**

### 8.1 — Embeddings & reranking → `sentence-transformers` / `transformers` on `torch` (was `fastembed`)
- **Originally chosen:** fastembed (Qdrant's ONNX-Runtime embedding lib) — no PyTorch, CPU-optimized, ships quantized weights. Right call for a CPU-only, PyTorch-free serving story, and it shipped that way for most of the project.
- **Revisited, deliberately, when the goal changed:** real `device='cuda'` GPU inference — `.to(device)`, `torch.inference_mode()` — isn't expressible through fastembed's ONNX-Runtime API at all. Getting it meant swapping the inference backend, not adding a flag.
- **The cost I measured, not assumed:** on CPU, the same rerank call went from ~25 s (fastembed/ONNX, quantized + graph-optimized) to ~500 s (`transformers`, eager-mode, unquantized) — a ~20× regression I chased down to its real cause rather than papering over (§4.15). CPU is now a correctness fallback, not a usable free tier, and I say so in the deploy docs rather than let a reader discover it.
- **Rejected (again, with the new constraint):** staying on fastembed (no GPU path at all — ruled out by the new requirement) · a hosted embeddings/reranking API (data-leaves-the-machine governance violation, same as before).
- **Why the swap was still the right call:** the project's constraints changed — the requirement became "prove real GPU inference," not "stay CPU-only" — and a design that was correct for the old constraint stopped being correct for the new one. The honest move was to make the trade explicit (README, DEPLOY.md) rather than let the regression hide behind "should be faster on a real GPU."

### 8.2 — Vector store → **Postgres + `pgvector`**
- **Pros:** **one store** for vectors *and* keyword (FTS) *and* metadata *and* the governance filter (`WHERE tier`); ACID; production-credible; no extra service.
- **Cons:** not a billion-scale vector-native engine; HNSW memory/tuning is manual.
- **Rejected:** *Pinecone* (managed & scalable, but `$` + hosted + no SQL governance) · *Weaviate/Qdrant/Milvus* (powerful vector-native, but *another service* and vectors split from metadata) · *FAISS* (blazing library, but no persistence/filtering/SQL — you'd build a DB around it) · *Chroma* (great prototyping DX, less production-hardened).
- **Why:** the **governance tiers live in SQL**, so keeping vectors + metadata + FTS in one ACID store is simpler *and* more defensible than gluing a vector DB to a metadata DB.

### 8.3 — LLM serving → `Ollama`
- **Pros:** trivial local serving + model management, GGUF-quantized models, an OpenAI-ish API, runs on CPU.
- **Cons:** not a high-throughput multi-tenant server; slower than vLLM on GPU.
- **Rejected:** *vLLM* (GPU throughput king — PagedAttention, continuous batching — but GPU-centric, heavy) · *llama.cpp direct* (max control, more setup) · *TGI* (HuggingFace production GPU serving) · *OpenAI/Anthropic API* (best quality, but `$` + data leaves the machine).
- **Why:** local + offline + zero-cost + copyrighted context stays on-device. The config swaps to any Ollama-compatible host — or a small contained edit to an API client — for production speed.

### 8.4 — Web framework → `FastAPI`
- **Pros:** async ASGI (real streaming), built-in pydantic validation, auto OpenAPI docs, `StreamingResponse` for NDJSON, minimal boilerplate.
- **Cons:** less batteries-included than Django (no ORM/admin); younger.
- **Rejected:** *Flask* (simple, but WSGI-sync — streaming is awkward, no built-in validation) · *Django* (batteries + ORM + admin, but heavy and sync-first for a pure API) · *Litestar* (very close, smaller community).
- **Why:** async streaming + edge validation are *exactly* the serving needs; Django's weight buys nothing here.

### 8.5 — DB driver → `psycopg 3` + `psycopg_pool`
- **Pros:** modern API, a **native pool with check-on-checkout**, clean pgvector integration (`register_vector`/`register_vector_async`), and — the one that mattered most for serving — **the same library ships both a sync and an async pool** (`ConnectionPool` for CLI/eval/ingest scripts, `AsyncConnectionPool` for the FastAPI serving path), so I didn't have to run two different database libraries to get both a scriptable CLI and a non-blocking server.
- **Rejected:** *psycopg2* (mature but older API, bolt-on pooling) · *asyncpg* (fastest async, but async-only, different API) · *SQLAlchemy* (ORM/Core with migrations — but it **hides the SQL I want to write** for vector ops and tier filters).
- **Why:** I *want* explicit SQL (vector operators, governance filters); an ORM would obscure the very thing that makes the design legible — and psycopg3's pool health-check is the resilience feature I lean on. The async pool's DI wiring (`Depends(get_conn)`) is the same discipline applied to concurrency: routes ask for a connection, they don't manage one.
- **A platform lesson that cost real debugging time:** `AsyncConnectionPool` cannot run under Windows's default `ProactorEventLoop` — a hard incompatibility in the async driver, not a config knob. Found by running the server, not by reading documentation in advance; fixed with a platform-conditional event-loop policy.

### 8.6 — Config → `pydantic-settings`
- **Pros:** typed settings, `.env` + env-var override, validation, one `Settings` object → the 12-factor deploy swap.
- **Rejected:** *raw `os.environ`* (untyped, fragile) · *configparser* (ini, no types) · *dynaconf* (powerful but heavier than needed).

### 8.7 — Packaging & tooling → `uv` + `hatchling`
- **`uv`** (Rust): extremely fast installs, a real lockfile, venv + Python management. **Rejected:** *pip* (slow, no lockfile by default), *poetry* (good, slower/heavier), *conda* (data-science, heavy).
- **Why it matters here:** the Docker build's reproducibility depends on `uv sync --frozen` against the committed `uv.lock`.
- **`hatchling`** (modern PEP-517 backend) over *setuptools* (legacy, more config).

### 8.8 — Testing → `pytest`
- **Pros:** fixtures, `parametrize`, plain-`assert` introspection, `pythonpath` config (so tests import the single-source `_scoring`).
- **Rejected:** *unittest* (stdlib, but class-based boilerplate).

### 8.9 — The reranker → `jinaai/jina-reranker-v2-base-multilingual`
- **Pros:** **multilingual** (ES + EN — a hard requirement), strong quality, small.
- **Rejected:** *bge-reranker* (excellent, but strongest in EN/zh) · *ms-marco cross-encoders* (**EN-only → fails the cross-lingual pairs**) · *cohere-rerank* (great, but API `$` + data leaves) · *mixedbread* rerankers (viable alternative).
- **Why:** the golden set is bilingual, so multilingual reranking isn't a preference — it's a **correctness requirement**.
- **The security detail worth naming:** this model loads with `trust_remote_code=True` — it ships a custom architecture, which means loading it executes Python from the model's own HF repository, not just weights. That's a real, if narrow, supply-chain surface an ONNX-converted model doesn't have. I pin the load to a specific, reviewed commit hash rather than "latest" — a later push to that repo can't silently change what code runs in my server. Naming the risk and bounding it is the point; pretending `trust_remote_code=True` is free would be the actual mistake.

### 8.10 — The Python vocabulary actually used
Generators (`yield` — the streaming backbone, and FastAPI's dependency-teardown mechanism) · `lru_cache` (the model singleton) · context managers (`with`/`async with pool.connection()`) · `asyncio`: `async`/`await`, `Depends()`-based DI, `run_in_executor` against a dedicated `ThreadPoolExecutor` (moving CPU-bound `torch` work off the event loop without oversubscribing it) · comprehensions & `*`-unpacking · type hints + pydantic models · `functools`/`argparse` · decorators (FastAPI routes) · `if __name__ == "__main__"` (module-as-script). Each is a small thing; together they're the difference between a script and a system.

---

## Part 9 — The Concepts to Master (your syllabus)

> The checklist to *own* this subject, not just this repo. 🟢 solid · 🟡 review before an interview. Each line is "the concept — why it matters here."

**Information Retrieval**
- 🟢 Precision / recall / F1; MRR; nDCG; MAP — *the language of "did it find the answer, and how high?"*
- 🟢 tf-idf, BM25, inverted indexes — *the lexical half of hybrid.*
- 🟢 Vocabulary mismatch — *the reason dense **and** lexical both exist.*

**Representation learning**
- 🟡 The transformer & self-attention — *what makes an embedding contextual.*
- 🟡 Contrastive learning / InfoNCE — *how embedders learn that paraphrases are "close."*
- 🟢 Bi-encoder vs cross-encoder — *the complexity argument behind the cascade.*
- 🟢 Tokenization (BPE/WordPiece) & context windows — *the truncation ceiling.*

**Vector search**
- 🟢 Cosine / dot / L2 & normalization — *the metric and why it's magnitude-invariant.*
- 🟡 ANN: HNSW, IVF, LSH; the recall↔speed tradeoff — *how search scales past brute force.*
- 🔴 Quantization (Product/Scalar) — *the next lever at real scale; worth learning.*

**RAG**
- 🟢 The retrieve→augment→generate paradigm; RAG vs fine-tuning — *cheap-to-update, citable, less hallucination.*
- 🟢 Chunking strategies — *the recall/precision knob I measured both ways.*
- 🟢 Rank fusion (RRF) — *scale-free combination of two rankers.*
- 🟢 Grounding / citation / abstention — *the three anti-hallucination disciplines.*
- 🟢 Prompt injection & the system-role defense — *untrusted retrieved text can't override the rules.*

**Evaluation**
- 🟢 Golden sets & stratification — *measure, don't eyeball.*
- 🟢 Retrieval metrics vs generation metrics — *membership check vs semantic judgment.*
- 🟡 LLM-as-judge / RAGAS (faithfulness, context precision/recall) — *the semantic layer a substring metric can't reach.*
- 🟢 Small-sample statistics & "directional" honesty — *knowing what n=20 can and can't claim.*

**Systems & MLOps**
- 🟢 Offline/online split; idempotent, resumable pipelines — *the scalability & survivability backbone.*
- 🟢 Connection pooling, streaming, concurrency — *serving strangers, not a notebook.*
- 🟢 Async I/O vs. CPU-bound work: `asyncio`, dependency-injected pooled resources, `run_in_executor` off a dedicated thread — *why a non-blocking DB pool and an unbounded thread pool for GPU/CPU inference are two different problems with two different fixes.*
- 🟡 GPU serving basics: device placement, warmup/cold-start, fail-fast health gates — *the difference between "loaded" and "warm," and between "slow" and "misconfigured."*
- 🟡 Inference backend tradeoffs: eager `torch` vs. a quantized, graph-optimized runtime (ONNX/TensorRT) — *a measured 20× CPU cost, not a vibe.*
- 🟢 Config-driven deploy & containerization — *the same artifact everywhere; governance at the image layer.*
- 🔴 Observability (tracing, metrics, eval-in-prod) — *the honest next gap.*

**Generation**
- 🟡 Decoding (greedy/sampling/temperature), `num_predict` — *controlling the output.*
- 🟡 Quantization (GGUF), local vs hosted serving — *the cost/speed frontier.*
- 🟢 Prefill vs decode cost — *why CPU generation stalls on a big RAG prompt.*

**The frontiers to grow into (ties to the career roadmap)**
- 🔴 RAGAS + semantic eval · **query transformation / translation** (my measured cross-lingual lever) · agentic & multi-hop RAG · fine-tuning an embedder on domain data. *These are where "next steps" answers become real projects.*

> **How to use Part 9:** turn every 🟡/🔴 into a one-paragraph note in your own words, then a tiny experiment. That's the same *measure-it-yourself* loop that built the rest of this document — the only study method this project actually endorses.
