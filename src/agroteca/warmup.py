"""GPU warmup: load the reranker + embedder onto `settings.device` and run one real forward
pass through each, so CUDA context init, cuDNN kernel autotuning, and VRAM allocation are paid
once at process start -- not on the first real user request.

IMPORTANT process-boundary note: `load_and_warm()` only warms the process it runs IN. If this
module is executed standalone (`python -m agroteca.warmup`) as a step before `uvicorn` in the
Dockerfile ENTRYPOINT, its warmed CUDA context does NOT carry over to the separate uvicorn
process that starts afterward -- OS processes don't share GPU memory or loaded model weights.
The warmup that actually matters for the first request's latency is the identical
`load_and_warm()` call made inside `agroteca.api`'s lifespan, IN the server process, before it
starts accepting connections. This module's standalone entrypoint exists for a different,
legitimate job: a fail-fast preflight check (does this container actually have a usable CUDA
device?) that aborts the container before it even tries to bind a port, rather than starting a
"GPU microservice" that silently serves on the CPU (see `require_cuda`).
"""
from agroteca.config import settings
from agroteca.ingest.embed import embed_passages, embed_query
from agroteca.retrieve.rerank import warm as warm_reranker

WARMUP_BATCH = [f"warmup sequence {i}" for i in range(1, settings.warmup_batch_size + 1)]


def require_cuda() -> None:
    """Fail loud, not slow: a GPU microservice that silently serves on CPU when the container
    lost its GPU (bad --gpus flag, missing nvidia-container-toolkit, wrong base image) is a much
    worse failure mode than refusing to start."""
    import torch
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "AGROTECA_DEVICE=cuda but no CUDA device is visible to this process -- check "
            "`docker run --gpus`, the nvidia-container-toolkit install, and the base image's "
            "CUDA runtime. Set AGROTECA_DEVICE=cpu explicitly for CPU-only local iteration."
        )


def load_and_warm() -> None:
    """Load both models onto `settings.resolved_device` and force one real forward pass through
    each -- this is what actually pays the CUDA-context/cuDNN-autotune/VRAM-allocation cost.
    Must run in the SAME process that will serve requests (see module docstring)."""
    if settings.hosted:
        # No GPU and no torch here: load the ONNX embedder once (downloads the model on a cold
        # instance) so the first real question doesn't pay for it. The reranker is a remote API.
        embed_query("warmup")
        return
    import torch
    require_cuda()
    embed_passages(WARMUP_BATCH)   # dense embedder: init CUDA context, allocate its VRAM
    warm_reranker()                # cross-encoder: same, for the rerank model
    if settings.resolved_device == "cuda":
        torch.cuda.synchronize()   # block until the warmup kernels actually finish, not just enqueue


if __name__ == "__main__":
    # Standalone preflight (invoked by the Dockerfile ENTRYPOINT before uvicorn). This also
    # warms *this* process, mainly as a side effect that surfaces a broken model/weights load
    # at container start instead of on a user's first request -- not to pre-warm the uvicorn
    # process that starts next (it can't; see module docstring).
    load_and_warm()
    print(f"warmup: {settings.rerank_model} + {settings.embed_model} loaded and warmed on {settings.resolved_device}")
