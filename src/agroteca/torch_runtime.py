"""A single, persistent worker thread for every torch CPU-bound inference call in the serving
path (the embedder and the reranker).

Why this exists: PyTorch's CPU backend (its OpenMP/MKL intra-op thread pool) initializes lazily
on whichever OS thread first runs a torch op. `asyncio.to_thread()` draws from a default
executor of up to `min(32, os.cpu_count() + 4)` threads, so successive calls can land on
DIFFERENT worker threads -- each one separately (re)initializing its own OpenMP pool. On a
12-core box that can mean over a dozen OpenMP pools competing for 12 physical cores at once.
Measured in this repo: a rerank call that takes ~10s run directly on the main thread took over
600s when dispatched through `asyncio.to_thread`. Routing every torch call through ONE
dedicated, reused thread avoids the oversubscription entirely -- torch's thread pool
initializes exactly once, in exactly one thread, for the life of the process.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, TypeVar

T = TypeVar("T")

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torch-worker")

# Admission control, not cancellation. A cancelled asyncio task only detaches from the
# run_in_executor future -- it does NOT stop the OS thread underneath it (Python threads
# aren't preemptible). So a client that disconnects mid-rerank leaves its call running to
# completion on the one dedicated worker regardless; on AGROTECA_DEVICE=cpu that's measured at
# ~500s (see eval/eval_pipeline.py's --timeout default). Without a cap, every OTHER request's
# run_torch() call queues silently behind however much orphaned work has piled up -- a single
# burst of disconnects (or just real concurrent load) can starve every subsequent user for
# minutes with no visible error. Capping in-flight admissions turns that into a fast, explicit
# rejection instead of an invisible unbounded wait. Sized at 2 (not 1) so two genuinely
# concurrent requests still queue normally -- a third arrival while two are already ahead of it
# is the actual overload signal.
_MAX_INFLIGHT = 2
_admission: asyncio.Semaphore | None = None


class TorchWorkerBusy(RuntimeError):
    """Raised when the dedicated torch worker already has _MAX_INFLIGHT calls queued/running.
    Callers (the API routes) are expected to turn this into an explicit failure -- a 503 for
    /retrieve, an NDJSON error frame for /ask/stream -- rather than let the caller queue
    indefinitely."""


def _semaphore() -> asyncio.Semaphore:
    # Constructed lazily, inside the running loop that will actually use it, rather than at
    # import time -- avoids binding to whatever loop happens to exist first (e.g. a test
    # runner's loop) and then being reused, incorrectly, from uvicorn's.
    global _admission
    if _admission is None:
        _admission = asyncio.Semaphore(_MAX_INFLIGHT)
    return _admission


async def run_torch(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking torch call (embed_query, embed_passages, the reranker) on the single
    dedicated worker thread instead of asyncio.to_thread's ad-hoc pool.

    Raises TorchWorkerBusy immediately, without queuing, if _MAX_INFLIGHT calls are already
    queued or running -- see the module docstring above for why unbounded queuing here is the
    real bug a client disconnect exposes."""
    sem = _semaphore()
    if sem.locked():
        raise TorchWorkerBusy(
            f"torch worker saturated ({_MAX_INFLIGHT} calls already queued/running)"
        )
    async with sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, partial(func, *args, **kwargs))
