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


async def run_torch(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking torch call (embed_query, embed_passages, the reranker) on the single
    dedicated worker thread instead of asyncio.to_thread's ad-hoc pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, partial(func, *args, **kwargs))
