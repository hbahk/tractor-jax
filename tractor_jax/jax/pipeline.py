"""CPU-build <-> GPU-solve prefetch pipelining.

Every tractor-jax consumer alternates a Python/numpy-heavy CPU build
(``extract_model_data`` / ``build_padded_batches``) with a jitted GPU solve
per work unit (tile / cutout / exposure). Run serially, the GPU idles during
the build and vice versa. :func:`prefetch_pipeline` overlaps them with a
bounded look-ahead::

    for built in prefetch_pipeline(tiles, build_tile, depth=2):
        fut = solver(*built)            # jitted; dispatches asynchronously
        futs.append(fut)                # do NOT np.asarray here
    results = list(lagged_collect(futs, lag=1))

JAX's async dispatch supplies the other half of the overlap: the jitted call
returns device futures immediately, and materializing them one step behind
dispatch (:func:`lagged_collect`) keeps at most ``lag`` result buffers alive
on the device while the host builds ahead.

The engine owns overlap-correctness and backpressure only; everything inside
``build_fn`` and the ``items`` descriptors (FITS IO, WCS, catalog slicing,
tile geometry) stays user code.
"""
import itertools
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import jax

__all__ = ["prefetch_pipeline", "lagged_collect"]


def _process_worker_init():
    # BLAS oversubscription in workers is a 30-100x trap on loaded nodes;
    # workers must also never initialize CUDA (spawn contexts honor these,
    # fork inherits the parent's already-loaded libs — build_fn should be
    # numpy-only either way).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def prefetch_pipeline(items, build_fn, *, depth=2, executor="thread"):
    """Yield ``build_fn(item)`` for each item, computed ahead of the consumer.

    Parameters
    ----------
    items : iterable
        Work descriptors (paths, index tuples, ... — NOT built arrays).
        Consumed lazily; may be a generator.
    build_fn : callable
        ``item -> built``. Pure CPU; runs in the worker. In ``"process"``
        mode it must be picklable (module-level) and numpy-only (workers
        must not touch JAX/CUDA).
    depth : int
        Maximum builds submitted but not yet yielded (bounded look-ahead =
        host-memory backpressure; steady state needs 1 in flight + 1 ready,
        so 2-3 is plenty).
    executor : {"thread", "process", "sync"}
        ``"thread"`` (default) overlaps the build with GPU/JAX work that
        releases the GIL; builds do not overlap each other. ``"process"``
        additionally overlaps builds with builds (``depth`` workers; items
        and built pytrees are pickled — tens of MB/item is fine, measure
        beyond that). ``"sync"`` disables overlap (debugging; identical
        ordering/error semantics).

    Yields
    ------
    The built objects, strictly in ``items`` order. An exception raised by
    ``build_fn`` re-raises here at the corresponding yield, after which the
    pipeline shuts down. Abandoning the generator early (``break`` /
    ``itertools.islice``) cancels pending builds and joins the workers.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if executor == "sync":
        for item in items:
            yield build_fn(item)
        return
    if executor == "thread":
        pool = ThreadPoolExecutor(max_workers=1,
                                  thread_name_prefix="tj-prefetch")
    elif executor == "process":
        pool = ProcessPoolExecutor(max_workers=depth,
                                   initializer=_process_worker_init)
    else:
        raise ValueError(f"unknown executor {executor!r}; expected "
                         "'thread', 'process' or 'sync'")

    it = iter(items)
    window = deque()
    try:
        for item in itertools.islice(it, depth):
            window.append(pool.submit(build_fn, item))
        while window:
            result = window.popleft().result()   # re-raises build errors
            # Top up BEFORE yielding so the worker builds while the
            # consumer processes `result`.
            for item in itertools.islice(it, 1):
                window.append(pool.submit(build_fn, item))
            yield result
    finally:
        for fut in window:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def lagged_collect(futs_iter, lag=1):
    """Materialize device futures ``lag`` steps behind their dispatch.

    Yields ``tree_map(np.asarray, fut)`` for each element of ``futs_iter``
    (arrays or pytrees of arrays), keeping at most ``lag`` unmaterialized
    results alive — the host stays ahead of the device by ``lag`` solves
    without accumulating every result buffer in device memory.
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    buf = deque()
    for fut in futs_iter:
        buf.append(fut)
        if len(buf) > lag:
            yield jax.tree_util.tree_map(np.asarray, buf.popleft())
    while buf:
        yield jax.tree_util.tree_map(np.asarray, buf.popleft())
