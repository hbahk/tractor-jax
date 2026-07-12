"""Tests for tractor_jax.jax.pipeline.prefetch_pipeline / lagged_collect.

Design contract (notebooks/research_notes/2026-07-12-prefetch-pipeline-design.md
in proj-spherex-gpupipe): ordering preserved across executors, bounded
look-ahead (backpressure), build errors re-raise at the consumer's yield,
early abandonment cancels pending work and joins workers, thread mode
actually overlaps a GIL-releasing consumer with the build.

Run in the `spherex` conda env:  pytest tests/test_pipeline.py -q
"""
import itertools
import threading
import time

import numpy as np
import pytest

from tractor_jax.jax.pipeline import lagged_collect, prefetch_pipeline


def build_square(i):
    """Module-level (picklable) deterministic build."""
    rng = np.random.default_rng(i)
    return rng.normal(size=8) ** 2


def build_raise_at_3(i):
    """Module-level (picklable) build that fails on item 3."""
    if i == 3:
        raise RuntimeError("boom at 3")
    return i


ITEMS = list(range(12))


# --------------------------------------------------------------------------- #
# equivalence + ordering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("executor", ["sync", "thread", "process"])
def test_executors_bit_identical(executor):
    ref = [build_square(i) for i in ITEMS]
    got = list(prefetch_pipeline(ITEMS, build_square, depth=3,
                                 executor=executor))
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert np.array_equal(a, b)


def test_ordering_with_variable_build_times():
    def build(i):
        time.sleep(0.03 if i % 3 == 0 else 0.001)
        return i * 10
    got = list(prefetch_pipeline(range(10), build, depth=3,
                                 executor="thread"))
    assert got == [i * 10 for i in range(10)]


def test_items_may_be_a_generator():
    got = list(prefetch_pipeline((i for i in range(5)), build_square,
                                 depth=2, executor="thread"))
    assert len(got) == 5


# --------------------------------------------------------------------------- #
# backpressure
# --------------------------------------------------------------------------- #
def test_bounded_lookahead():
    built = []

    def build(i):
        built.append(i)
        return i

    gen = prefetch_pipeline(range(10), build, depth=2, executor="thread")
    consumed = 0
    for _ in gen:
        consumed += 1
        time.sleep(0.02)          # slow consumer
        # never more than `depth` builds ahead of consumption
        assert len(built) - consumed <= 2
    assert consumed == 10


# --------------------------------------------------------------------------- #
# error propagation + early abandonment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("executor", ["sync", "thread", "process"])
def test_build_error_reraises_at_yield(executor):
    gen = prefetch_pipeline(range(6), build_raise_at_3, depth=2,
                            executor=executor)
    got = [next(gen), next(gen), next(gen)]
    assert got == [0, 1, 2]
    with pytest.raises(RuntimeError, match="boom at 3"):
        next(gen)


def test_early_abandon_joins_workers():
    before = threading.active_count()
    got = list(itertools.islice(
        prefetch_pipeline(range(100), build_square, depth=3,
                          executor="thread"), 4))
    assert len(got) == 4
    # islice closed the generator -> finally joined the worker thread
    deadline = time.time() + 5.0
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= before


def test_bad_args():
    with pytest.raises(ValueError, match="depth"):
        list(prefetch_pipeline([1], build_square, depth=0))
    with pytest.raises(ValueError, match="executor"):
        list(prefetch_pipeline([1], build_square, executor="fiber"))
    with pytest.raises(ValueError, match="lag"):
        list(lagged_collect(iter([]), lag=-1))


# --------------------------------------------------------------------------- #
# overlap actually happens (thread mode, GIL-releasing consumer)
# --------------------------------------------------------------------------- #
def test_thread_mode_overlaps_sleepy_consumer():
    a = b = 0.04
    n = 8

    def build(i):
        time.sleep(a)             # releases the GIL, like BLAS/IO
        return i

    def run(executor):
        t0 = time.perf_counter()
        for _ in prefetch_pipeline(range(n), build, depth=2,
                                   executor=executor):
            time.sleep(b)         # stands in for the GPU solve
        return time.perf_counter() - t0

    t_sync = run("sync")
    t_thread = run("thread")
    # perfect overlap would be ~max(a,b)*n + a; require >= ~1.4x
    assert t_thread < 0.75 * t_sync, (t_sync, t_thread)


# --------------------------------------------------------------------------- #
# lagged_collect
# --------------------------------------------------------------------------- #
def test_lagged_collect_order_and_values():
    arrs = [np.full(3, i, dtype=float) for i in range(6)]
    out = list(lagged_collect(iter(arrs), lag=2))
    assert len(out) == 6
    for i, a in enumerate(out):
        assert np.array_equal(a, arrs[i])
        assert isinstance(a, np.ndarray)


def test_lagged_collect_pytrees_and_lag():
    materialized = []

    class Probe:
        """numpy-coercible probe recording materialization order."""
        def __init__(self, i):
            self.i = i
        def __array__(self, dtype=None, copy=None):
            materialized.append(self.i)
            return np.full(2, self.i, dtype=float)

    def feed():
        for i in range(4):
            yield {"f": Probe(i), "v": Probe(i + 100)}
            # with lag=1, item i-1 must be materialized only after item i
            # was produced
            if i > 0:
                assert (i - 1) in materialized
            assert i not in materialized

    out = list(lagged_collect(feed(), lag=1))
    assert len(out) == 4
    assert np.array_equal(out[2]["f"], np.full(2, 2.0))
    assert np.array_equal(out[2]["v"], np.full(2, 102.0))
