"""Tests for tractor_jax.jax.batching.autotune_batch_size.

Synthetic timing models (sleep-based) exercise the knee-finding logic
without touching a device; an end-to-end check runs the real builder+solver
on the CPU backend at tiny sizes.

Run in the `spherex` conda env:  JAX_PLATFORMS=cpu pytest tests/test_autotune.py -q
"""
import time

import numpy as np
import pytest

from tractor_jax.jax.batching import (
    autotune_batch_size,
    estimate_solve_bytes_per_view,
)


def model_run(overhead_s, per_item_s, regress_at=None, calls=None):
    """run_batch stub: t(B) = overhead + B*per_item (launch-overhead model:
    throughput rises then plateaus); optional hard regression above a B."""
    def run(b):
        if calls is not None:
            calls.append(b)
        t = overhead_s + b * per_item_s
        if regress_at is not None and b >= regress_at:
            t *= 4.0
        time.sleep(t)
    return run


def test_finds_knee_of_plateau():
    # t(B) = 2ms + 0.5ms*B: doubling gain = 2*(2+0.5B)/(2+1B) drops below
    # 1.10 for B > 18 -> the tuner should stop at B=32 and return 16
    best, report = autotune_batch_size(
        model_run(2e-3, 5e-4), start=1, max_batch=1024, min_gain=0.10,
        repeats=2)
    assert 8 <= best <= 64
    # report is monotone increasing up to the returned knee
    bs = sorted(report)
    tps = [report[b] for b in bs]
    assert tps.index(max(tps)) >= bs.index(best)
    assert best in report


def test_regression_returns_previous_candidate():
    best, report = autotune_batch_size(
        model_run(1e-3, 1e-4, regress_at=16), start=1, max_batch=1024,
        min_gain=0.10, repeats=2)
    assert best == 8
    assert 16 in report            # the regressing candidate was measured


def test_memory_cap_limits_candidates():
    calls = []
    best, report = autotune_batch_size(
        model_run(4e-3, 1e-5, calls=calls), start=1, max_batch=1024,
        repeats=1, per_item_bytes=10 * 2 ** 20,
        mem_budget_bytes=60 * 2 ** 20)          # cap = 6 -> candidates 1,2,4
    assert max(report) <= 6
    assert best <= 6
    assert max(calls) <= 6


def test_warmup_plus_repeats_calls():
    calls = []
    autotune_batch_size(model_run(1e-4, 1e-5, calls=calls), start=1,
                        max_batch=4, min_gain=-1.0, repeats=3)
    # min_gain=-1 never triggers the knee -> all of 1,2,4 run warmup+3
    assert calls.count(1) == 4 and calls.count(2) == 4 and calls.count(4) == 4


def test_max_batch_respected_and_report_returned():
    best, report = autotune_batch_size(model_run(5e-3, 1e-6), start=2,
                                       max_batch=8, repeats=1)
    assert best == 8
    assert sorted(report) == [2, 4, 8]


def test_bad_args():
    with pytest.raises(ValueError, match="start"):
        autotune_batch_size(lambda b: None, start=0)
    with pytest.raises(ValueError, match="repeats"):
        autotune_batch_size(lambda b: None, repeats=0)
    with pytest.raises(ValueError, match="per_item_bytes"):
        autotune_batch_size(lambda b: None, mem_budget_bytes=1 << 20)


def test_estimate_solve_bytes_per_view():
    n_flux, shape = 60, (125, 125)
    est = estimate_solve_bytes_per_view(n_flux, shape)
    exact = 4 * (60 * 125 * 125 + 2 * 60 * 60 + 4 * 125 * 125)
    assert est == exact
    assert estimate_solve_bytes_per_view(n_flux, shape, dtype_bytes=8) == 2 * exact


# --------------------------------------------------------------------------- #
# end-to-end on the real builder + solver (CPU backend, tiny sizes)
# --------------------------------------------------------------------------- #
def test_end_to_end_with_real_solver():
    import jax
    from astropy.table import Table
    from tractor_jax.jax.batching import (batches_in_axes,
                                          build_padded_batches,
                                          make_batched_solver)

    rng = np.random.default_rng(3)
    n_src = 4
    tab = Table({"shape_r": np.zeros(n_src), "shape_ab": np.zeros(n_src),
                 "shape_phi": np.zeros(n_src), "sersic": np.zeros(n_src)})
    sx = rng.uniform(3, 12, n_src)
    sy = rng.uniform(3, 12, n_src)
    psf = np.exp(-0.5 * ((np.mgrid[:25, :25][0] - 12) ** 2
                         + (np.mgrid[:25, :25][1] - 12) ** 2) / 16.0)
    psf /= psf.sum()
    view = {"data": rng.normal(5, 1, (15, 15)),
            "invvar": np.full((15, 15), 4.0),
            "psf": psf, "src_indices": list(range(n_src)), "origin": (0, 0)}

    solver_cache = {}

    def run_batch(b):
        bundle = build_padded_batches([view] * b, tab, sx, sy,
                                      psf_sampling=0.2)
        fn = solver_cache.get(b)
        if fn is None:
            fn = make_batched_solver("linear", in_axes=bundle.in_axes,
                                     cache=False, rcond=1e-12)
            solver_cache[b] = fn
        f, _ = fn(bundle.initial_fluxes, bundle.images_data, bundle.batches)
        jax.block_until_ready(f)

    best, report = autotune_batch_size(run_batch, start=1, max_batch=8,
                                       repeats=1)
    assert best in report and 1 <= best <= 8
    assert all(tp > 0 for tp in report.values())
