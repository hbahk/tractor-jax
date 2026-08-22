"""The ``eig_method`` option of the eigfloor family (2026-08-22).

``eig_method="cusolver"`` (default) is the historical ``jnp.linalg.eigh``;
``"host"`` ships the batch of normalized Grams to LAPACK through
``jax.pure_callback`` (one thread-pool task per matrix).  The contract:

* the default path is untouched (same callable, same numbers);
* the host path is an fp32 eigensolver-level equivalent -- eigenvalues agree,
  the reconstruction V diag(w) V^T == G holds, the resulting fluxes agree with
  the cuSOLVER path to ~1e-3 relative, and it works under jit+vmap (the
  callback receives the whole batch, ``vmap_method="expand_dims"``);
* an unknown method raises.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_render_stamp import _build, _templates  # noqa: E402

from tractor_jax.jax import batching as tjb  # noqa: E402
from tractor_jax.jax.optimizer import _eigh_dispatch, _EIG_METHODS  # noqa: E402


def _sym_batch(n_batch=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_batch, n, n)).astype(np.float32)
    g = np.einsum("bij,bkj->bik", a, a) / n + np.eye(n, dtype=np.float32)
    return jnp.asarray(g)


def test_methods_and_default():
    assert _EIG_METHODS == ("cusolver", "host")
    G = _sym_batch()
    w0, v0 = jnp.linalg.eigh(G)
    w1, v1 = _eigh_dispatch(G)                       # default == cusolver path
    assert np.array_equal(np.asarray(w0), np.asarray(w1))
    assert np.array_equal(np.asarray(v0), np.asarray(v1))
    with pytest.raises(ValueError):
        _eigh_dispatch(G, "foo")


def test_host_eigh_matches_and_reconstructs_under_jit_vmap():
    G = _sym_batch(n_batch=7, n=16, seed=1)
    w_ref, _ = jnp.linalg.eigh(G)

    # eager, batched call
    w_h, v_h = _eigh_dispatch(G, "host", 2)
    assert w_h.shape == G.shape[:-1] and v_h.shape == G.shape
    assert w_h.dtype == G.dtype and v_h.dtype == G.dtype
    assert np.allclose(np.asarray(w_h), np.asarray(w_ref), rtol=1e-4, atol=1e-5)
    rec = np.einsum("bij,bj,bkj->bik", np.asarray(v_h), np.asarray(w_h), np.asarray(v_h))
    assert np.allclose(rec, np.asarray(G), rtol=1e-4, atol=1e-4)
    # ascending order, like jnp.linalg.eigh
    assert np.all(np.diff(np.asarray(w_h), axis=-1) >= -1e-6)

    # jit + vmap: the callback must receive the whole batch at once
    fn = jax.jit(jax.vmap(lambda g: _eigh_dispatch(g, "host", 2)))
    w_v, v_v = fn(G)
    assert np.allclose(np.asarray(w_v), np.asarray(w_ref), rtol=1e-4, atol=1e-5)
    rec = np.einsum("bij,bj,bkj->bik", np.asarray(v_v), np.asarray(w_v), np.asarray(v_v))
    assert np.allclose(rec, np.asarray(G), rtol=1e-4, atol=1e-4)


def _noiseless_problem():
    b = _build()
    n_flux = b.initial_fluxes.shape[1]
    truth = np.zeros(n_flux, np.float32)
    truth[:b.meta["max_ps"]] = [3.0, 5.0, 2.0]
    truth[b.meta["max_ps"]:b.meta["max_ps"] + 4] = [4.0, 6.0, 1.5, 8.0]
    truth[b.meta["bg_idx"]] = 0.2
    imgs = [np.tensordot(truth, _templates(b, i), axes=(0, 0)) for i in range(2)]
    data_pad = np.stack(imgs).astype(np.float32)
    imgd = dict(b.images_data)
    imgd["data"] = jnp.asarray(data_pad)
    return b, imgd, truth


@pytest.mark.parametrize("kind", ["eigfloor", "eigfloor_prior"])
def test_batched_eigfloor_family_host_matches_cusolver(kind):
    b, imgd, truth = _noiseless_problem()
    n_views, n_flux = b.initial_fluxes.shape

    def solve(method):
        fn = tjb.make_batched_solver(kind, in_axes=b.in_axes, floor=1e-2,
                                     eig_method=method, eig_host_threads=2,
                                     cache=False)
        if kind == "eigfloor_prior":
            lam = jnp.zeros((n_views, n_flux), jnp.float32)
            fp = jnp.zeros((n_views, n_flux), jnp.float32)
            f, v = fn(b.initial_fluxes, imgd, b.batches, lam, fp)
        else:
            f, v = fn(b.initial_fluxes, imgd, b.batches)
        return np.asarray(f), np.asarray(v)

    f_c, v_c = solve("cusolver")
    f_h, v_h = solve("host")
    live = truth != 0
    for i in range(n_views):
        # both recover the noiseless truth ...
        assert np.allclose(f_c[i][live], truth[live], rtol=2e-3, atol=2e-3)
        assert np.allclose(f_h[i][live], truth[live], rtol=2e-3, atol=2e-3)
        # ... and agree with each other at the fp32 eigensolver level
        assert np.allclose(f_h[i][live], f_c[i][live], rtol=1e-3, atol=1e-3)
        assert np.allclose(v_h[i][live], v_c[i][live], rtol=1e-2, atol=1e-6)
