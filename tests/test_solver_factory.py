"""Tests for tractor_jax.jax.batching.make_batched_solver.

The factory must be bit-identical to the hand-rolled
``jax.jit(jax.vmap(partial(solve_fluxes_*, ...), in_axes=...))`` it replaces,
memoize callables on their static configuration, and — for lasso — accept
per-image penalty weights as a RUNTIME argument so weight values that differ
per call share one compiled executable (the per-cutout-recompile fix).

Run in the `spherex` conda env:  pytest tests/test_solver_factory.py -q
"""
from functools import partial

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tractor_jax import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor_jax.brightness import Flux
from tractor_jax.wcs import PixPos
from tractor_jax.psf import GaussianMixturePSF
from tractor_jax.jax.optimizer import (
    extract_model_data,
    render_image,
    solve_fluxes_linear,
    solve_fluxes_eigfloor,
    solve_fluxes_lasso,
)
from tractor_jax.jax.batching import (
    batches_in_axes,
    clear_solver_cache,
    make_batched_solver,
    penalty_weights_from_slots,
)


# --------------------------------------------------------------------------- #
# helpers (mirrors tests/test_lasso_solver.py toy_scene, multi-image)
# --------------------------------------------------------------------------- #
def batched_scene(n_img=3, H=24, W=24, noise_sigma=0.05, seed=3):
    rng = np.random.default_rng(seed)
    psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                             np.array([[[2.5, 0.0], [0.0, 2.5]]]))
    positions = [(6.3, 6.8), (8.1, 7.4), (16.6, 15.2), (18.9, 18.1)]
    true_fluxes = np.array([50.0, 8.0, 30.0, 0.0])

    srcs = [PointSource(PixPos(x, y), Flux(1.0)) for (x, y) in positions]
    cat = Catalog(*srcs)
    imgs = []
    for i in range(n_img):
        img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)) / noise_sigma,
                    psf=psf, wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
        img.name = f"toy{i}"
        imgs.append(img)
    tr = Tractor(imgs, cat)
    images_data, batches, init_flux = extract_model_data(tr)
    for i, img in enumerate(imgs):
        single = jax.tree_util.tree_map(lambda x: x[i], images_data)
        sb = {"PointSource": {
            "flux_idx": batches["PointSource"]["flux_idx"][i],
            "pos_pix": batches["PointSource"]["pos_pix"][i],
            "mask": batches["PointSource"]["mask"][i],
        }}
        model = np.array(render_image(jnp.array(true_fluxes), single, sb))[:H, :W]
        img.data += model + rng.normal(size=(H, W)) * noise_sigma
    images_data, batches, init_flux = extract_model_data(tr)
    return images_data, batches, init_flux


@pytest.fixture(scope="module")
def scene():
    return batched_scene()


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_solver_cache()
    yield
    clear_solver_cache()


# --------------------------------------------------------------------------- #
# bit-identity vs the hand-rolled wrapper
# --------------------------------------------------------------------------- #
def test_linear_matches_handrolled(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("linear", in_axes=bia, rcond=1e-12)
    ref_fn = jax.jit(jax.vmap(partial(solve_fluxes_linear, rcond=1e-12,
                                      return_variances=True),
                              in_axes=(0, 0, bia)))
    f, v = fn(init, images_data, batches)
    rf, rv = ref_fn(init, images_data, batches)
    assert np.array_equal(np.asarray(f), np.asarray(rf))
    assert np.array_equal(np.asarray(v), np.asarray(rv))


def test_eigfloor_matches_handrolled(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("eigfloor", in_axes=bia, floor=1e-2)
    ref_fn = jax.jit(jax.vmap(partial(solve_fluxes_eigfloor, floor=1e-2,
                                      return_variances=True),
                              in_axes=(0, 0, bia)))
    f, v = fn(init, images_data, batches)
    rf, rv = ref_fn(init, images_data, batches)
    assert np.array_equal(np.asarray(f), np.asarray(rf))
    assert np.array_equal(np.asarray(v), np.asarray(rv))


def test_lasso_matches_handrolled(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    kw = dict(alpha=1.0, penalty_mode="snr", nonneg=True, debias=True,
              debias_signfree="none", n_iter=400)
    fn = make_batched_solver("lasso", in_axes=bia, **kw)

    def _solve(i, d, b, pw):
        return solve_fluxes_lasso(i, d, b, penalty_weights=pw,
                                  return_variances=True, **kw)
    ref_fn = jax.jit(jax.vmap(_solve, in_axes=(0, 0, bia, 0)))
    pw = jnp.ones_like(init)
    f, v = fn(init, images_data, batches, pw)
    rf, rv = ref_fn(init, images_data, batches, pw)
    assert np.array_equal(np.asarray(f), np.asarray(rf))
    assert np.array_equal(np.asarray(v), np.asarray(rv))


def test_lasso_default_pw_equals_ones(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("lasso", in_axes=bia, alpha=1.0, n_iter=400)
    f0, v0 = fn(init, images_data, batches)                      # None -> ones
    f1, v1 = fn(init, images_data, batches, jnp.ones_like(init))
    assert np.array_equal(np.asarray(f0), np.asarray(f1))
    assert np.array_equal(np.asarray(v0), np.asarray(v1))


# --------------------------------------------------------------------------- #
# cache semantics
# --------------------------------------------------------------------------- #
def test_cache_identity(scene):
    _, batches, _ = scene
    bia = batches_in_axes(batches)
    a = make_batched_solver("linear", in_axes=bia, rcond=1e-12)
    b = make_batched_solver("linear", in_axes=bia, rcond=1e-12)
    assert a is b
    c = make_batched_solver("linear", in_axes=bia, rcond=1e-6)
    assert c is not a
    d = make_batched_solver("eigfloor", in_axes=bia, floor=1e-2)
    e = make_batched_solver("eigfloor", in_axes=bia, floor=1e-3)
    assert d is not e
    f = make_batched_solver("linear", in_axes=bia, rcond=1e-12, cache=False)
    assert f is not a


def test_single_trace_across_pw_values(scene):
    """Different penalty-weight VALUES (same shape) must reuse one executable
    — this is the fix for the per-cutout lasso recompile."""
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("lasso", in_axes=bia, alpha=1.0, n_iter=400)
    n_img, n_flux = np.asarray(init).shape
    pw1 = jnp.ones((n_img, n_flux))
    pw2 = pw1.at[:, 1].set(200.0)        # crush source 1 out of the support
    f1, _ = fn(init, images_data, batches, pw1)
    f2, _ = fn(init, images_data, batches, pw2)
    jax.block_until_ready(f2)
    assert fn._jitted._cache_size() == 1
    # and the weights actually reached the solve: source 1 got zeroed
    assert np.all(np.asarray(f2)[:, 1] == 0.0)
    assert np.all(np.asarray(f1)[:, 1] > 0.0)


def test_penalty_weights_rejected_for_non_lasso(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("linear", in_axes=bia)
    with pytest.raises(ValueError, match="lasso-only"):
        fn(init, images_data, batches, jnp.ones_like(init))


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="unknown solver"):
        make_batched_solver("cg", in_axes={})


# --------------------------------------------------------------------------- #
# vmap-vs-sequential parity (mirrors test_lasso_solver conventions)
# --------------------------------------------------------------------------- #
def test_lasso_vmap_vs_sequential(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    kw = dict(alpha=1.0, penalty_mode="snr", nonneg=True, debias=True,
              debias_signfree="none", n_iter=1000)
    fn = make_batched_solver("lasso", in_axes=bia, **kw)
    f, _ = fn(init, images_data, batches)
    n_img = np.asarray(init).shape[0]
    for i in range(n_img):
        single = jax.tree_util.tree_map(lambda x: x[i], images_data)
        sb = jax.tree_util.tree_map(lambda x: x[i], batches)
        fi, _ = solve_fluxes_lasso(init[i], single, sb,
                                   return_variances=True, **kw)
        np.testing.assert_allclose(np.asarray(f[i]), np.asarray(fi),
                                   rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_batches_in_axes_structure(scene):
    _, batches, _ = scene
    bia = batches_in_axes(batches)
    assert set(bia) == set(batches)
    assert bia["PointSource"] == {"flux_idx": 0, "pos_pix": 0, "mask": 0}
    assert batches_in_axes({"Background": {"flux_idx": jnp.array([3])}}) == \
        {"Background": {"flux_idx": None}}


def test_penalty_weights_from_slots():
    src_slot = [{10: 0, 11: 1, 12: 3}, {11: 2}]
    pw = np.asarray(penalty_weights_from_slots(src_slot, 2, 4, {11, 12}))
    expect = np.ones((2, 4))
    expect[0, 1] = 0.0
    expect[0, 3] = 0.0
    expect[1, 2] = 0.0
    assert np.array_equal(pw, expect)
