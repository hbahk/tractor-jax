"""Per-source residual diagnostics returned by the linear flux solvers.

``return_diagnostics=True`` must (1) leave the fluxes and variances exactly as
they are without it, (2) equal a numpy computation from the engine's own
templates, and (3) flag what it is for: a bad pixel under a source raises that
source's chi2, a masked pixel under it raises its mask_frac.

Run in the `spherex` conda env:  pytest tests/test_solve_diagnostics.py -q
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tractor_jax.jax.optimizer import (
    _render_source_templates,
    solve_fluxes_eigfloor,
    solve_fluxes_eigfloor_prior,
    solve_fluxes_linear,
)
from tractor_jax.jax.batching import batches_in_axes, clear_solver_cache, make_batched_solver

from test_lasso_solver import single_image_inputs, toy_scene
from test_solver_factory import batched_scene

SOLVERS = {
    "linear": (solve_fluxes_linear, {}),
    "eigfloor": (solve_fluxes_eigfloor, {"floor": 1e-3}),
    "eigfloor_prior": (solve_fluxes_eigfloor_prior, {"floor": 1e-3}),
}


def numpy_diagnostics(single, sb, n_flux, fluxes):
    A = np.array(_render_source_templates(single, sb, n_flux)).reshape(n_flux, -1).T
    w = np.array(single["invvar"]).ravel()
    d = np.array(single["data"]).ravel()
    r = d - A @ np.asarray(fluxes)
    good = (w > 0).astype(float)
    t_good = A.T @ good
    t_all = A.sum(axis=0)
    chi2 = np.where(t_good > 0, (A.T @ (w * r * r)) / np.where(t_good > 0, t_good, 1.0), np.nan)
    mask_frac = np.where(t_all > 0, 1.0 - t_good / np.where(t_all > 0, t_all, 1.0), np.nan)
    return chi2, mask_frac


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_solver_cache()
    yield
    clear_solver_cache()


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_fluxes_and_variances_unchanged(name):
    fn, kw = SOLVERS[name]
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    f, v = fn(f0, single, sb, return_variances=True, **kw)
    f2, v2, diag = fn(f0, single, sb, return_variances=True, return_diagnostics=True, **kw)
    assert np.array_equal(np.array(f), np.array(f2))
    assert np.array_equal(np.array(v), np.array(v2))
    assert set(diag) == {"chi2", "mask_frac"}
    # without variances the diagnostics come second
    f3, diag3 = fn(f0, single, sb, return_diagnostics=True, **kw)
    assert np.array_equal(np.array(f), np.array(f3))
    assert np.array_equal(np.array(diag["chi2"]), np.array(diag3["chi2"]))


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_matches_numpy(name):
    fn, kw = SOLVERS[name]
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    f, _, diag = fn(f0, single, sb, return_variances=True, return_diagnostics=True, **kw)
    chi2, mfrac = numpy_diagnostics(single, sb, f0.shape[0], f)
    assert np.allclose(np.array(diag["chi2"]), chi2, rtol=1e-9, equal_nan=True)
    assert np.allclose(np.array(diag["mask_frac"]), mfrac, atol=1e-12, equal_nan=True)


def test_good_fit_chi2_near_one():
    tr, _ = toy_scene(noise_sigma=0.05)
    single, sb, f0 = single_image_inputs(tr)
    _, diag = solve_fluxes_eigfloor(f0, single, sb, return_diagnostics=True, floor=1e-3)
    chi2 = np.array(diag["chi2"])
    assert np.all((chi2 > 0.3) & (chi2 < 2.5)), chi2
    # nothing is masked; only the image padding beyond the 24 px edge has
    # zero weight, which source 3 at (18.9, 18.1) barely reaches
    mf = np.array(diag["mask_frac"])
    assert np.all(mf[:3] < 1e-4) and mf[3] < 0.01, mf


def test_bad_pixel_raises_chi2_of_its_source_only():
    tr, _ = toy_scene(noise_sigma=0.05)
    single, sb, f0 = single_image_inputs(tr)
    _, base = solve_fluxes_eigfloor(f0, single, sb, return_diagnostics=True, floor=1e-3)
    # a low pixel with a small error next to source 2 (16.6, 15.2), as in a
    # SPHEREx visit whose cold pixel was not flagged
    data = np.array(single["data"])
    invvar = np.array(single["invvar"])
    data[15, 17] -= 2.0
    invvar[15, 17] *= 8.0
    bad = dict(single, data=jnp.asarray(data), invvar=jnp.asarray(invvar))
    _, diag = solve_fluxes_eigfloor(f0, bad, sb, return_diagnostics=True, floor=1e-3)
    chi2, chi2_0 = np.array(diag["chi2"]), np.array(base["chi2"])
    assert chi2[2] > 10 * chi2_0[2]
    far = [0, 1]          # sources at (6.3, 6.8) and (8.1, 7.4)
    assert np.allclose(chi2[far], chi2_0[far], rtol=0.05)


def test_masked_pixels_raise_mask_frac():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    invvar = np.array(single["invvar"])
    invvar[5:9, 5:8] = 0.0            # under source 0 at (6.3, 6.8)
    masked = dict(single, invvar=jnp.asarray(invvar))
    _, diag = solve_fluxes_eigfloor(f0, masked, sb, return_diagnostics=True, floor=1e-3)
    mf = np.array(diag["mask_frac"])
    assert 0.3 < mf[0] < 1.0
    assert mf[2] < 1e-4 and mf[3] < 0.01


def test_factory_returns_stacked_diagnostics():
    images_data, batches, init = batched_scene(n_img=3)
    ax = batches_in_axes(batches)
    plain = make_batched_solver("eigfloor", in_axes=ax, floor=1e-3)
    diag_fn = make_batched_solver("eigfloor", in_axes=ax, floor=1e-3, return_diagnostics=True)
    assert diag_fn is not plain
    f, v = plain(init, images_data, batches)
    f2, v2, diag = diag_fn(init, images_data, batches)
    # Under jit the extra outputs can change XLA's fusion of the shared graph,
    # so equality is to rounding, not to the bit (unlike the eager calls above)
    assert np.allclose(np.array(f), np.array(f2), rtol=1e-12, atol=1e-12)
    assert np.allclose(np.array(v), np.array(v2), rtol=1e-12, atol=1e-12)
    assert np.array(diag["chi2"]).shape == np.array(f).shape
    assert make_batched_solver("eigfloor", in_axes=ax, floor=1e-3, return_diagnostics=True) is diag_fn


def test_factory_prior_solver_with_diagnostics():
    images_data, batches, init = batched_scene(n_img=2)
    ax = batches_in_axes(batches)
    fn = make_batched_solver("eigfloor_prior", in_axes=ax, floor=1e-3, return_diagnostics=True)
    f, v, diag = fn(init, images_data, batches)
    assert np.array(diag["mask_frac"]).shape == np.array(f).shape


def test_lasso_rejects_diagnostics():
    images_data, batches, init = batched_scene(n_img=1)
    with pytest.raises(ValueError, match="lasso"):
        make_batched_solver("lasso", in_axes=batches_in_axes(batches), return_diagnostics=True)
