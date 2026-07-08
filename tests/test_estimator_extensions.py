"""Tests for the estimator extensions (2026-07-05).

Two additions motivated by the SED-shape / photo-z bias analysis
(proj-spherex-gpupipe research notes lasso_alpha/12 and 01 §10):

  A. `debias_signfree` in solve_fluxes_lasso / solve_fluxes_lasso_batched /
     _lasso_core - the DEBIAS refit's non-negativity clip becomes
     configurable ("none" | "protected" | "all") while the SELECTION prox
     keeps `nonneg`. Rationale: clipping a faint protected target at zero is
     a band-dependent positive (rectification) bias that the error bar does
     not absorb and that distorts colours / photo-z.

  B. solve_fluxes_eigfloor - direct linear solve with an eigenvalue floor on
     AtWA: sign-free L2 damping of only the degenerate directions; candidate
     blind-survey default estimator.

Run in the `spherex` conda env:  pytest tests/test_estimator_extensions.py -q
"""
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
    _render_source_templates,
    render_image,
    solve_fluxes_eigfloor,
    solve_fluxes_lasso,
    solve_fluxes_lasso_batched,
    solve_fluxes_linear,
    optimize_fluxes,
)

from test_lasso_solver import toy_scene, single_image_inputs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def normal_equations(single, sb, n_flux):
    """Numpy G = AtWA, b = AtWd from the engine's own templates."""
    templates = np.array(_render_source_templates(single, sb, n_flux))
    A = templates.reshape(n_flux, -1).T
    w = np.array(single["invvar"]).ravel()
    d = np.array(single["data"]).ravel()
    Aw = A * w[:, None]
    return Aw.T @ A, Aw.T @ d


def scene_with_negative_target(k_sigma=4.0):
    """Toy scene where the protected source (index 3, truth 0) has a clearly
    NEGATIVE maximum-likelihood flux: subtract k_sigma * sigma_f3 of its own
    unit template from the data."""
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    n = f0.shape[0]

    _, var_lin = solve_fluxes_linear(f0, single, sb, return_variances=True)
    sigma3 = float(np.sqrt(np.array(var_lin)[3]))

    e3 = jnp.zeros(n).at[3].set(1.0)
    tmpl3 = render_image(e3, single, sb)
    single = dict(single)
    single["data"] = single["data"] - k_sigma * sigma3 * tmpl3
    return single, sb, f0, sigma3


# --------------------------------------------------------------------------- #
# A1. debias_signfree: "none" clips the protected target, "protected" frees it
# --------------------------------------------------------------------------- #
def test_signfree_protected_target_keeps_negative_flux():
    single, sb, f0, sigma3 = scene_with_negative_target(k_sigma=4.0)
    w = jnp.array([1.0, 1.0, 1.0, 0.0])          # source 3 protected
    kw = dict(alpha=2.0, penalty_mode="snr", penalty_weights=w,
              nonneg=True, debias=True, n_iter=4000)

    f_none = np.array(solve_fluxes_lasso(f0, single, sb,
                                         debias_signfree="none", **kw))
    f_prot = np.array(solve_fluxes_lasso(f0, single, sb,
                                         debias_signfree="protected", **kw))

    # original behavior: the protected target is rectified to exactly zero
    assert f_none[3] == 0.0
    # sign-free debias: the ~-4 sigma excursion survives
    assert f_prot[3] < -2.0 * sigma3
    # the clip is per-coordinate: all other (positive) coordinates unchanged
    assert np.allclose(f_none[:3], f_prot[:3], rtol=0, atol=0)


def test_signfree_invalid_value_raises():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    with pytest.raises(ValueError):
        solve_fluxes_lasso(f0, single, sb, alpha=2.0,
                           debias_signfree="everything")


# --------------------------------------------------------------------------- #
# A2. "all" mode == exact unclipped pinned refit (numpy oracle)
# --------------------------------------------------------------------------- #
def test_signfree_all_matches_numpy_pinned_refit():
    single, sb, f0, _ = scene_with_negative_target(k_sigma=4.0)
    n = f0.shape[0]
    w = jnp.array([1.0, 1.0, 1.0, 0.0])
    rcond = 1e-12
    kw = dict(alpha=2.0, penalty_mode="snr", penalty_weights=w,
              nonneg=True, debias=True, n_iter=4000, rcond=rcond)

    f_all, var_all, aux = solve_fluxes_lasso(
        f0, single, sb, debias_signfree="all", return_variances=True,
        return_aux=True, **kw)
    s = np.array(aux["support"])

    G, b = normal_equations(single, sb, n)
    reg_j = rcond * np.clip(np.diag(G), 0.0, None)
    Gs = G * s[:, None] * s[None, :] + np.diag(1.0 - s + reg_j)
    f_ref = np.linalg.solve(Gs, b * s) * s

    assert np.allclose(np.array(f_all), f_ref, rtol=1e-9, atol=1e-11)


# --------------------------------------------------------------------------- #
# A3. the headline: nonneg debias has a positive clip bias at truth zero,
#     sign-free debias removes it (batched noise realizations)
# --------------------------------------------------------------------------- #
def test_signfree_removes_clip_bias_statistically():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    n = f0.shape[0]

    # noiseless model with the protected target (index 3) at truth ZERO
    truth = jnp.array([50.0, 8.0, 30.0, 0.0])
    model = render_image(truth, single, sb)
    invvar = np.array(single["invvar"])
    sigma_pix = np.where(invvar > 0, 1.0 / np.sqrt(np.where(invvar > 0,
                                                            invvar, 1.0)), 0.0)

    B = 400
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(B,) + model.shape) * sigma_pix[None]
    data_stack = jnp.asarray(np.array(model)[None] + noise)

    w = jnp.array([1.0, 1.0, 1.0, 0.0])
    kw = dict(alpha=2.0, penalty_mode="snr", penalty_weights=w,
              nonneg=True, debias=True, n_iter=2000)

    f_none = np.array(solve_fluxes_lasso_batched(
        f0, single, sb, data_stack, debias_signfree="none", **kw))
    f_prot = np.array(solve_fluxes_lasso_batched(
        f0, single, sb, data_stack, debias_signfree="protected", **kw))

    t_none, t_prot = f_none[:, 3], f_prot[:, 3]
    sig_emp = t_prot.std()                       # empirical refit sigma
    se = sig_emp / np.sqrt(B)

    # sign-free: symmetric about the truth (0); roughly half go negative
    assert abs(t_prot.mean()) < 4.0 * se
    assert 0.3 < np.mean(t_prot < 0) < 0.7
    # nonneg: rectified - never negative, mean biased high by ~sigma/sqrt(2pi)
    assert t_none.min() >= 0.0
    assert t_none.mean() > 0.25 * sig_emp
    assert t_none.mean() > t_prot.mean() + 5.0 * se


# --------------------------------------------------------------------------- #
# B1. eigfloor == numpy eigen-decomposition oracle (fluxes and variances)
# --------------------------------------------------------------------------- #
def test_eigfloor_matches_numpy_reference():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    n = f0.shape[0]
    floor = 1e-3

    f_eng, v_eng = solve_fluxes_eigfloor(f0, single, sb,
                                         return_variances=True, floor=floor)

    # Jacobi-normalized (unit-diagonal) eigen-floor oracle
    G, b = normal_equations(single, sb, n)
    D = np.sqrt(np.diag(G))
    Ghat = G / np.outer(D, D)
    evals, evecs = np.linalg.eigh(Ghat)
    evals_f = np.maximum(evals, floor * max(evals[-1], 1e-30))
    f_ref = (evecs @ ((evecs.T @ (b / D)) / evals_f)) / D
    v_ref = np.sum(evecs**2 / evals_f[None, :], axis=1) / D**2

    assert np.allclose(np.array(f_eng), f_ref, rtol=1e-9, atol=1e-11)
    assert np.allclose(np.array(v_eng), v_ref, rtol=1e-9, atol=1e-11)


# --------------------------------------------------------------------------- #
# B2. tiny floor -> plain (un-regularized) linear solve
# --------------------------------------------------------------------------- #
def test_eigfloor_tiny_floor_equals_linear():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)

    f_ef = np.array(solve_fluxes_eigfloor(f0, single, sb, floor=1e-13))
    f_lin = np.array(solve_fluxes_linear(f0, single, sb, rcond=1e-13))
    assert np.max(np.abs(f_ef - f_lin)) / np.max(np.abs(f_lin)) < 1e-8


# --------------------------------------------------------------------------- #
# B3. degenerate pair: the anti-correlated split is damped, the sum survives
# --------------------------------------------------------------------------- #
def blended_pair_scene(sep=0.2, noise_sigma=0.2, seed=11):
    """Two nearly coincident sources (sep*sqrt(2) px apart, PSF sigma ~1.6 px).

    At sep=0.2 the pair-difference eigenvalue ratio is
    (1-rho)/(1+rho) ~ 4e-3 (rho = exp(-d^2/4 sigma^2)), safely below the
    1e-2 floor used in the test, so the floor genuinely bites."""
    rng = np.random.default_rng(seed)
    H = W = 24
    psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                             np.array([[[2.5, 0.0], [0.0, 2.5]]]))
    positions = [(11.8, 12.1), (11.8 + sep, 12.1 + sep)]
    truth = np.array([20.0, 10.0])
    srcs = [PointSource(PixPos(x, y), Flux(1.0)) for (x, y) in positions]
    img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)) / noise_sigma,
                psf=psf, wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
    img.name = "pair"
    tr = Tractor([img], Catalog(*srcs))
    single, sb, f0 = single_image_inputs(tr)
    model = render_image(jnp.array(truth), single, sb)
    single = dict(single)
    single["data"] = (single["data"] + model
                      + rng.normal(size=model.shape) * noise_sigma)
    return single, sb, f0, truth


def test_eigfloor_damps_degenerate_split_keeps_sum():
    single, sb, f0, truth = blended_pair_scene()

    f_dir = np.array(solve_fluxes_linear(f0, single, sb, rcond=1e-12))
    f_ef = np.array(solve_fluxes_eigfloor(f0, single, sb, floor=1e-2))

    # the well-constrained mode (pair sum) is preserved
    assert abs(f_ef.sum() - f_dir.sum()) < 0.02 * abs(f_dir.sum())
    # the degenerate mode (pair split) is substantially shrunk
    assert abs(f_ef[0] - f_ef[1]) < 0.8 * abs(f_dir[0] - f_dir[1])

    # variance of the damped mode shrinks too
    _, v_dir = solve_fluxes_linear(f0, single, sb, return_variances=True,
                                   rcond=1e-12)
    _, v_ef = solve_fluxes_eigfloor(f0, single, sb, return_variances=True,
                                    floor=1e-2)
    assert np.all(np.array(v_ef) <= np.array(v_dir) * (1 + 1e-9))


# --------------------------------------------------------------------------- #
# B4. all-dead problem: fluxes pinned to 0, variances to inf, no NaN
# --------------------------------------------------------------------------- #
def test_eigfloor_dead_slots_pinned():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    single = dict(single)
    single["invvar"] = jnp.zeros_like(single["invvar"])

    f, v = solve_fluxes_eigfloor(f0, single, sb, return_variances=True)
    assert np.all(np.array(f) == 0.0)
    assert np.all(np.isinf(np.array(v)))


# --------------------------------------------------------------------------- #
# B5. high-level path: optimize_fluxes(solver="eigfloor") ~ linear on a
#     well-separated scene (all eigenvalues above the floor)
# --------------------------------------------------------------------------- #
def test_optimize_fluxes_eigfloor_smoke():
    tr, truth = toy_scene()
    res_lin = optimize_fluxes(tr, return_variances=True, solver="linear",
                              use_sharding=False)
    res_ef = optimize_fluxes(tr, return_variances=True, solver="eigfloor",
                             eig_floor=1e-6, use_sharding=False)
    f_lin, v_lin = np.array(res_lin[0][0]), np.array(res_lin[0][1])
    f_ef, v_ef = np.array(res_ef[0][0]), np.array(res_ef[0][1])
    assert np.allclose(f_ef, f_lin, rtol=1e-5, atol=1e-8)
    assert np.all(np.isfinite(f_ef))
    finite = np.isfinite(v_lin) & np.isfinite(v_ef)
    assert np.allclose(v_ef[finite], v_lin[finite], rtol=1e-4)


# --------------------------------------------------------------------------- #
# B6. a dominant-norm column (fit background) must NOT drag the floor up and
#     shrink the source fluxes — the field-sim failure mode that motivated the
#     Jacobi normalization (raw-AtWA flooring gave −50..−99% bias at high S/N
#     because lambda_max was the background mode)
# --------------------------------------------------------------------------- #
def test_eigfloor_immune_to_background_column_domination():
    tr, truth = toy_scene()
    res_lin = optimize_fluxes(tr, return_variances=True, solver="linear",
                              fit_background=True, use_sharding=False)
    res_ef = optimize_fluxes(tr, return_variances=True, solver="eigfloor",
                             eig_floor=1e-4, fit_background=True,
                             use_sharding=False)
    f_lin = np.array(res_lin[0][0])
    f_ef = np.array(res_ef[0][0])
    # sources are well separated: with the background column present, the
    # floored solve must still agree with the plain solve to sub-percent
    bright = np.abs(f_lin) > 1.0
    rel = np.abs(f_ef[bright] - f_lin[bright]) / np.abs(f_lin[bright])
    assert np.max(rel) < 1e-2, rel


# --------------------------------------------------------------------------- #
# A4. alpha="auto" == manual sqrt(2 ln p) (universal-threshold rule)
# --------------------------------------------------------------------------- #
def test_alpha_auto_equals_manual_rule():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    n = f0.shape[0]
    w = jnp.array([1.0, 1.0, 1.0, 0.0])          # source 3 protected
    kw = dict(penalty_mode="snr", penalty_weights=w, nonneg=True,
              debias=True, n_iter=3000)

    f_auto = np.array(solve_fluxes_lasso(f0, single, sb, alpha="auto", **kw))
    # p = penalized live candidates = 3 (sources 0-2; src 3 protected)
    a_manual = float(np.sqrt(2.0 * np.log(3.0)))
    f_manual = np.array(solve_fluxes_lasso(f0, single, sb, alpha=a_manual,
                                           **kw))
    assert np.allclose(f_auto, f_manual, rtol=1e-12, atol=1e-14)

    with pytest.raises(ValueError):
        solve_fluxes_lasso(f0, single, sb, alpha="rule", **kw)
