"""Tests for the Gaussian-flux-prior eigfloor solver (ridge-toward-prior).

``solve_fluxes_eigfloor_prior`` solves
``(AtWA + Lambda) f = AtWd + Lambda f_prior`` with
``Lambda = diag(1/sigma_prior^2)``, in the Jacobi-normalized coordinates of
``solve_fluxes_eigfloor``; the eigenvalue floor acts on the REGULARIZED
normalized Gram. Contracts checked here:

- Lambda = 0 reproduces ``solve_fluxes_eigfloor`` exactly (fp64 scene and
  a float32-cast scene, i.e. the production dtype);
- analytic ridge on a single-source system:
  ``f_hat = (AtWd + lam f_prior) / (AtWA + lam)``;
- sigma_prior -> 0 pins f -> f_prior; sigma_prior -> inf recovers OLS;
- variances match ``diag((AtWA + Lambda)^{-1})`` from a numpy inverse;
- padded (dead) slots are pinned to 0 / inf variance, never NaN, and do
  not influence real slots even when a prior is placed on them;
- the ``make_batched_solver`` factory takes ``(lambda_diag, f_prior)`` as
  RUNTIME arrays (one executable across values), and its ``None`` default
  is bit-identical to the ``"eigfloor"`` factory solver.

Run in the `spherex` conda env (CPU is fine):
    JAX_PLATFORMS=cpu pytest tests/test_eigfloor_prior.py -q
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tractor_jax import (Tractor, Image, PointSource, Catalog, NullWCS,
                         ConstantSky)
from tractor_jax.brightness import Flux
from tractor_jax.wcs import PixPos
from tractor_jax.psf import GaussianMixturePSF
from tractor_jax.jax.optimizer import (
    _eigfloor_prior_core,
    _render_source_templates,
    extract_model_data,
    render_image,
    solve_fluxes_eigfloor,
    solve_fluxes_eigfloor_prior,
)
from tractor_jax.jax.batching import (
    batches_in_axes,
    clear_solver_cache,
    make_batched_solver,
    prior_arrays_from_slots,
)


# --------------------------------------------------------------------------- #
# helpers (mirrors tests/test_solver_factory.py batched_scene)
# --------------------------------------------------------------------------- #
def batched_scene(n_img=3, H=24, W=24, noise_sigma=0.05, seed=3,
                  positions=None, true_fluxes=None):
    rng = np.random.default_rng(seed)
    psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                             np.array([[[2.5, 0.0], [0.0, 2.5]]]))
    if positions is None:
        positions = [(6.3, 6.8), (8.1, 7.4), (16.6, 15.2), (18.9, 18.1)]
    if true_fluxes is None:
        true_fluxes = np.array([50.0, 8.0, 30.0, 0.0])
    true_fluxes = np.asarray(true_fluxes, dtype=np.float64)

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


def single_image(scene, i=0):
    images_data, batches, init = scene
    single = jax.tree_util.tree_map(lambda x: x[i], images_data)
    sb = jax.tree_util.tree_map(lambda x: x[i], batches)
    return init[i], single, sb


def normal_equations(init, single, sb):
    """(AtWA, AtWd) exactly as the solvers build them internally."""
    n_flux = init.shape[0]
    templates = _render_source_templates(single, sb, n_flux)
    A = np.asarray(templates).reshape(n_flux, -1).T
    w = np.asarray(single["invvar"]).ravel()
    d = np.asarray(single["data"]).ravel()
    AtWA = (A * w[:, None]).T @ A
    AtWd = (A * w[:, None]).T @ d
    return AtWA, AtWd


def cast_f32(tree):
    def c(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            return x.astype(jnp.complex64)
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(c, tree)


def random_system(n, seed, lam_scale=0.0):
    """Well-conditioned random normal equations (+ optional random priors)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(4 * n, n))
    AtWA = A.T @ A + n * np.eye(n)
    AtWd = rng.normal(size=n) * 10.0
    lam = lam_scale * rng.uniform(0.5, 2.0, size=n)
    fp = rng.normal(size=n) * 5.0
    return AtWA, AtWd, lam, fp


@pytest.fixture(scope="module")
def scene():
    return batched_scene()


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_solver_cache()
    yield
    clear_solver_cache()


# --------------------------------------------------------------------------- #
# Lambda = 0 equivalence with the existing eigfloor
# --------------------------------------------------------------------------- #
def test_lambda_zero_matches_eigfloor_fp64(scene):
    init, single, sb = single_image(scene)
    f0, v0 = solve_fluxes_eigfloor(init, single, sb, return_variances=True,
                                   floor=1e-4)
    f1, v1 = solve_fluxes_eigfloor_prior(init, single, sb,
                                         return_variances=True, floor=1e-4)
    np.testing.assert_allclose(np.asarray(f1), np.asarray(f0),
                               rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v0),
                               rtol=1e-14, atol=0)


def test_lambda_zero_matches_eigfloor_float32(scene):
    """Production dtype: float32-cast scene, explicit zero prior arrays."""
    init, single, sb = single_image(scene)
    init32, single32, sb32 = cast_f32(init), cast_f32(single), cast_f32(sb)
    f0, v0 = solve_fluxes_eigfloor(init32, single32, sb32,
                                   return_variances=True, floor=1e-4)
    lam = jnp.zeros_like(init32)
    fp = jnp.zeros_like(init32)
    f1, v1 = solve_fluxes_eigfloor_prior(init32, single32, sb32,
                                         lambda_diag=lam, f_prior=fp,
                                         return_variances=True, floor=1e-4)
    np.testing.assert_allclose(np.asarray(f1), np.asarray(f0),
                               rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v0),
                               rtol=1e-6, atol=0)


def test_lambda_zero_core_random_systems():
    """Core math on random well-conditioned systems, fp64 and float32."""
    for seed in range(5):
        AtWA, AtWd, _, _ = random_system(6, seed)
        for dt, rtol in ((np.float64, 1e-14), (np.float32, 2e-5)):
            G = jnp.asarray(AtWA, dtype=dt)
            b = jnp.asarray(AtWd, dtype=dt)
            z = jnp.zeros(6, dtype=dt)
            f1, v1 = _eigfloor_prior_core(G, b, z, z, floor=1e-4,
                                          return_variances=True)
            # reference: the eigfloor math (unit test of the shared identity)
            D = np.sqrt(np.diag(AtWA))
            Ghat = AtWA / np.outer(D, D)
            evals, evecs = np.linalg.eigh(Ghat)
            evals_f = np.maximum(evals, 1e-4 * evals[-1])
            xhat = evecs @ ((evecs.T @ (AtWd / D)) / evals_f)
            f_ref = xhat / D
            v_ref = np.sum(evecs * evecs / evals_f[None, :], axis=1) / (D * D)
            np.testing.assert_allclose(np.asarray(f1), f_ref, rtol=rtol)
            np.testing.assert_allclose(np.asarray(v1), v_ref, rtol=rtol)


# --------------------------------------------------------------------------- #
# analytic ridge, single source
# --------------------------------------------------------------------------- #
def test_analytic_ridge_single_source():
    """1-source scene: f_hat = (AtWd + lam*f_prior) / (AtWA + lam)."""
    sc = batched_scene(n_img=1, positions=[(11.5, 12.2)],
                       true_fluxes=np.array([20.0]), seed=7)
    init, single, sb = single_image(sc)
    AtWA, AtWd = normal_equations(init, single, sb)
    g, b = AtWA[0, 0], AtWd[0]

    lam, fp = 0.5 * g, 35.0     # prior precision comparable to the data's
    f, v = solve_fluxes_eigfloor_prior(
        init, single, sb, lambda_diag=jnp.array([lam]),
        f_prior=jnp.array([fp]), return_variances=True, floor=1e-4)
    f_expect = (b + lam * fp) / (g + lam)
    np.testing.assert_allclose(float(f[0]), f_expect, rtol=1e-12)
    np.testing.assert_allclose(float(v[0]), 1.0 / (g + lam), rtol=1e-12)
    # equivalently: convex combination of OLS and the prior mean
    f_ols = b / g
    w = g / (g + lam)
    np.testing.assert_allclose(float(f[0]), w * f_ols + (1 - w) * fp,
                               rtol=1e-12)


def test_sigma_limits_single_source():
    """sigma_prior -> 0 pins f -> f_prior; sigma -> inf recovers OLS."""
    sc = batched_scene(n_img=1, positions=[(11.5, 12.2)],
                       true_fluxes=np.array([20.0]), seed=11)
    init, single, sb = single_image(sc)
    AtWA, AtWd = normal_equations(init, single, sb)
    f_ols = AtWd[0] / AtWA[0, 0]
    fp = 35.0

    # sigma -> 0: huge (finite) precision; single source, so the floor
    # (relative to the one regularized eigenvalue) is a no-op.
    lam_pin = 1e12 * AtWA[0, 0]
    f_pin = solve_fluxes_eigfloor_prior(
        init, single, sb, lambda_diag=jnp.array([lam_pin]),
        f_prior=jnp.array([fp]), floor=1e-4)
    np.testing.assert_allclose(float(f_pin[0]), fp, rtol=1e-9)

    # sigma -> inf: lambda -> 0, exact OLS/eigfloor.
    f_free = solve_fluxes_eigfloor_prior(
        init, single, sb, lambda_diag=jnp.array([0.0]),
        f_prior=jnp.array([fp]), floor=1e-4)
    np.testing.assert_allclose(float(f_free[0]), f_ols, rtol=1e-12)


def test_sigma_limits_multi_source_core():
    """Pinning one coordinate leaves the others at their (Lambda-reduced)
    ridge solution; sigma -> inf on every coordinate recovers OLS."""
    AtWA, AtWd, _, _ = random_system(5, seed=2)
    fp = np.array([1.0, -2.0, 3.0, 0.5, 4.0])

    # all sigma -> inf (lam = 0), tiny floor: plain OLS
    f = _eigfloor_prior_core(jnp.asarray(AtWA), jnp.asarray(AtWd),
                             jnp.zeros(5), jnp.asarray(fp), floor=1e-15)
    np.testing.assert_allclose(np.asarray(f), np.linalg.solve(AtWA, AtWd),
                               rtol=1e-10)

    # pin coordinate 2 hard (sigma -> 0); floor tiny so the inflated
    # lambda_max does not damp the free coordinates. The eigh-based solve
    # carries O(eps * lambda_max) absolute noise into the free coordinates,
    # so the pin strength and tolerances are matched (1e8 -> ~1e-8 abs).
    lam = np.zeros(5)
    lam[2] = 1e8 * AtWA[2, 2]
    f = _eigfloor_prior_core(jnp.asarray(AtWA), jnp.asarray(AtWd),
                             jnp.asarray(lam), jnp.asarray(fp), floor=1e-15)
    np.testing.assert_allclose(float(f[2]), fp[2], rtol=1e-6)
    f_ref = np.linalg.solve(AtWA + np.diag(lam), AtWd + lam * fp)
    np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- #
# variances vs numpy inverse
# --------------------------------------------------------------------------- #
def test_variance_matches_numpy_inverse():
    for seed in range(4):
        AtWA, AtWd, lam, fp = random_system(6, seed, lam_scale=3.0)
        lam[0] = 0.0     # one protected coordinate
        f, v = _eigfloor_prior_core(jnp.asarray(AtWA), jnp.asarray(AtWd),
                                    jnp.asarray(lam), jnp.asarray(fp),
                                    floor=1e-15, return_variances=True)
        cov = np.linalg.inv(AtWA + np.diag(lam))
        np.testing.assert_allclose(np.asarray(v), np.diag(cov), rtol=1e-10)
        np.testing.assert_allclose(np.asarray(f),
                                   cov @ (AtWd + lam * fp), rtol=1e-10)


def test_prior_tightens_variance(scene):
    """Adding a prior can only reduce the marginal variances."""
    init, single, sb = single_image(scene)
    _, v0 = solve_fluxes_eigfloor_prior(init, single, sb,
                                        return_variances=True)
    lam = jnp.zeros_like(init).at[1].set(1.0)
    _, v1 = solve_fluxes_eigfloor_prior(init, single, sb, lambda_diag=lam,
                                        f_prior=jnp.zeros_like(init),
                                        return_variances=True)
    assert float(v1[1]) < float(v0[1])


# --------------------------------------------------------------------------- #
# padded-slot safety
# --------------------------------------------------------------------------- #
def test_dead_slot_core_no_nan_no_crosstalk():
    """A dead (zero-template) slot is pinned to 0/inf and leaves the live
    coordinates exactly as in the dead-slot-free system, even when a prior
    is (wrongly) placed on the dead slot."""
    AtWA, AtWd, lam, fp = random_system(4, seed=5, lam_scale=2.0)
    n = 5
    G = np.zeros((n, n))
    G[:4, :4] = AtWA
    b = np.zeros(n)
    b[:4] = AtWd
    lam5 = np.append(lam, 3.0)      # prior on the dead slot
    fp5 = np.append(fp, 7.0)

    f5, v5 = _eigfloor_prior_core(jnp.asarray(G), jnp.asarray(b),
                                  jnp.asarray(lam5), jnp.asarray(fp5),
                                  floor=1e-4, return_variances=True)
    f4, v4 = _eigfloor_prior_core(jnp.asarray(AtWA), jnp.asarray(AtWd),
                                  jnp.asarray(lam), jnp.asarray(fp),
                                  floor=1e-4, return_variances=True)
    assert np.all(np.isfinite(np.asarray(f5)))
    assert float(f5[4]) == 0.0
    assert np.isposinf(float(v5[4]))
    np.testing.assert_allclose(np.asarray(f5)[:4], np.asarray(f4),
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v5)[:4], np.asarray(v4),
                               rtol=1e-12)


def test_masked_source_scene_pinned(scene):
    """Scene-level: masking a source (padding-style dead slot) pins it even
    under a prior, and produces no NaNs anywhere."""
    init, single, sb = single_image(scene)
    sb_dead = jax.tree_util.tree_map(lambda x: x, sb)
    sb_dead["PointSource"] = dict(sb["PointSource"])
    sb_dead["PointSource"]["mask"] = sb["PointSource"]["mask"].at[3].set(0.0)
    lam = jnp.zeros_like(init).at[3].set(1e4)
    fp = jnp.zeros_like(init).at[3].set(123.0)
    f, v = solve_fluxes_eigfloor_prior(init, single, sb_dead,
                                       lambda_diag=lam, f_prior=fp,
                                       return_variances=True)
    assert np.all(np.isfinite(np.asarray(f)))
    assert float(f[3]) == 0.0
    assert np.isposinf(float(v[3]))
    assert np.all(np.isfinite(np.asarray(v)[:3]))


# --------------------------------------------------------------------------- #
# batching factory + slot helper
# --------------------------------------------------------------------------- #
def test_factory_default_matches_eigfloor(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn_ref = make_batched_solver("eigfloor", in_axes=bia, floor=1e-2)
    fn = make_batched_solver("eigfloor_prior", in_axes=bia, floor=1e-2)
    rf, rv = fn_ref(init, images_data, batches)
    f, v = fn(init, images_data, batches)          # None -> zero priors
    np.testing.assert_allclose(np.asarray(f), np.asarray(rf),
                               rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(v), np.asarray(rv),
                               rtol=1e-14, atol=0)


def test_factory_single_trace_across_prior_values(scene):
    """Different (lambda_diag, f_prior) VALUES must reuse one executable,
    like lasso's penalty_weights."""
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("eigfloor_prior", in_axes=bia, floor=1e-4)
    n_img, n_flux = np.asarray(init).shape
    lam1 = jnp.zeros((n_img, n_flux))
    fp1 = jnp.zeros((n_img, n_flux))
    lam2 = lam1.at[:, 1].set(1e9)      # pin source 1 hard...
    fp2 = fp1.at[:, 1].set(42.0)       # ...to 42
    f1, _ = fn(init, images_data, batches, lam1, fp1)
    f2, _ = fn(init, images_data, batches, lam2, fp2)
    jax.block_until_ready(f2)
    assert fn._jitted._cache_size() == 1
    np.testing.assert_allclose(np.asarray(f2)[:, 1], 42.0, rtol=1e-5)
    assert not np.allclose(np.asarray(f1)[:, 1], 42.0)


def test_factory_matches_sequential(scene):
    images_data, batches, init = scene
    bia = batches_in_axes(batches)
    fn = make_batched_solver("eigfloor_prior", in_axes=bia, floor=1e-4)
    n_img, n_flux = np.asarray(init).shape
    rng = np.random.default_rng(0)
    lam = jnp.asarray(rng.uniform(0.0, 2.0, size=(n_img, n_flux)))
    fp = jnp.asarray(rng.normal(size=(n_img, n_flux)) * 10.0)
    f, v = fn(init, images_data, batches, lam, fp)
    for i in range(n_img):
        init_i, single, sb = single_image(scene, i)
        fi, vi = solve_fluxes_eigfloor_prior(init_i, single, sb,
                                             lambda_diag=lam[i],
                                             f_prior=fp[i],
                                             return_variances=True,
                                             floor=1e-4)
        np.testing.assert_allclose(np.asarray(f[i]), np.asarray(fi),
                                   rtol=1e-10)
        np.testing.assert_allclose(np.asarray(v[i]), np.asarray(vi),
                                   rtol=1e-10)


def test_prior_arrays_from_slots():
    # catalog: 5 sources; image 0 sees {0->slot0, 2->slot1, 4->slot3},
    # image 1 sees {1->slot0, 3->slot2}; n_flux=5 (slot 4 = background/pad).
    slots = [{0: 0, 2: 1, 4: 3}, {1: 0, 3: 2}]
    f_prior = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
    sigma = np.array([2.0, 4.0, 0.0, 1.0, np.inf])
    lam, fp = prior_arrays_from_slots(slots, 2, 5, f_prior, sigma,
                                      protected=[0])
    lam = np.asarray(lam)
    fp = np.asarray(fp)
    assert lam.shape == fp.shape == (2, 5)
    assert np.all(np.isfinite(lam)) and np.all(np.isfinite(fp))
    # source 0: protected -> no prior
    assert lam[0, 0] == 0.0 and fp[0, 0] == 0.0
    # source 2: sigma=0 -> skipped (0/inf-safe)
    assert lam[0, 1] == 0.0
    # source 4: sigma=inf -> skipped
    assert lam[0, 3] == 0.0
    # source 1 in image 1: normal prior
    np.testing.assert_allclose(lam[1, 0], 1.0 / 16.0)
    assert fp[1, 0] == 20.0
    # source 3: f_prior=nan -> skipped
    assert lam[1, 2] == 0.0 and fp[1, 2] == 0.0
    # untouched slots (padding / background) stay 0
    assert lam[0, 2] == 0.0 and lam[0, 4] == 0.0 and lam[1, 4] == 0.0
    # boolean-mask form of `protected` behaves identically
    mask = np.array([True, False, False, False, False])
    lam_b, fp_b = prior_arrays_from_slots(slots, 2, 5, f_prior, sigma,
                                          protected=mask)
    np.testing.assert_array_equal(np.asarray(lam_b), lam)
    np.testing.assert_array_equal(np.asarray(fp_b), fp)
