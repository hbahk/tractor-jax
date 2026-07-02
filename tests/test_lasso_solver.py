"""Tests for the LASSO solver (solve_fluxes_lasso / _lasso_fista).

Test plan from proj-spherex-gpupipe/notebooks/research_notes/lasso_alpha/
05_implementation_plan.md (Stage B3), numbered as in the spec:

  1. sklearn parity (raw mode, conversion lambda_raw = n * alpha_sklearn)
  2. KKT oracle: reported kkt matches an independent numpy recomputation
  3. alpha=0 regressions: == solve_fluxes_linear (signed) and == NNLS (nonneg)
  4. path/support behavior: max-alpha empty support, protected always in,
     no NaN/inf along the path (no monotonicity assertion)
  5. grad-check through the debiased solve at fixed support
  6. vmap/sequential parity on a toy scene
  7. f32 vs f64 support agreement (Jacobi normalization sanity)

Run in the `spherex` conda env:  pytest tests/test_lasso_solver.py -q
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
    _lasso_fista,
    solve_fluxes_lasso,
    solve_fluxes_linear,
    extract_model_data,
    optimize_fluxes,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def random_problem(n=60, p=25, seed=0, spread=True):
    """Random correlated design with very different column norms."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=n)      # correlated pair
    if spread:
        X *= 10.0 ** rng.uniform(-1.5, 1.5, size=p)    # 3 dex of column norms
    w_true = np.zeros(p)
    w_true[rng.choice(p, size=5, replace=False)] = rng.uniform(1.0, 4.0, size=5)
    y = X @ w_true + 0.3 * rng.normal(size=n)
    return X, y, w_true


def toy_scene(n_img=1, H=24, W=24, noise_sigma=0.05, seed=3):
    """Small MoG-PSF scene with 4 point sources; returns (tractor, true_fluxes)."""
    rng = np.random.default_rng(seed)
    psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                             np.array([[[2.5, 0.0], [0.0, 2.5]]]))
    positions = [(6.3, 6.8), (8.1, 7.4), (16.6, 15.2), (18.9, 18.1)]
    true_fluxes = np.array([50.0, 8.0, 30.0, 0.0])     # one truly-zero source

    srcs = [PointSource(PixPos(x, y), Flux(1.0)) for (x, y) in positions]
    cat = Catalog(*srcs)

    imgs = []
    for i in range(n_img):
        img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)) / noise_sigma,
                    psf=psf, wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
        img.name = f"toy{i}"
        imgs.append(img)

    tr = Tractor(imgs, cat)
    images_data, batches, _ = extract_model_data(tr)
    from tractor_jax.jax.optimizer import render_image
    for i, img in enumerate(imgs):
        single = jax.tree_util.tree_map(lambda x: x[i], images_data)
        sb = {"PointSource": {
            "flux_idx": batches["PointSource"]["flux_idx"][i],
            "pos_pix": batches["PointSource"]["pos_pix"][i],
            "mask": batches["PointSource"]["mask"][i],
        }}
        model = np.array(render_image(jnp.array(true_fluxes), single, sb))[:H, :W]
        img.data += model + rng.normal(size=(H, W)) * noise_sigma
    return tr, true_fluxes


def single_image_inputs(tr):
    """extract_model_data -> single-image slices for direct solver calls."""
    images_data, batches, init_flux = extract_model_data(tr)
    single = jax.tree_util.tree_map(lambda x: x[0], images_data)
    sb = {"PointSource": {
        "flux_idx": batches["PointSource"]["flux_idx"][0],
        "pos_pix": batches["PointSource"]["pos_pix"][0],
        "mask": batches["PointSource"]["mask"][0],
    }}
    return single, sb, init_flux[0]


# --------------------------------------------------------------------------- #
# 1. sklearn parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nonneg", [False, True])
def test_sklearn_parity(nonneg):
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, _ = random_problem()
    n = X.shape[0]
    G = jnp.array(X.T @ X)
    b = jnp.array(X.T @ y)

    # 5 alphas spanning empty -> full support (sklearn units)
    a_max = np.max(np.abs(X.T @ y)) / n
    for a_sk in [2 * a_max, 0.5 * a_max, 0.05 * a_max, 5e-3 * a_max, 1e-4 * a_max]:
        lam = jnp.full(X.shape[1], n * a_sk)           # lambda_raw = n * alpha_sklearn
        f, kkt = _lasso_fista(G, b, lam, nonneg=nonneg, n_iter=20000, reg=0.0)
        f = np.array(f)
        ref = sklearn.Lasso(alpha=a_sk, fit_intercept=False, positive=nonneg,
                            tol=1e-14, max_iter=1000000).fit(X, y).coef_
        assert np.max(np.abs(f - ref)) < 1e-8, (a_sk, np.max(np.abs(f - ref)))
        assert np.array_equal(f > 0, ref > 0) or np.array_equal(f != 0, ref != 0)
        assert float(kkt) < 1e-8


# --------------------------------------------------------------------------- #
# 2. KKT oracle
# --------------------------------------------------------------------------- #
def test_kkt_matches_numpy():
    X, y, _ = random_problem(seed=1)
    G = X.T @ X
    b = X.T @ y
    lam = 0.3 * np.sqrt(np.clip(np.diag(G), 0, None))  # snr-style penalty
    f, kkt = _lasso_fista(jnp.array(G), jnp.array(b), jnp.array(lam),
                          nonneg=True, n_iter=20000, reg=0.0)
    f = np.array(f)
    # independent numpy recomputation in normalized (S/N) units
    D = np.sqrt(np.diag(G))
    grad = (G @ f - b) / D
    lamn = lam / D
    active = f > 0
    viol = np.where(active, np.abs(grad + lamn), np.maximum(-grad - lamn, 0.0))
    assert abs(float(kkt) - viol.max()) < 1e-10
    assert viol.max() < 1e-8


# --------------------------------------------------------------------------- #
# 3. alpha=0 regressions
# --------------------------------------------------------------------------- #
def test_alpha0_equals_linear_solver():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)
    rc = 1e-12
    f_lin = np.array(solve_fluxes_linear(f0, single, sb, rcond=rc))
    f_lasso = np.array(solve_fluxes_lasso(
        f0, single, sb, alpha=0.0, nonneg=False, debias=False,
        n_iter=20000, rcond=rc))
    assert np.max(np.abs(f_lasso - f_lin)) / np.max(np.abs(f_lin)) < 1e-6


def test_alpha0_nonneg_equals_nnls():
    from scipy.optimize import nnls
    from tractor_jax.jax.optimizer import _render_source_templates
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)

    T = np.array(_render_source_templates(single, sb, f0.shape[0]))
    A = T.reshape(f0.shape[0], -1).T
    w = np.array(single["invvar"]).ravel()
    d = np.array(single["data"]).ravel()
    f_ref, _ = nnls(A * np.sqrt(w)[:, None], d * np.sqrt(w))

    f = np.array(solve_fluxes_lasso(f0, single, sb, alpha=0.0, nonneg=True,
                                    debias=False, n_iter=50000, rcond=0.0))
    assert np.max(np.abs(f - f_ref)) < 1e-5 * max(1.0, np.max(np.abs(f_ref)))


# --------------------------------------------------------------------------- #
# 4. path / support behavior
# --------------------------------------------------------------------------- #
def test_path_support_behavior():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)

    # protected source 1 (the faint one) must survive any alpha
    wgt = np.ones(f0.shape[0]); wgt[1] = 0.0
    grid = np.array([0.5, 2.0, 8.0, 1e4])
    f, var, aux = solve_fluxes_lasso(
        f0, single, sb, penalty_weights=jnp.array(wgt),
        selection_mode="path", grid=jnp.array(grid), return_path=True,
        return_variances=True, return_aux=True, n_iter=5000)
    path = np.array(aux["path_fluxes"])
    assert np.all(np.isfinite(path))
    assert np.all(np.isfinite(np.array(aux["criterion_values"])))
    assert float(aux["kkt"]) < 1e-6

    # at an absurdly large alpha only the protected source stays
    f_hi, aux_hi = solve_fluxes_lasso(
        f0, single, sb, alpha=1e4, penalty_weights=jnp.array(wgt),
        return_aux=True, n_iter=5000)
    s_hi = np.array(aux_hi["support"])
    assert s_hi[1] == 1.0 and s_hi.sum() == 1.0

    # variances: finite on support, inf off support
    s = np.array(aux["support"]).astype(bool)
    v = np.array(var)
    assert np.all(np.isfinite(v[s])) and np.all(np.isinf(v[~s]))


# --------------------------------------------------------------------------- #
# 5. grad-check through the debiased solve (fixed support)
# --------------------------------------------------------------------------- #
def test_grad_through_debias():
    tr, _ = toy_scene()
    single, sb, f0 = single_image_inputs(tr)

    def total_flux(data):
        sd = dict(single); sd["data"] = data
        f = solve_fluxes_lasso(f0, sd, sb, alpha=1.0, n_iter=2000)
        return jnp.sum(f)

    g = jax.grad(total_flux)(single["data"])
    assert np.all(np.isfinite(np.array(g)))

    eps = 1e-4
    probe = np.zeros_like(np.array(single["data"])); probe[10, 10] = 1.0
    fp = total_flux(jnp.array(np.array(single["data"]) + eps * probe))
    fm = total_flux(jnp.array(np.array(single["data"]) - eps * probe))
    fd = (fp - fm) / (2 * eps)
    assert abs(float(fd) - float(np.array(g)[10, 10])) < 1e-6 * max(1.0, abs(float(fd)))


# --------------------------------------------------------------------------- #
# 6. vmap / sequential parity via optimize_fluxes
# --------------------------------------------------------------------------- #
def test_optimize_fluxes_vmap_vs_sequential():
    kw = dict(solver="lasso", penalty={"mode": "snr", "alpha": 1.5},
              return_variances=True, return_aux=True, lasso_n_iter=5000)
    tr1, _ = toy_scene(n_img=2)
    res_v = optimize_fluxes(tr1, vmap_images=True, use_sharding=False, **kw)
    tr2, _ = toy_scene(n_img=2)
    res_s = optimize_fluxes(tr2, vmap_images=False, use_sharding=False, **kw)

    for (fv, vv, av), (fs, vs, as_) in zip(res_v, res_s):
        assert np.allclose(fv, fs, rtol=1e-5, atol=1e-8)
        finite = np.isfinite(vv) & np.isfinite(vs)
        assert np.allclose(vv[finite], vs[finite], rtol=1e-5)
        assert np.array_equal(av["support"], as_["support"])
        assert float(av["kkt"]) < 1e-6 and float(as_["kkt"]) < 1e-6


# --------------------------------------------------------------------------- #
# 7. f32 vs f64 support agreement (run f32 in a subprocess-free way:
#    jax x64 is process-global, so emulate f32 by casting inputs down)
# --------------------------------------------------------------------------- #
def test_f32_support_agreement():
    X, y, _ = random_problem(n=200, p=60, seed=7)
    G64 = X.T @ X
    b64 = X.T @ y
    lam = 0.8 * np.sqrt(np.clip(np.diag(G64), 0, None))

    f64, _ = _lasso_fista(jnp.array(G64), jnp.array(b64), jnp.array(lam),
                          nonneg=True, n_iter=20000, reg=0.0)
    f32, _ = _lasso_fista(jnp.array(G64, dtype=jnp.float32),
                          jnp.array(b64, dtype=jnp.float32),
                          jnp.array(lam, dtype=jnp.float32),
                          nonneg=True, n_iter=20000, reg=0.0)
    s64 = np.array(f64) > 0
    s32 = np.array(f32) > 0
    assert np.array_equal(s64, s32)
    rel = np.max(np.abs(np.array(f32) - np.array(f64))) / max(1.0, np.max(np.abs(np.array(f64))))
    assert rel < 1e-3
