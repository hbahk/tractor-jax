"""Tests for tractor_jax.jax.batching.build_padded_batches.

The builder packs per-view (tile / window) sub-problems of one parent
catalog into a single padded vmap-ready batch: shared-PSF FFT computed once
and broadcast, per-view source arrays padded to common (or capped) widths in
the [ps | gal | bg] flux layout, MoG profiles looked up once per galaxy.

Oracles: a batched build must solve identically to per-view single builds
(PSF-share + common-pad exactness), caps must be output-invariant under
eigfloor/x64 (Jacobi dead-slot pinning), and overflowing a cap must raise.

Run in the `spherex` conda env:  pytest tests/test_batching.py -q
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.table import Table

from tractor_jax.jax.batching import (
    BatchBundle,
    batches_in_axes,
    build_padded_batches,
    clear_solver_cache,
    make_batched_solver,
    psf_to_fft,
    slice_fluxes,
)


# --------------------------------------------------------------------------- #
# synthetic parent scene: raw arrays + catalog table (engine batch-path style)
# --------------------------------------------------------------------------- #
PSF_SAMPLING = 0.2      # 5x oversampled PSF


def gaussian_psf(n=25, sigma=4.0):
    y, x = np.mgrid[:n, :n] - n // 2
    p = np.exp(-0.5 * (x * x + y * y) / sigma ** 2)
    return p / p.sum()


def parent_scene(seed=7, H=40, W=40, n_ps=5, n_gal=3):
    rng = np.random.default_rng(seed)
    sx = rng.uniform(3, W - 3, n_ps + n_gal)
    sy = rng.uniform(3, H - 3, n_ps + n_gal)
    shape_r = np.zeros(n_ps + n_gal)
    shape_r[n_ps:] = rng.uniform(1.0, 4.0, n_gal)
    tab = Table({
        "shape_r": shape_r,
        "shape_ab": np.where(shape_r > 0, rng.uniform(0.4, 1.0,
                                                      n_ps + n_gal), 0.0),
        "shape_phi": rng.uniform(0.0, 180.0, n_ps + n_gal),
        "sersic": np.where(shape_r > 0, rng.choice([1.0, 2.5, 4.0],
                                                   n_ps + n_gal), 0.0),
    })
    data = rng.normal(10.0, 2.0, (H, W))
    invvar = np.full((H, W), 4.0)
    psf = gaussian_psf()
    cd_inv = np.linalg.inv(np.eye(2) * (6.15 / 3600.0))
    return dict(data=data, invvar=invvar, psf=psf, tab=tab,
                sx=sx, sy=sy, cd_inv=cd_inv, H=H, W=W)


def carve_views(scene, size=20, origins=((0, 0), (20, 0), (10, 15)),
                psf_per_view=None):
    views = []
    sx, sy = scene["sx"], scene["sy"]
    for i, (x0, y0) in enumerate(origins):
        ids = [ci for ci in range(len(sx))
               if x0 <= sx[ci] < x0 + size and y0 <= sy[ci] < y0 + size]
        views.append({
            "data": np.ascontiguousarray(
                scene["data"][y0:y0 + size, x0:x0 + size]),
            "invvar": np.ascontiguousarray(
                scene["invvar"][y0:y0 + size, x0:x0 + size]),
            "psf": (psf_per_view[i] if psf_per_view is not None
                    else scene["psf"]),
            "src_indices": ids,
            "origin": (x0, y0),
        })
    return views


def build(scene, views, **kw):
    return build_padded_batches(views, scene["tab"], scene["sx"], scene["sy"],
                                psf_sampling=PSF_SAMPLING,
                                cd_inv=scene["cd_inv"], **kw)


def solve(bundle, solver="linear", **kw):
    fn = make_batched_solver(solver, in_axes=bundle.in_axes, cache=False, **kw)
    f, v = fn(bundle.initial_fluxes, bundle.images_data, bundle.batches)
    return np.asarray(f), np.asarray(v)


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_solver_cache()
    yield
    clear_solver_cache()


# --------------------------------------------------------------------------- #
# batched build == per-view single builds (PSF-share + common-pad exactness)
# --------------------------------------------------------------------------- #
def test_batched_equals_perview_singles():
    scene = parent_scene()
    views = carve_views(scene)
    bundle = build(scene, views)
    f_b, _ = solve(bundle)
    batched = slice_fluxes(f_b, bundle.meta)
    for i, view in enumerate(views):
        b1 = build(scene, [view])
        f1, _ = solve(b1)
        single = slice_fluxes(f1, b1.meta)[0]
        np.testing.assert_allclose(batched[i], single, rtol=1e-9, atol=1e-12)


def test_bundle_structure_and_meta():
    scene = parent_scene()
    views = carve_views(scene)
    bundle = build(scene, views)
    assert isinstance(bundle, BatchBundle)
    assert bundle.in_axes == batches_in_axes(bundle.batches)
    meta = bundle.meta
    n_views = len(views)
    assert len(meta["src_slot"]) == n_views == len(meta["counts"])
    assert meta["n_flux"] == meta["max_ps"] + meta["max_gal"] + 1
    assert meta["bg_idx"] == meta["max_ps"] + meta["max_gal"]
    for (n_ps, n_gal), slots, view in zip(meta["counts"], meta["src_slot"],
                                          views):
        assert n_ps + n_gal == len(view["src_indices"])
        assert set(slots) == set(view["src_indices"])
    # init fluxes seeded from the data pixel under each source
    init = np.asarray(bundle.initial_fluxes)
    slots0 = meta["src_slot"][0]
    for ci, slot in slots0.items():
        x0, y0 = views[0]["origin"]
        ix = int(round(float(scene["sx"][ci]) - x0))
        iy = int(round(float(scene["sy"][ci]) - y0))
        assert init[0, slot] == np.float32(views[0]["data"][iy, ix])


# --------------------------------------------------------------------------- #
# shared-PSF FFT: broadcast equals per-view stack exactly
# --------------------------------------------------------------------------- #
def test_shared_psf_broadcast_equals_stack():
    scene = parent_scene()
    shared = carve_views(scene)                                # same object
    copies = carve_views(scene, psf_per_view=[scene["psf"].copy()
                                              for _ in range(3)])
    b_shared = build(scene, shared)
    b_copies = build(scene, copies)
    fft_s = np.asarray(b_shared.images_data["psf"]["fft"])
    fft_c = np.asarray(b_copies.images_data["psf"]["fft"])
    assert np.array_equal(fft_s, fft_c)
    f_s, _ = solve(b_shared)
    f_c, _ = solve(b_copies)
    assert np.array_equal(f_s, f_c)


def test_psf_fft_cache_reuse():
    scene = parent_scene()
    views = carve_views(scene)
    cache = {}
    b1 = build(scene, views, psf_fft_cache=cache)
    assert len(cache) == 1
    (fft1,) = cache.values()
    b2 = build(scene, views, psf_fft_cache=cache)
    assert len(cache) == 1
    (fft2,) = cache.values()
    assert fft2 is fft1                     # reused, not recomputed
    assert np.array_equal(np.asarray(b1.images_data["psf"]["fft"]),
                          np.asarray(b2.images_data["psf"]["fft"]))
    # and equal to an uncached build
    b3 = build(scene, views)
    assert np.array_equal(np.asarray(b3.images_data["psf"]["fft"]),
                          np.asarray(b1.images_data["psf"]["fft"]))


def test_psf_to_fft_no_resample_when_matched():
    psf = gaussian_psf()
    fft = psf_to_fft(psf, psf_sampling=PSF_SAMPLING, target_shape=(125, 125),
                     target_sampling=5.0)
    pad = np.zeros((125, 125))
    cy = cx = 125 // 2
    pad[cy - 12:cy + 13, cx - 12:cx + 13] = psf
    ref = np.fft.rfft2(np.fft.ifftshift(pad))
    np.testing.assert_allclose(np.asarray(fft), ref, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# F2 caps: pad-invariance (eigfloor/x64) and overflow guard
# --------------------------------------------------------------------------- #
def test_caps_pad_invariance_eigfloor():
    scene = parent_scene()
    views = carve_views(scene)
    plain = build(scene, views)
    capped = build(scene, views, max_ps_cap=9, max_gal_cap=7, max_mog_k_cap=12)
    f0, _ = solve(plain, "eigfloor", floor=1e-2)
    f1, _ = solve(capped, "eigfloor", floor=1e-2)
    a = slice_fluxes(f0, plain.meta)
    b = slice_fluxes(f1, capped.meta)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


def test_cap_overflow_raises():
    scene = parent_scene()
    views = carve_views(scene)
    with pytest.raises(ValueError, match="exceeds cap"):
        build(scene, views, max_ps_cap=1)
    with pytest.raises(ValueError, match="exceeds cap"):
        build(scene, views, max_gal_cap=0)
    with pytest.raises(ValueError, match="exceeds cap"):
        build(scene, views, max_mog_k_cap=2)


def test_capped_shapes_are_fixed():
    scene = parent_scene()
    views = carve_views(scene)
    capped = build(scene, views, max_ps_cap=9, max_gal_cap=7,
                   max_mog_k_cap=12)
    assert capped.batches["PointSource"]["mask"].shape == (3, 9)
    assert capped.batches["Galaxy"]["mask"].shape == (3, 7)
    assert capped.batches["Galaxy"]["profile"]["amp"].shape == (3, 7, 12)
    assert capped.meta["n_flux"] == 9 + 7 + 1


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_empty_views_raises():
    scene = parent_scene()
    with pytest.raises(ValueError, match="views is empty"):
        build(scene, [])


def test_mismatched_view_shapes_raise():
    scene = parent_scene()
    views = carve_views(scene)
    views[1]["data"] = views[1]["data"][:-1]
    with pytest.raises(ValueError, match="same data shape"):
        build(scene, views)
