"""``pixel_integration="point"``: rendering with an effective PSF.

An *optical* PSF is integrated over each native pixel after the high-res render
(``"window"``, the historical path). An *effective* PSF (the SPHEREx R7 ePSF)
already contains the pixel response, so the high-res render is sampled at the
native pixel centres instead (``"point"``). The physics these tests pin down:

* the two are the same model. Box-convolving an optical kernel with the native
  pixel window on the high-res grid gives its effective kernel; rendering the
  optical kernel through ``"window"`` and the effective kernel through
  ``"point"`` must produce the same native templates, for point sources and
  galaxies, on the full padded grid and on the compact stamp (the box
  convolution commutes with the phase-ramp shift and with the galaxy
  convolution, so agreement is at FFT round-off);
* normalisation: a kernel of unit sum on the high-res grid gives unit-flux
  templates in both modes;
* the decimation is aligned with the block that the window path sums (odd
  factors take its middle sample, even factors the mean of the two middle ones)
  and refuses non-integer factors;
* the option reaches the batched solvers (fluxes recovered through the ePSF in
  point mode equal the optical/window truth), is differentiable (finite
  differences vs ``jax.grad`` through the point-mode renderer), and the default
  leaves the window path bit-identical;
* the CPU ``PixelizedPSF(pixel_integrated=True)`` point-samples instead of
  block-integrating, on both its patch and Fourier paths.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from tractor_jax.jax import batching as tjb
from tractor_jax.jax.optimizer import (
    _render_source_templates,
    render_batch_point_sources,
)
from tractor_jax.jax.rendering import (
    decimate_int_point,
    downsample_image,
    rebin_downsample_int_flux,
)

PIXSCALE_ARCSEC = 6.15
CD_INV = np.eye(2) * (3600.0 / PIXSCALE_ARCSEC)
H = W = 21
K = 5
S_STAMP = 80


class _Cat(dict):
    def __len__(self):
        return len(self["shape_r"])


def _optical_psf(n=61, sigma_hr=4.0):
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2
    k = np.exp(-0.5 * ((yy - c) ** 2 + (xx - c) ** 2) / sigma_hr ** 2)
    return k / k.sum()


def _effective_psf(optical, k=K):
    """The ePSF of an optical kernel: the fraction of the flux in a native pixel
    centred on each high-res sample (the k x k box sum), stored with unit sum on
    the high-res grid, as the SPHEREx product is. ``uniform_filter`` is that box
    sum divided by k^2, which is exactly the stored normalisation."""
    return uniform_filter(optical, size=k, mode="constant")


def _catalog():
    shape_r = np.array([0.0, 0.0, 0.0, 0.8, 1.5, 0.5, 30.0])
    sersic = np.array([0.0, 0.0, 0.0, 1.0, 4.0, 2.0, 4.0])
    shape_ab = np.array([1.0, 1.0, 1.0, 0.7, 0.5, 0.9, 0.8])
    shape_phi = np.array([0.0, 0.0, 0.0, 30.0, 120.0, 75.0, 10.0])
    sx = np.array([12.3, 16.6, 20.4, 13.2, 18.9, 12.7, 16.0])
    sy = np.array([4.1, 9.2, 3.6, 15.5, 20.4, 18.8, 12.5])
    cat = _Cat(shape_r=shape_r, sersic=sersic, shape_ab=shape_ab,
               shape_phi=shape_phi)
    return cat, sx, sy


def _views(psf, n_views=2):
    rng = np.random.default_rng(0)
    origins = [(0.0, 0.0), (12.0, 3.0)]
    return [{
        "data": rng.normal(size=(H, W)).astype(np.float32),
        "invvar": np.ones((H, W), np.float32),
        "psf": psf,
        "src_indices": list(range(7)),
        "origin": origins[i],
    } for i in range(n_views)]


def _build(psf, render_stamp=None, **kw):
    cat, sx, sy = _catalog()
    return tjb.build_padded_batches(
        _views(psf), cat, sx, sy, psf_sampling=1.0 / K, fixed_max_factor=float(K),
        fit_background=True, cd_inv=CD_INV, render_stamp=render_stamp, **kw)


def _single(bundle, i):
    imgd = jax.tree_util.tree_map(lambda a: a[i], bundle.images_data)
    bat = {}
    for key, val in bundle.batches.items():
        axes = bundle.in_axes[key]
        bat[key] = jax.tree_util.tree_map(
            lambda a, ax: a[i] if ax == 0 else a, val, axes)
    return imgd, bat


def _templates(bundle, i, **kw):
    imgd, bat = _single(bundle, i)
    n_flux = bundle.initial_fluxes.shape[1]
    return np.asarray(_render_source_templates(imgd, bat, n_flux,
                                               sampling_factor=float(K), **kw))


# --------------------------------------------------------------------------- #
# the decimation primitive
# --------------------------------------------------------------------------- #
def test_decimate_takes_the_block_centre_and_scales_by_k2():
    rng = np.random.default_rng(1)
    # constant within blocks: point == window
    blocks = rng.normal(size=(4, 6))
    img = jnp.asarray(np.kron(blocks, np.ones((5, 5))))
    assert np.allclose(decimate_int_point(img, 5, 5), rebin_downsample_int_flux(img, 5, 5))
    # linear ramp: the centre sample equals the block mean, so point == window
    yy, xx = np.indices((20, 30))
    ramp = jnp.asarray(0.3 * xx - 0.7 * yy + 2.0)
    assert np.allclose(decimate_int_point(ramp, 5, 5), rebin_downsample_int_flux(ramp, 5, 5),
                       atol=1e-9)
    # the sample really is the middle one (index 2 of [0, 5)) times k^2
    img = jnp.asarray(rng.normal(size=(10, 15)))
    expect = np.asarray(img)[2::5, 2::5] * 25
    assert np.allclose(decimate_int_point(img, 5, 5), expect)
    # even factor: mean of the two middle samples (indices 1, 2 of [0, 4))
    img4 = jnp.asarray(rng.normal(size=(8, 8)))
    a = np.asarray(img4)
    expect4 = 0.25 * (a[1::4, 1::4] + a[1::4, 2::4] + a[2::4, 1::4] + a[2::4, 2::4]) * 16
    assert np.allclose(decimate_int_point(img4, 4, 4), expect4)


def test_downsample_image_dispatch_and_integer_requirement():
    img = jnp.asarray(np.random.default_rng(2).normal(size=(50, 50)))
    assert np.array_equal(downsample_image(img, (10, 10)),
                          downsample_image(img, (10, 10), "window"))
    assert np.allclose(downsample_image(img, (10, 10), "point"),
                       decimate_int_point(img, 5, 5))
    with pytest.raises(ValueError, match="integer"):
        downsample_image(img, (11, 11), "point")
    with pytest.raises(ValueError, match="pixel_integration"):
        downsample_image(img, (10, 10), "boxcar")


# --------------------------------------------------------------------------- #
# the equivalence: optical + window == effective + point
# --------------------------------------------------------------------------- #
def _compare(t_ref, t_new, tol, tile=(slice(0, H), slice(0, W))):
    assert t_ref.shape == t_new.shape
    for slot in range(t_ref.shape[0]):
        peak = np.abs(t_ref[slot]).max()
        if peak == 0.0:
            assert np.abs(t_new[slot]).max() == 0.0
            continue
        diff = np.abs(t_new[slot][tile] - t_ref[slot][tile]).max() / peak
        assert diff < tol, f"slot {slot}: {diff}"


@pytest.mark.parametrize("render_stamp", [None, S_STAMP])
def test_effective_point_equals_optical_window(render_stamp):
    opt = _optical_psf()
    eff = _effective_psf(opt)
    b_opt = _build(opt, render_stamp=render_stamp)
    b_eff = _build(eff, render_stamp=render_stamp)
    for i in range(2):
        t_win = _templates(b_opt, i, pixel_integration="window")
        t_pt = _templates(b_eff, i, pixel_integration="point")
        _compare(t_win, t_pt, 2e-5)
        # and the wrong pairing is visibly different: the window twice
        t_wrong = _templates(b_eff, i, pixel_integration="window")
        peak = np.abs(t_win[0]).max()
        assert np.abs(t_wrong[0] - t_win[0]).max() / peak > 0.02


def test_default_is_the_window_path_bit_for_bit():
    opt = _optical_psf()
    b = _build(opt)
    for i in range(2):
        assert np.array_equal(_templates(b, i), _templates(b, i, pixel_integration="window"))


def test_point_mode_templates_have_unit_flux():
    eff = _effective_psf(_optical_psf())
    b = _build(eff)
    n_ps = b.meta["max_ps"]
    for i in range(2):
        t = _templates(b, i, pixel_integration="point")
        for slot in range(3):                      # the three point sources
            # sources near the tile edge lose flux to the clip; the first one
            # in view 0 sits well inside
            if i == 0 and slot == 0:
                assert abs(t[slot].sum() - 1.0) < 1e-3
        assert abs(t[b.meta["bg_idx"]].sum() - t.shape[1] * t.shape[2]) < 1e-6
        assert n_ps == 3


# --------------------------------------------------------------------------- #
# solvers, gradients, model rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", ["linear", "eigfloor"])
def test_batched_solver_recovers_fluxes_through_the_epsf(solver):
    opt = _optical_psf()
    eff = _effective_psf(opt)
    b_opt = _build(opt)
    b_eff = _build(eff)
    n_flux = b_opt.initial_fluxes.shape[1]
    truth = np.zeros(n_flux, np.float32)
    truth[:b_opt.meta["max_ps"]] = [3.0, 5.0, 2.0]
    truth[b_opt.meta["max_ps"]:b_opt.meta["max_ps"] + 4] = [4.0, 6.0, 1.5, 8.0]
    truth[b_opt.meta["bg_idx"]] = 0.2
    imgs = [np.tensordot(truth, _templates(b_opt, i), axes=(0, 0)) for i in range(2)]
    data_pad = np.stack(imgs).astype(np.float32)

    def solve(bundle, **kw):
        imgd = dict(bundle.images_data)
        imgd["data"] = jnp.asarray(data_pad)
        fn = tjb.make_batched_solver(solver, in_axes=bundle.in_axes, cache=False, **kw)
        f, _ = fn(bundle.initial_fluxes, imgd, bundle.batches)
        return np.asarray(f)

    kw = {"rcond": 1e-12} if solver == "linear" else {"floor": 1e-6}
    f_win = solve(b_opt, **kw)
    f_pt = solve(b_eff, pixel_integration="point", **kw)
    live = truth != 0
    for i in range(2):
        assert np.allclose(f_win[i][live], truth[live], rtol=2e-3, atol=2e-3)
        assert np.allclose(f_pt[i][live], truth[live], rtol=2e-3, atol=2e-3)


def test_point_mode_is_differentiable_in_position():
    """Finite differences vs jax.grad of a weighted model sum with respect to
    the source positions, through render_batch_point_sources in point mode."""
    eff = _effective_psf(_optical_psf())
    b = _build(eff)
    imgd, bat = _single(b, 0)
    psf_data = imgd["psf"]
    Hp, Wp = imgd["data"].shape
    rng = np.random.default_rng(3)
    weight = jnp.asarray(rng.normal(size=(Hp, Wp)).astype(np.float32))
    fluxes = jnp.asarray([3.0, 5.0, 2.0], jnp.float32)
    pos0 = bat["PointSource"]["pos_pix"][:3]

    def loss(pos):
        model = render_batch_point_sources(fluxes, pos, psf_data, (Hp, Wp),
                                           sampling_factor=float(K),
                                           pixel_integration="point")
        return jnp.sum(model * weight)

    g = np.asarray(jax.grad(loss)(pos0))
    h = 2e-2
    fd = np.zeros_like(g)
    for s in range(3):
        for a in range(2):
            e = np.zeros(pos0.shape, np.float32)
            e[s, a] = h
            fd[s, a] = (float(loss(pos0 + e)) - float(loss(pos0 - e))) / (2 * h)
    scale = np.abs(g).max()
    assert np.abs(g - fd).max() / scale < 2e-3, (g, fd)


def test_render_image_point_mode_matches_templates():
    from tractor_jax.jax.optimizer import render_image
    eff = _effective_psf(_optical_psf())
    b = _build(eff)
    imgd, bat = _single(b, 1)
    n_flux = b.initial_fluxes.shape[1]
    fl = np.zeros(n_flux, np.float32)
    fl[:3] = [1.0, 2.0, 0.5]
    fl[3:7] = [1.5, 0.7, 2.2, 3.0]
    fl[b.meta["bg_idx"]] = 0.1
    t = _templates(b, 1, pixel_integration="point")
    expect = np.tensordot(fl, t, axes=(0, 0))
    got = np.asarray(render_image(jnp.asarray(fl), imgd, bat, sampling_factor=float(K),
                                  pixel_integration="point"))
    assert np.abs(got - expect).max() < 1e-5 * np.abs(expect).max()


# --------------------------------------------------------------------------- #
# CPU PixelizedPSF
# --------------------------------------------------------------------------- #
def test_cpu_pixelized_psf_pixel_integrated():
    from tractor_jax.psf import PixelizedPSF
    opt = _optical_psf()
    eff = _effective_psf(opt)
    p_opt = PixelizedPSF(opt, sampling=1.0 / K)
    p_eff = PixelizedPSF(eff, sampling=1.0 / K, pixel_integrated=True)
    assert not p_opt.pixel_integrated and p_eff.pixel_integrated
    for px, py in [(10.0, 10.0), (10.3, 9.6), (11.45, 10.05)]:
        a = p_opt.getPointSourcePatch(px, py)
        e = p_eff.getPointSourcePatch(px, py)
        assert (a.x0, a.y0) == (e.x0, e.y0)
        ia, ie = a.patch, e.patch
        n = min(ia.shape[0], ie.shape[0]), min(ia.shape[1], ie.shape[1])
        ia, ie = ia[:n[0], :n[1]], ie[:n[0], :n[1]]
        assert abs(ie.sum() - 1.0) < 2e-3
        assert np.abs(ia - ie).max() / ia.max() < 3e-3
    # the Fourier (galaxy) path: same kernel from both objects
    Pa, ca, sha, _ = p_opt.getFourierTransform(10.3, 9.6, 12)
    Pe, ce, she, _ = p_eff.getFourierTransform(10.3, 9.6, 12)
    assert sha == she and ca == ce
    ka = np.fft.irfft2(Pa, s=sha)
    ke = np.fft.irfft2(Pe, s=she)
    assert np.abs(ka - ke).max() / ka.max() < 3e-3
    assert abs(ke.sum() - 1.0) < 2e-3
