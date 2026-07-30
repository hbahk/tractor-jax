"""Sub-pixel PSF re-registration as an FFT-domain phase ramp, plus the two
defensive fixes that live next to it in :mod:`tractor_jax.jax.batching`.

Why a phase ramp: a translation is exactly a linear phase in the Fourier
domain, and the renderer already holds ``rfft2(PSF)`` per view, so shifting the
kernel costs one elementwise multiply on an array in memory — no resampling, no
interpolation kernel, no extra render cost. The motivating measurement is the
delivered SPHEREx L2 PSF planes, whose CORE sits ~-0.05 native px from the
declared fiducial (CRPIX1=CRPIX2=51.0 of a 101x101, 10x-oversampled plane)
while the pipeline pins the array centre on the catalog position.

Oracles used here, in decreasing order of independence:

* the renderer's OWN position convention — a kernel ramped by ``(dy, dx)`` must
  render identically to the unramped kernel with the source moved by
  ``(+dx, +dy)``. This is what pins the SIGN;
* an analytic Gaussian re-evaluated at the shifted centre (a Gaussian is
  exactly shiftable, so this is an exact oracle for a kernel that decays to
  ~1e-16 inside its support);
* ``scipy.ndimage.shift`` order-5 splines, a genuinely different resampler;
* the -0.5 high-res px mis-centring that an EVEN kernel provably suffers under
  ``y0 = cy - ph // 2``.

Run in the `spherex` conda env:
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu pytest tests/test_psf_subpixel_shift.py -q
"""
import warnings

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.numpy.fft as jfft
from astropy.table import Table

from tractor_jax.jax.batching import (
    build_padded_batches,
    psf_fft_phase_ramp,
    psf_to_fft,
    shift_psf_fft,
)
from tractor_jax.jax.optimizer import render_batch_point_sources
from tractor_jax.jax.rendering import render_point_source_fft

PSF_SAMPLING = 0.2          # 5x oversampled PSF
TARGET_SAMPLING = 5.0       # 1 native px == 5 high-res px
GRID = (255, 260)           # H odd, W even (as _even_hr_width_pad guarantees)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def gauss(n=51, sigma=3.0, dy=0.0, dx=0.0):
    """Gaussian centred on the array fiducial ``(n - 1) / 2`` plus (dy, dx).

    ``(n - 1) / 2`` is the 0-based FITS fiducial (CRPIX - 1) and is also the
    centre ``jax.image.resize`` preserves, so it is the convention the
    center-pad in :func:`psf_to_fft` has to honour.

    Default 51x51 / sigma=3 decays to 8e-16 of the peak at the array edge, so
    truncation contributes nothing at the tolerances asserted below.
    """
    c = (n - 1) / 2.0
    y, x = np.mgrid[:n, :n]
    p = np.exp(-0.5 * ((x - c - dx) ** 2 + (y - c - dy) ** 2) / sigma ** 2)
    return p / p.sum()


def hr_image(psf_fft, shape=GRID):
    """Inverse transform back to the (ifftshift-ed) high-res grid."""
    return np.asarray(jfft.irfft2(psf_fft, s=shape))


def hr_centroid(img):
    """Flux centroid on an ifftshift-ed grid, unwrapped about the origin.

    Exact for a symmetric kernel: the coordinate axes are the *signed* integer
    offsets ``fftfreq(N) * N``, so no periodic branch cut crosses the kernel.
    """
    h, w = img.shape
    yy = np.fft.fftfreq(h) * h
    xx = np.fft.fftfreq(w) * w
    tot = img.sum()
    return ((img.sum(axis=1) * yy).sum() / tot,
            (img.sum(axis=0) * xx).sum() / tot)


def native_centroid(img):
    h, w = img.shape
    tot = img.sum()
    return ((img.sum(axis=1) * np.arange(h)).sum() / tot,
            (img.sum(axis=0) * np.arange(w)).sum() / tot)


def fft_of(psf, **kw):
    return psf_to_fft(psf, psf_sampling=PSF_SAMPLING, target_shape=GRID,
                      target_sampling=TARGET_SAMPLING, **kw)


def one_view_bundle(psf, xs=(19.5,), ys=(19.5,), view_keys=None, size=40,
                    **kw):
    """Single-view point-source-only bundle, positions in native px."""
    n = len(xs)
    tab = Table({"shape_r": np.zeros(n), "shape_ab": np.zeros(n),
                 "shape_phi": np.zeros(n), "sersic": np.zeros(n)})
    view = {"data": np.zeros((size, size)), "invvar": np.ones((size, size)),
            "psf": psf, "src_indices": list(range(n)), "origin": (0, 0)}
    if view_keys:
        view.update(view_keys)
    return build_padded_batches(
        [view], tab, np.asarray(xs, float), np.asarray(ys, float),
        psf_sampling=PSF_SAMPLING, **kw)


def render_native(bundle):
    """Render the bundle's sources at unit flux, at NATIVE resolution."""
    psf_data = {k: v[0] for k, v in bundle.images_data["psf"].items()}
    padded = tuple(np.asarray(bundle.images_data["data"]).shape[1:])
    n = bundle.batches["PointSource"]["pos_pix"].shape[1]
    return np.asarray(render_batch_point_sources(
        jnp.ones(n), bundle.batches["PointSource"]["pos_pix"][0], psf_data,
        padded, sampling_factor=TARGET_SAMPLING))


# --------------------------------------------------------------------------- #
# 1. default / zero shift is BIT-EXACT identity
# --------------------------------------------------------------------------- #
def test_no_shift_is_the_legacy_transform_exactly():
    """Not passing a shift must reproduce the pre-ramp result bit-for-bit."""
    psf = gauss()
    got = np.asarray(fft_of(psf))
    pad = np.zeros(GRID)
    cy, cx = GRID[0] // 2, GRID[1] // 2
    ph, pw = psf.shape
    pad[cy - ph // 2:cy - ph // 2 + ph, cx - pw // 2:cx - pw // 2 + pw] = psf
    ref = np.fft.rfft2(np.fft.ifftshift(pad))
    # the legacy code path is untouched, so this is exact to the FFT backend,
    # not merely close
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_zero_shift_ramp_is_exactly_one():
    ramp = psf_fft_phase_ramp((GRID[0], GRID[1] // 2 + 1), (0.0, 0.0))
    assert ramp.shape == (GRID[0], GRID[1] // 2 + 1)
    assert np.all(ramp == 1.0)                 # exactly 1, not 1 - 1e-17
    assert np.max(np.abs(np.abs(ramp) - 1.0)) == 0.0


def test_zero_shift_is_bit_exact_identity():
    """An explicitly requested zero shift must be BIT-identical, not close.

    The ramp is exactly ``1 - 0j`` there and the complex multiply is
    bit-preserving, so this holds through the real code path (no short-circuit
    is needed or used).
    """
    psf = gauss()
    base = np.asarray(fft_of(psf))
    zero = np.asarray(fft_of(psf, shift_hr=(0.0, 0.0)))
    assert np.array_equal(base, zero)
    assert np.array_equal(base.view(np.uint8), zero.view(np.uint8))

    also = np.asarray(shift_psf_fft(jnp.asarray(base), (0.0, 0.0)))
    assert np.array_equal(base, also)

    # ... and through the builder, in native-pixel units
    b0 = one_view_bundle(psf)
    bz = one_view_bundle(psf, view_keys={"psf_shift": (0.0, 0.0)})
    assert np.array_equal(np.asarray(b0.images_data["psf"]["fft"]),
                          np.asarray(bz.images_data["psf"]["fft"]))
    assert np.array_equal(render_native(b0), render_native(bz))


def test_zero_shift_bit_exact_on_the_psf_basis_path():
    psf = gauss()
    basis = [gauss(sigma=s) for s in (2.6, 3.0, 3.6)]
    w = np.array([0.5, 0.3, 0.2])
    keys = {"psf_basis": basis, "psf_weights": w}
    b0 = one_view_bundle(psf, view_keys=dict(keys))
    bz = one_view_bundle(psf, view_keys=dict(keys, psf_shift=(0.0, 0.0)))
    bzz = one_view_bundle(psf, view_keys=dict(
        keys, psf_basis_shifts=np.zeros((3, 2))))
    a = np.asarray(b0.images_data["psf"]["fft"])
    assert np.array_equal(a, np.asarray(bz.images_data["psf"]["fft"]))
    assert np.array_equal(a, np.asarray(bzz.images_data["psf"]["fft"]))


# --------------------------------------------------------------------------- #
# 2. a requested shift lands where requested, with the SIGN pinned
# --------------------------------------------------------------------------- #
SHIFTS_HR = [(0.0, 0.25), (0.25, 0.0), (-0.4, 0.7), (2.5, -1.75),
             (0.265, 0.265), (-0.265, -0.265)]


@pytest.mark.parametrize("dy,dx", SHIFTS_HR)
def test_requested_hr_shift_lands_where_requested(dy, dx):
    """High-res grid: measured translation == requested, to ~1e-6.

    Measured max |error| over these six cases: 4.4e-15 high-res px (float64).
    """
    psf = gauss()
    base = fft_of(psf)
    c0 = hr_centroid(hr_image(base))
    c1 = hr_centroid(hr_image(fft_of(psf, shift_hr=(dy, dx))))
    got = (c1[0] - c0[0], c1[1] - c0[1])
    assert abs(got[0] - dy) < 1e-6, f"dy {got[0]} != {dy}"
    assert abs(got[1] - dx) < 1e-6, f"dx {got[1]} != {dx}"
    # SIGN, stated separately so a flipped ramp fails here and not only above
    for req, meas in ((dy, got[0]), (dx, got[1])):
        if req != 0.0:
            assert np.sign(meas) == np.sign(req), (
                f"sign flip: requested {req}, measured {meas}")


def test_shift_sign_matches_the_renderer_position_convention():
    """The ramp's sign is the renderer's own sign, pinned against it.

    ``render_point_source_fft`` places a source at ``pos`` by multiplying the
    PSF FFT by ``exp(-2i pi (x f_x + y f_y))``. So ramping the kernel by
    ``(dy, dx)`` MUST equal leaving the kernel alone and moving the source to
    ``pos + (dx, dy)`` — positive shift moves content toward larger x/y.

    Measured: agreement 1.7e-17 for the correct sign against a 3.3e-3
    discrepancy (19% of the peak) for the flipped one, so a sign flip cannot
    slip through.
    """
    f0 = fft_of(gauss())
    dy, dx = 0.265, -0.4
    pos = jnp.array([100.0, 120.0])         # (x, y)
    ramped = np.asarray(render_point_source_fft(
        1.0, pos, shift_psf_fft(f0, (dy, dx)), GRID))
    moved_plus = np.asarray(render_point_source_fft(
        1.0, pos + jnp.array([dx, dy]), f0, GRID))
    moved_minus = np.asarray(render_point_source_fft(
        1.0, pos - jnp.array([dx, dy]), f0, GRID))
    peak = ramped.max()
    assert np.max(np.abs(ramped - moved_plus)) < 1e-6 * peak
    # and the wrong sign is nowhere near — this is the loud part
    assert np.max(np.abs(ramped - moved_minus)) > 1e-2 * peak


@pytest.mark.parametrize("dy,dx", [(0.0, 0.053), (0.05, 0.0), (0.05, 0.053),
                                   (-0.05, -0.053), (0.2, -0.1)])
def test_builder_shift_is_in_native_pixels(dy, dx):
    """``views[i]['psf_shift']`` is NATIVE px, recovered from the NATIVE render.

    The batch path maps at exactly ``target_sampling`` per axis
    (``target_h == padded_h * max_factor`` and ``_even_hr_width_pad`` keeps the
    width factor exact), so one native px is 5.0 high-res px with no slop.
    Measured max |error| over these five cases: 4.3e-8 native px, using a
    well-sampled sigma=2-native-px kernel so that the native-resolution
    centroid is itself faithful.
    """
    psf = gauss(n=101, sigma=2.0 * TARGET_SAMPLING)
    ref = render_native(one_view_bundle(psf))
    got = render_native(one_view_bundle(psf,
                                        view_keys={"psf_shift": (dy, dx)}))
    c0, c1 = native_centroid(ref), native_centroid(got)
    assert abs((c1[0] - c0[0]) - dy) < 1e-6
    assert abs((c1[1] - c0[1]) - dx) < 1e-6
    # units: a native-px request must NOT be read as high-res px (that would
    # land at dy/5, i.e. 5x too small)
    if dy != 0.0:
        assert abs((c1[0] - c0[0]) - dy / TARGET_SAMPLING) > 0.5 * abs(dy)


def test_spherex_core_offset_correction_sign():
    """The applied shift is MINUS the measured core offset.

    Mirrors the delivered-plane geometry: the kernel's core sits at
    ``-0.053`` native px from the array centre, and the pipeline puts the array
    centre on the source's catalog position. Passing ``psf_shift = -offset``
    must land the core ON the source; passing ``+offset`` must DOUBLE the error.

    Measured: uncorrected -0.05309 native px, corrected -1.6e-11, sign-flipped
    -0.10617 (exactly twice).
    """
    off = -0.053
    kern = gauss(n=101, sigma=2.0 * TARGET_SAMPLING,
                 dy=off * TARGET_SAMPLING, dx=off * TARGET_SAMPLING)
    clean = gauss(n=101, sigma=2.0 * TARGET_SAMPLING)
    c_ref = native_centroid(render_native(one_view_bundle(clean)))

    def core_offset(view_keys):
        c = native_centroid(render_native(
            one_view_bundle(kern, view_keys=view_keys)))
        return (c[0] - c_ref[0], c[1] - c_ref[1])

    raw = core_offset(None)
    fixed = core_offset({"psf_shift": (-off, -off)})       # correct: -measured
    flipped = core_offset({"psf_shift": (off, off)})       # the sign mistake

    assert abs(raw[0] - off) < 1e-6 and abs(raw[1] - off) < 1e-6
    assert abs(fixed[0]) < 1e-6, f"corrected core at dy={fixed[0]}"
    assert abs(fixed[1]) < 1e-6, f"corrected core at dx={fixed[1]}"
    # a sign flip does not merely fail to help, it doubles the error
    assert abs(flipped[0] - 2 * off) < 1e-6
    assert abs(flipped[1] - 2 * off) < 1e-6


# --------------------------------------------------------------------------- #
# 3. the ramp IS a shift: independent-resampler agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dy,dx", [(0.5, -0.3), (0.265, 0.265), (-1.2, 0.75)])
def test_ramp_equals_analytically_resampled_kernel(dy, dx):
    """Ramping == re-evaluating the (exactly shiftable) Gaussian off-centre.

    Measured max |difference| 6.9e-18 absolute, 4.0e-16 relative to the peak.
    """
    psf = gauss()
    got = hr_image(fft_of(psf, shift_hr=(dy, dx)))
    ref = hr_image(fft_of(gauss(dy=dy, dx=dx)))
    peak = got.max()
    assert np.max(np.abs(got - ref)) < 1e-6 * peak


@pytest.mark.parametrize("dy,dx", [(0.5, -0.3), (0.265, 0.265)])
def test_ramp_equals_spline_resampled_kernel(dy, dx):
    """Second, genuinely different resampler: order-5 splines.

    Measured max |difference| 5.6e-8 absolute = 3.2e-6 of the peak; that
    residual is the SPLINE's error (the analytic oracle above puts the ramp at
    4e-16), so the tolerance here is 1e-5 of the peak. The recovered centroid
    is checked at the tighter 1e-6.
    """
    ndi = pytest.importorskip("scipy.ndimage")
    psf = gauss()
    base = fft_of(psf)
    got = hr_image(fft_of(psf, shift_hr=(dy, dx)))
    centred = np.fft.fftshift(hr_image(base))
    spl = np.fft.ifftshift(ndi.shift(centred, (dy, dx), order=5,
                                     mode="constant"))
    peak = got.max()
    assert np.max(np.abs(got - spl)) < 1e-5 * peak
    c_got, c_spl = hr_centroid(got), hr_centroid(spl)
    assert abs(c_got[0] - c_spl[0]) < 1e-6
    assert abs(c_got[1] - c_spl[1]) < 1e-6


@pytest.mark.parametrize("dy,dx", [(0.265, -0.265), (1.7, 3.1), (-0.5, 0.5)])
def test_shift_then_unshift_round_trips(dy, dx):
    """Measured max |difference| 2.3e-16 on the FFT, 1.7e-18 on the image."""
    base = fft_of(gauss())
    back = shift_psf_fft(shift_psf_fft(base, (dy, dx)), (-dy, -dx))
    a, b = np.asarray(base), np.asarray(back)
    assert np.max(np.abs(b - a)) < 1e-12 * np.max(np.abs(a))
    assert np.max(np.abs(hr_image(back) - hr_image(base))) < 1e-12


# --------------------------------------------------------------------------- #
# 4. flux is conserved EXACTLY (unit-modulus ramp)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dy,dx", [(0.265, -0.265), (0.5, 0.5), (3.3, -7.1),
                                   (0.05, 0.053)])
def test_flux_conserved_exactly(dy, dx):
    """The DC bin — the total flux — is multiplied by exactly 1.

    ``fftfreq(H)[0] == rfftfreq(W)[0] == 0``, so ``ramp[0, 0]`` is exactly
    ``1 - 0j`` and the DC coefficient comes out BIT-identical. The summed image
    then differs only by the inverse transform's own rounding (measured
    <= 2.3e-16 relative).
    """
    base = fft_of(gauss())
    shifted = shift_psf_fft(base, (dy, dx))
    a, s = np.asarray(base), np.asarray(shifted)
    assert s[0, 0] == a[0, 0]                       # exact, not approximate
    assert np.array_equal(s[0:1, 0:1].view(np.uint8), a[0:1, 0:1].view(np.uint8))
    ramp = psf_fft_phase_ramp(a.shape, (dy, dx))
    assert ramp[0, 0] == 1.0
    assert np.max(np.abs(np.abs(ramp) - 1.0)) < 1e-15   # unit modulus
    s0, s1 = hr_image(base).sum(), hr_image(shifted).sum()
    assert abs(s1 / s0 - 1.0) < 1e-14


def test_ramp_preserves_the_fft_dtype():
    """Nothing already flowing through changes dtype.

    The ramp is built in float64 (it is pure geometry) but cast to the PSF
    FFT's own dtype at the multiply. Production runs WITHOUT ``jax_enable_x64``,
    where the FFT is complex64; measured there (fresh interpreter, x64 off):
    zero shift bit-exact, recovered shift 0.265 -> error 3.3e-7 high-res px,
    round trip 6.0e-8 absolute.
    """
    base = fft_of(gauss())
    assert base.dtype == jnp.complex128            # this module enables x64
    c64 = base.astype(jnp.complex64)
    out = shift_psf_fft(c64, (0.265, 0.265))
    assert out.dtype == jnp.complex64
    assert np.array_equal(np.asarray(shift_psf_fft(c64, (0.0, 0.0))),
                          np.asarray(c64))
    c0 = hr_centroid(hr_image(c64))
    c1 = hr_centroid(hr_image(out))
    assert abs((c1[0] - c0[0]) - 0.265) < 1e-5
    assert abs((c1[1] - c0[1]) - 0.265) < 1e-5


def test_flux_conserved_through_the_native_render():
    psf = gauss(n=101, sigma=2.0 * TARGET_SAMPLING)
    ref = render_native(one_view_bundle(psf)).sum()
    got = render_native(one_view_bundle(
        psf, view_keys={"psf_shift": (0.05, 0.053)})).sum()
    assert abs(got / ref - 1.0) < 1e-14


# --------------------------------------------------------------------------- #
# 5. per-basis-element ramps compose with zone blending
# --------------------------------------------------------------------------- #
def test_per_basis_shifts_compose_with_zone_blending():
    """``sum_k w_k * shift(K_k)`` == the ramped-basis result.

    Each zone kernel has its own core offset, so the ramp must be applied PER
    BASIS ELEMENT BEFORE the weighted blend. Measured: 0.0 in the FFT domain
    (identical arithmetic) and 5.2e-18 in the image domain against
    independently shifted-then-blended kernels.
    """
    basis = [gauss(sigma=s) for s in (2.6, 3.0, 3.6)]
    w = np.array([0.5, 0.3, 0.2])
    shifts_native = np.array([[0.05, 0.053], [-0.02, 0.01], [0.0, -0.04]])
    got = one_view_bundle(gauss(), view_keys={
        "psf_basis": basis, "psf_weights": w,
        "psf_basis_shifts": shifts_native})
    g = np.asarray(got.images_data["psf"]["fft"][0])
    assert g.shape == (GRID[0], GRID[1] // 2 + 1)

    ffts = [fft_of(k) for k in basis]
    hr = shifts_native * TARGET_SAMPLING
    oracle = sum(w[i] * shift_psf_fft(ffts[i], tuple(hr[i]))
                 for i in range(len(basis)))
    o = np.asarray(oracle)
    assert np.max(np.abs(g - o)) < 1e-6 * np.max(np.abs(o))

    # image domain: blend of individually shifted kernels
    blend = sum(w[i] * hr_image(shift_psf_fft(ffts[i], tuple(hr[i])))
                for i in range(len(basis)))
    got_img = hr_image(got.images_data["psf"]["fft"][0])
    assert np.max(np.abs(blend - got_img)) < 1e-6 * np.abs(got_img).max()

    # ... and this is NOT the same as one post-blend ramp when the shifts
    # differ, which is the whole reason the ramp goes on per element.
    naive = shift_psf_fft(sum(w[i] * ffts[i] for i in range(len(basis))),
                          tuple(hr.mean(axis=0)))
    n_img = hr_image(naive)
    assert np.max(np.abs(n_img - got_img)) > 1e-4 * np.abs(got_img).max()


def test_shared_shift_on_basis_path_equals_post_blend_ramp():
    """The degenerate case: one shift for every basis element.

    Then (and only then) per-element and post-blend ramping agree — measured
    2.2e-16 relative.
    """
    basis = [gauss(sigma=s) for s in (2.6, 3.0, 3.6)]
    w = np.array([0.5, 0.3, 0.2])
    shift_native = (0.05, 0.053)
    got = one_view_bundle(gauss(), view_keys={
        "psf_basis": basis, "psf_weights": w, "psf_shift": shift_native})
    ffts = [fft_of(k) for k in basis]
    post = shift_psf_fft(sum(w[i] * ffts[i] for i in range(len(basis))),
                         tuple(np.multiply(shift_native, TARGET_SAMPLING)))
    g, p = np.asarray(got.images_data["psf"]["fft"][0]), np.asarray(post)
    assert np.max(np.abs(g - p)) < 1e-9 * np.max(np.abs(p))
    # equal per-element shifts must also reproduce it exactly
    same = one_view_bundle(gauss(), view_keys={
        "psf_basis": basis, "psf_weights": w,
        "psf_basis_shifts": np.tile(shift_native, (3, 1))})
    assert np.array_equal(g, np.asarray(same.images_data["psf"]["fft"][0]))


def test_multi_view_basis_shifts_share_the_ramped_basis():
    """Views handing over the same shift table must all get the right FFT.

    The ramped basis is cached on (basis identity, shift table), so this
    exercises the reuse path across views with DIFFERENT zone weights.
    """
    basis = [gauss(sigma=s) for s in (2.6, 3.0, 3.6)]
    shifts_native = np.array([[0.05, 0.053], [-0.02, 0.01], [0.0, -0.04]])
    weights = [np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.1, 0.8]),
               np.array([0.0, 1.0, 0.0])]
    tab = Table({"shape_r": np.zeros(1), "shape_ab": np.zeros(1),
                 "shape_phi": np.zeros(1), "sersic": np.zeros(1)})
    views = [{"data": np.zeros((40, 40)), "invvar": np.ones((40, 40)),
              "psf": basis[0], "src_indices": [0], "origin": (0, 0),
              "psf_basis": basis, "psf_weights": w,
              "psf_basis_shifts": shifts_native} for w in weights]
    bundle = build_padded_batches(views, tab, np.array([19.5]),
                                  np.array([19.5]),
                                  psf_sampling=PSF_SAMPLING)
    ffts = [fft_of(k) for k in basis]
    hr = shifts_native * TARGET_SAMPLING
    shifted = [shift_psf_fft(ffts[i], tuple(hr[i])) for i in range(len(basis))]
    for vi, w in enumerate(weights):
        oracle = np.asarray(sum(w[i] * shifted[i] for i in range(len(basis))))
        got = np.asarray(bundle.images_data["psf"]["fft"][vi])
        assert np.max(np.abs(got - oracle)) < 1e-9 * np.max(np.abs(oracle))


def test_shift_psf_fft_broadcasts_over_a_stack():
    """One shift applied to a (K, H, Wf) basis stack == per-element calls."""
    basis = [gauss(sigma=s) for s in (2.6, 3.0, 3.6)]
    stack = jnp.stack([fft_of(k) for k in basis])
    shift = (0.265, -0.4)
    got = np.asarray(shift_psf_fft(stack, shift))
    assert got.shape == stack.shape
    for i, k in enumerate(basis):
        one = np.asarray(shift_psf_fft(fft_of(k), shift))
        assert np.array_equal(got[i], one)


def test_per_view_shifts_are_not_shared():
    """Two views, same PSF object, different shifts -> different FFTs."""
    psf = gauss()
    n = 1
    tab = Table({"shape_r": np.zeros(n), "shape_ab": np.zeros(n),
                 "shape_phi": np.zeros(n), "sersic": np.zeros(n)})

    def view(shift):
        return {"data": np.zeros((40, 40)), "invvar": np.ones((40, 40)),
                "psf": psf, "src_indices": [0], "origin": (0, 0),
                "psf_shift": shift}

    cache = {}
    bundle = build_padded_batches(
        [view((0.0, 0.0)), view((0.05, 0.053))], tab, np.array([19.5]),
        np.array([19.5]), psf_sampling=PSF_SAMPLING, psf_fft_cache=cache)
    f = np.asarray(bundle.images_data["psf"]["fft"])
    assert f.shape[0] == 2
    assert not np.array_equal(f[0], f[1])
    assert np.array_equal(f[0], np.asarray(fft_of(psf)))   # the zero-shift one
    assert len(cache) == 2                     # cache keys on the shift


# --------------------------------------------------------------------------- #
# 6. the even-parity guard
# --------------------------------------------------------------------------- #
def test_even_kernel_raises_by_default():
    """The guard fires, and names the exact consequence."""
    for n in (24, 50):
        with pytest.raises(ValueError, match=r"EVEN post-resize PSF size"):
            fft_of(gauss(n=n, sigma=3.0))
    with pytest.raises(ValueError, match=r"-0\.5 high-res px"):
        fft_of(gauss(n=50, sigma=3.0))
    # one even axis is enough
    even_w = gauss(n=51, sigma=3.0)[:, :-1]
    with pytest.raises(ValueError, match=r"axis 1 \(width 50\)"):
        fft_of(even_w)
    # odd is untouched
    fft_of(gauss(n=51, sigma=3.0))


def test_even_kernel_guard_fires_through_the_builder():
    with pytest.raises(ValueError, match=r"EVEN post-resize PSF size"):
        one_view_bundle(gauss(n=50, sigma=3.0))
    # and on the post-RESIZE shape: 25x25 at psf_sampling=0.5 resizes to 62x62
    with pytest.raises(ValueError, match=r"shape \(62, 62\)"):
        psf_to_fft(gauss(n=25, sigma=1.5), psf_sampling=0.5,
                   target_shape=GRID, target_sampling=TARGET_SAMPLING)


def test_even_kernel_warn_and_allow_reproduce_the_legacy_offset():
    """The legacy path really is -0.5 high-res px off, on each even axis."""
    kern = gauss(n=50, sigma=3.0)
    with pytest.warns(RuntimeWarning, match=r"-0\.5 high-res px"):
        warned = fft_of(kern, even_parity="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # "allow" must be silent
        allowed = fft_of(kern, even_parity="allow")
    assert np.array_equal(np.asarray(warned), np.asarray(allowed))
    cy, cx = hr_centroid(hr_image(allowed))
    assert abs(cy - (-0.5)) < 1e-9
    assert abs(cx - (-0.5)) < 1e-9


def test_even_kernel_fix_lands_on_the_origin():
    fixed = fft_of(gauss(n=50, sigma=3.0), even_parity="fix")
    cy, cx = hr_centroid(hr_image(fixed))
    assert abs(cy) < 1e-9
    assert abs(cx) < 1e-9
    # "fix" composes with a requested shift
    both = fft_of(gauss(n=50, sigma=3.0), even_parity="fix",
                  shift_hr=(0.265, -0.4))
    cy2, cx2 = hr_centroid(hr_image(both))
    assert abs(cy2 - 0.265) < 1e-9
    assert abs(cx2 - (-0.4)) < 1e-9


def test_odd_kernel_is_unaffected_by_every_parity_mode():
    psf = gauss(n=51, sigma=3.0)
    ref = np.asarray(fft_of(psf))
    for mode in ("raise", "warn", "fix", "allow"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # no mode may warn on odd
            got = np.asarray(fft_of(psf, even_parity=mode))
        assert np.array_equal(ref, got), mode


def test_bad_even_parity_mode_raises():
    with pytest.raises(ValueError, match="even_parity must be one of"):
        fft_of(gauss(), even_parity="fixup")


# --------------------------------------------------------------------------- #
# 7. shift-key validation (all-or-none, like psf_basis)
# --------------------------------------------------------------------------- #
def _two_views(psf, keys0=None, keys1=None):
    tab = Table({"shape_r": np.zeros(1), "shape_ab": np.zeros(1),
                 "shape_phi": np.zeros(1), "sersic": np.zeros(1)})
    vs = []
    for keys in (keys0, keys1):
        v = {"data": np.zeros((40, 40)), "invvar": np.ones((40, 40)),
             "psf": psf, "src_indices": [0], "origin": (0, 0)}
        if keys:
            v.update(keys)
        vs.append(v)
    return vs, tab


def test_psf_shift_all_or_none():
    psf = gauss()
    vs, tab = _two_views(psf, keys0={"psf_shift": (0.05, 0.05)})
    with pytest.raises(ValueError, match="psf_shift must be given for every"):
        build_padded_batches(vs, tab, np.array([19.5]), np.array([19.5]),
                             psf_sampling=PSF_SAMPLING)


def test_psf_basis_shifts_all_or_none():
    psf = gauss()
    basis = [gauss(sigma=s) for s in (2.6, 3.0)]
    w = np.array([0.5, 0.5])
    common = {"psf_basis": basis, "psf_weights": w}
    vs, tab = _two_views(
        psf, keys0=dict(common, psf_basis_shifts=np.zeros((2, 2))),
        keys1=dict(common))
    with pytest.raises(ValueError,
                       match="psf_basis_shifts must be given for every"):
        build_padded_batches(vs, tab, np.array([19.5]), np.array([19.5]),
                             psf_sampling=PSF_SAMPLING)


def test_psf_shift_and_basis_shifts_are_mutually_exclusive():
    basis = [gauss(sigma=s) for s in (2.6, 3.0)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        one_view_bundle(gauss(), view_keys={
            "psf_basis": basis, "psf_weights": np.array([0.5, 0.5]),
            "psf_basis_shifts": np.zeros((2, 2)), "psf_shift": (0.0, 0.0)})


def test_basis_shifts_require_a_basis():
    with pytest.raises(ValueError, match="psf_basis_shifts requires psf_basis"):
        one_view_bundle(gauss(),
                        view_keys={"psf_basis_shifts": np.zeros((2, 2))})


def test_bad_shift_shapes_raise():
    with pytest.raises(ValueError, match=r"\(dy, dx\) pair"):
        one_view_bundle(gauss(), view_keys={"psf_shift": (0.1, 0.2, 0.3)})
    basis = [gauss(sigma=s) for s in (2.6, 3.0)]
    with pytest.raises(ValueError, match=r"shape \(K, 2\)"):
        one_view_bundle(gauss(), view_keys={
            "psf_basis": basis, "psf_weights": np.array([0.5, 0.5]),
            "psf_basis_shifts": np.zeros((3, 2))})


# --------------------------------------------------------------------------- #
# 8. float64 source positions
# --------------------------------------------------------------------------- #
def test_source_positions_are_float64_and_exact():
    """float32 storage quantised requested positions; float64 does not.

    Measured for these five positions: float32 max |error| 7.63e-07 native px
    (x=10.2 -> 10.199999809265137, x=33.7 -> 33.70000076293945), against
    exactly 0.0 in float64. The rendered consequence is 3.0e-07 absolute
    (8.7e-07 of the peak) and a 1.9e-07 native px centroid shift — small, but
    pure loss, and the same KIND of error as the PSF core offset the phase ramp
    above corrects.
    """
    xs = (10.2, 19.5, 33.7, 7.125, 28.4)
    ys = (11.3, 20.75, 30.2, 9.9, 26.6)
    bundle = one_view_bundle(gauss(), xs=xs, ys=ys)
    pos = bundle.batches["PointSource"]["pos_pix"]
    assert pos.dtype == jnp.float64            # under jax_enable_x64
    want = np.stack([np.asarray(xs), np.asarray(ys)], axis=1)
    got = np.asarray(pos[0])
    assert np.max(np.abs(got - want)) == 0.0

    quantised = want.astype(np.float32).astype(np.float64)
    assert np.max(np.abs(quantised - want)) > 1e-7      # the bug being fixed
    psf_data = {k: v[0] for k, v in bundle.images_data["psf"].items()}
    padded = tuple(np.asarray(bundle.images_data["data"]).shape[1:])
    r64 = np.asarray(render_batch_point_sources(
        jnp.ones(len(xs)), jnp.asarray(want), psf_data, padded,
        sampling_factor=TARGET_SAMPLING))
    r32 = np.asarray(render_batch_point_sources(
        jnp.ones(len(xs)), jnp.asarray(quantised), psf_data, padded,
        sampling_factor=TARGET_SAMPLING))
    assert np.max(np.abs(r64 - r32)) > 0.0     # it really did change the image
    assert np.max(np.abs(r64 - r32)) < 1e-5 * r64.max()


def test_galaxy_positions_are_float64_too():
    tab = Table({"shape_r": np.array([2.0]), "shape_ab": np.array([0.7]),
                 "shape_phi": np.array([30.0]), "sersic": np.array([1.0])})
    view = {"data": np.zeros((40, 40)), "invvar": np.ones((40, 40)),
            "psf": gauss(), "src_indices": [0], "origin": (0, 0)}
    bundle = build_padded_batches(
        [view], tab, np.array([10.2]), np.array([11.3]),
        psf_sampling=PSF_SAMPLING, cd_inv=np.eye(2) * 585.0)
    pos = bundle.batches["Galaxy"]["pos_pix"]
    assert pos.dtype == jnp.float64
    assert float(pos[0, 0, 0]) == 10.2
    assert float(pos[0, 0, 1]) == 11.3
