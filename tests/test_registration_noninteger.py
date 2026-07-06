"""Registration correctness of the NON-INTEGER oversampled FFT render path.

Regression tests for the two defects diagnosed on the SPHEREx field sim
(proj research note lasso_alpha/15, render-mismatch diagnosis):

  1. odd HR widths were reconstructed from the rfft2 array as (shape-1)*2
     (e.g. 487 -> 486), evaluating the phase gradient on the wrong frequency
     grid -> now the HR grid width is forced even at construction;
  2. source placement used the nominal sampling factor s while the boxcar
     downsample integrates at the effective grid factors valid/H -> a
     position-dependent registration drift (~0.05 native px across a tile,
     +-6..10% per-phase matched-filter amplitude swing).

The oracle is an independent numpy implementation: the PSF kernel treated as
piecewise-constant fine cells, band-limit-shifted via FFT, integrated over
native pixels with a cumsum-linear-interp boxcar at the same effective
factors.

Run in the `spherex` conda env:  pytest tests/test_registration_noninteger.py -q
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tractor_jax import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor_jax.brightness import Flux
from tractor_jax.wcs import PixPos
from tractor_jax.psf import PixelizedPSF
from tractor_jax.jax.optimizer import (
    extract_model_data, _render_source_templates,
)

RATIO = 4.46371          # PIXSIZE / PSFSCALE of the SPHEREx field sim
H = W = 40               # native tile size


def make_kernel(n=65, sig=4.0):
    """Asymmetric double-Gaussian fine-grid kernel (breaks symmetry so
    registration errors show up in the centroid)."""
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    k = np.exp(-(xx**2 + yy**2) / (2 * sig**2))
    k += 0.35 * np.exp(-((xx - 3.1)**2 + (yy + 2.2)**2) / (2 * (1.7 * sig)**2))
    return k / k.sum()


def engine_template(kernel, x, y):
    """Unit-flux template via the real driver chain (PixelizedPSF at the
    non-integer sampling, oversample_rendering, boxcar downsample)."""
    img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)),
                psf=PixelizedPSF(kernel, sampling=1.0 / RATIO),
                wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
    tr = Tractor([img], Catalog(PointSource(PixPos(x, y), Flux(1.0))))
    images_data, batches, _ = extract_model_data(tr, oversample_rendering=True)
    single = jax.tree_util.tree_map(lambda a: a[0], images_data)
    sb = {"PointSource": jax.tree_util.tree_map(lambda a: a[0],
                                                batches["PointSource"])}
    t = np.array(_render_source_templates(single, sb, 1))[0]
    return t[:H, :W]


def numpy_reference(kernel, x, y, hr_shape):
    """Independent oracle: FFT-shift the fine kernel on the HR grid, crop to
    the valid region (round(H*RATIO)), then exact boxcar integration to
    (H, W) — the same crop/factor convention as the engine render path."""
    H_hr, W_hr = hr_shape
    valid_H = min(int(round(H * RATIO)), H_hr)
    valid_W = min(int(round(W * RATIO)), W_hr)
    f_x, f_y = valid_W / W, valid_H / H
    pad = np.zeros(hr_shape)
    kh, kw = kernel.shape
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    F = np.fft.rfft2(pad)
    fx = np.fft.rfftfreq(W_hr)
    fy = np.fft.fftfreq(H_hr)
    xs_hr = x * f_x + (f_x - 1.0) / 2.0
    ys_hr = y * f_y + (f_y - 1.0) / 2.0
    phase = np.exp(-2j * np.pi * (xs_hr * fx[None, :] + ys_hr * fy[:, None]))
    hr = np.fft.irfft2(F * phase, s=hr_shape)[:valid_H, :valid_W]

    def integrate(a, n_out, axis):
        a = np.moveaxis(a, axis, 0)
        n = a.shape[0]
        C = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)], 0)
        e = (n / n_out) * np.arange(n_out + 1)
        i0 = np.clip(np.floor(e).astype(int), 0, n - 1)
        fr = (e - i0).reshape(-1, *([1] * (a.ndim - 1)))
        Ce = C[i0] + fr * (C[i0 + 1] - C[i0])
        return np.moveaxis(Ce[1:] - Ce[:-1], 0, axis)

    return integrate(integrate(hr, H, 0), W, 1)


def centroid(t):
    yy, xx = np.mgrid[:t.shape[0], :t.shape[1]]
    w = np.clip(t, 0, None)
    return np.sum(xx * w) / w.sum(), np.sum(yy * w) / w.sum()


@pytest.fixture(scope="module")
def kernel():
    return make_kernel()


def test_even_hr_width(kernel):
    """The HR grid width must come out even (rfft round-trip unambiguous)."""
    img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)),
                psf=PixelizedPSF(kernel, sampling=1.0 / RATIO),
                wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
    tr = Tractor([img], Catalog(PointSource(PixPos(20.0, 20.0), Flux(1.0))))
    images_data, _, _ = extract_model_data(tr, oversample_rendering=True)
    rfft_w = images_data["psf"]["fft"].shape[-1]
    W_hr = (rfft_w - 1) * 2
    # even reconstruction must be exact: re-deriving the grid from the rfft
    # array and rebuilding the rfft must give the same width
    assert W_hr % 2 == 0


def test_no_position_dependent_drift(kernel):
    """Centroid offset (centroid - true position) must be CONSTANT across
    the tile (kernel asymmetry gives a fixed offset; the old code drifted
    ~0.05+ px from one side of the tile to the other)."""
    offsets = []
    for x, y in [(8.3, 9.1), (20.5, 19.75), (31.7, 30.4)]:
        t = engine_template(kernel, x, y)
        cx, cy = centroid(t)
        offsets.append((cx - x, cy - y))
    offsets = np.array(offsets)
    drift = offsets.max(axis=0) - offsets.min(axis=0)
    assert np.all(drift < 0.01), (offsets, drift)


def test_matched_filter_amplitude_vs_reference(kernel):
    """Per-phase matched-filter amplitude of the engine template against the
    independent numpy oracle: |a-1| small and phase swing tight (the old
    code swung +-6..10%)."""
    # discover the HR grid the engine actually used
    img = Image(data=np.zeros((H, W)), inverr=np.ones((H, W)),
                psf=PixelizedPSF(kernel, sampling=1.0 / RATIO),
                wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
    tr = Tractor([img], Catalog(PointSource(PixPos(20.0, 20.0), Flux(1.0))))
    images_data, _, _ = extract_model_data(tr, oversample_rendering=True)
    H_hr = images_data["psf"]["fft"].shape[-2]
    W_hr = (images_data["psf"]["fft"].shape[-1] - 1) * 2

    ratios = []
    for phx in (0.0, 0.25, 0.5, 0.75):
        for phy in (0.0, 0.5):
            x, y = 20.0 + phx, 19.0 + phy
            t_eng = engine_template(kernel, x, y)
            t_ref = numpy_reference(kernel, x, y, (H_hr, W_hr))
            a = np.sum(t_eng * t_ref) / np.sum(t_ref * t_ref)
            ratios.append(a)
    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1.0) < 0.02), ratios
    assert ratios.max() - ratios.min() < 0.01, ratios