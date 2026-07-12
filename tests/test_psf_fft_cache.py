"""Tests for the PSF-FFT cache in extract_model_data (PixelizedPSF path).

The padded oversampled rfft2 depends only on the PSF stamp and the target
geometry, so it is stapled on the PSF object (psf._jax_fft_cache) and reused
by later extracts — bit-identical to a fresh compute, with the cached array
reused by identity.

Run in the `spherex` conda env:  pytest tests/test_psf_fft_cache.py -q
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from tractor_jax import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor_jax.brightness import Flux
from tractor_jax.wcs import PixPos
from tractor_jax.psf import PixelizedPSF
from tractor_jax.jax.optimizer import extract_model_data


def make_tractor(psf):
    rng = np.random.default_rng(5)
    srcs = [PointSource(PixPos(8.2, 9.1), Flux(20.0)),
            PointSource(PixPos(16.4, 14.7), Flux(5.0))]
    img = Image(data=rng.normal(0, 0.1, (24, 24)),
                inverr=np.ones((24, 24)) * 10.0,
                psf=psf, wcs=NullWCS(pixscale=1.0), sky=ConstantSky(0.0))
    return Tractor([img], Catalog(*srcs))


def gaussian_stamp(n=25, sigma=2.0):
    y, x = np.mgrid[:n, :n] - n // 2
    p = np.exp(-0.5 * (x * x + y * y) / sigma ** 2)
    return p / p.sum()


def test_cache_hit_and_bit_identity():
    psf = PixelizedPSF(gaussian_stamp(), sampling=0.2)
    tr = make_tractor(psf)
    d1, b1, f1 = extract_model_data(tr, oversample_rendering=True)
    assert hasattr(psf, "_jax_fft_cache") and len(psf._jax_fft_cache) == 1
    (cached,) = psf._jax_fft_cache.values()
    d2, b2, f2 = extract_model_data(tr, oversample_rendering=True)
    assert len(psf._jax_fft_cache) == 1                 # no new entry
    assert d2["psf"]["fft"] is not None
    # second extract reused the cached transform (values identical)
    assert np.array_equal(np.asarray(d1["psf"]["fft"]),
                          np.asarray(d2["psf"]["fft"]))
    assert np.array_equal(np.asarray(d1["psf"]["fft"][0]),
                          np.asarray(cached))


def test_cache_off_matches_cache_on():
    psf_a = PixelizedPSF(gaussian_stamp(), sampling=0.2)
    psf_b = PixelizedPSF(gaussian_stamp(), sampling=0.2)
    da, _, _ = extract_model_data(make_tractor(psf_a),
                                  oversample_rendering=True)
    db, _, _ = extract_model_data(make_tractor(psf_b),
                                  oversample_rendering=True,
                                  use_psf_fft_cache=False)
    assert not hasattr(psf_b, "_jax_fft_cache") or not psf_b._jax_fft_cache
    assert np.array_equal(np.asarray(da["psf"]["fft"]),
                          np.asarray(db["psf"]["fft"]))


def test_distinct_geometry_gets_new_entry():
    psf = PixelizedPSF(gaussian_stamp(), sampling=0.2)
    tr = make_tractor(psf)
    extract_model_data(tr, oversample_rendering=True)
    extract_model_data(tr, oversample_rendering=True,
                       fixed_target_shape=(200, 200), fixed_max_factor=5.0)
    assert len(psf._jax_fft_cache) == 2
