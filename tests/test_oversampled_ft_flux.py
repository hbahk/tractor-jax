"""Regression: FFT-convolved (galaxy) models through an oversampled PixelizedPSF
must carry unit flux, matching the point-source patch path.

_getOversampledFourierTransform used to omit the 1/sampling**2 pixel-area factor
(_sampleImage point-samples the oversampled stamp), so OO galaxy renders came out
~oversample**2 (=25x at sampling=0.2) too faint while point sources were correct.
"""

import numpy as np
import pytest

from tractor_jax import ConstantSky, Flux, Image, NullWCS, PixPos, PointSource, Tractor
from tractor_jax.galaxy import GalaxyShape
from tractor_jax.psf import PixelizedPSF
from tractor_jax.sersic import SersicGalaxy, SersicIndex


def _oversampled_gaussian_psf(n=51, oversamp=5, fwhm_native=2.5):
    sigma = fwhm_native / 2.3548200450309493 * oversamp
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma ** 2))
    return (p / p.sum()).astype(np.float32)


def _image(psf):
    return Image(data=np.zeros((40, 40), np.float32),
                 inverr=np.ones((40, 40), np.float32),
                 psf=psf, wcs=NullWCS(pixscale=6.15), sky=ConstantSky(0.0))


def test_galaxy_ft_render_carries_unit_flux():
    psf = PixelizedPSF(_oversampled_gaussian_psf(), sampling=0.2)
    tim = _image(psf)
    gal = SersicGalaxy(PixPos(18.0, 30.0), Flux(1.0),
                       GalaxyShape(1.5, 0.8, 30.0), SersicIndex(1.0))
    total = float(np.asarray(Tractor([tim], [gal]).getModelImage(0)).sum())
    assert total == pytest.approx(1.0, abs=0.02)


def test_galaxy_matches_point_source_flux_scale():
    psf = PixelizedPSF(_oversampled_gaussian_psf(), sampling=0.2)
    tim = _image(psf)
    ps_total = float(np.asarray(
        Tractor([tim], [PointSource(PixPos(12.0, 14.0), Flux(1.0))])
        .getModelImage(0)).sum())
    gal_total = float(np.asarray(
        Tractor([tim], [SersicGalaxy(PixPos(18.0, 30.0), Flux(1.0),
                                     GalaxyShape(1.5, 0.8, 30.0),
                                     SersicIndex(1.0))])
        .getModelImage(0)).sum())
    # both unit-flux sources must render with the same total (no 25x mismatch)
    assert gal_total == pytest.approx(ps_total, rel=0.02)
