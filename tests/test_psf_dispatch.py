"""Every supported PSF type must reach the batched solver.

Regression: `extract_model_data` dispatched on `isinstance(psf, PixelizedPSF)`
and `isinstance(psf, GaussianMixturePSF)` with a silent fall-through, so a
`NCircularGaussianPSF` or a `HybridPixelizedPSF` rendered as an all-zero
template and `optimize_fluxes` returned flux 0 with infinite variance for every
source — no error, no warning. The quickstart's own PSF hit this.
"""

import numpy as np
import pytest

from tractor_jax import (Catalog, Flux, GaussianMixturePSF,
                         HybridPixelizedPSF, Image, NCircularGaussianPSF,
                         PixelizedPSF, PixPos, PointSource, Tractor)
from tractor_jax.jax.optimizer import optimize_fluxes, psf_kind

H = W = 50
NOISE = 0.5
SIGMA = 1.5
TRUTH = [(24.0, 27.0, 100.0), (31.0, 18.0, 40.0)]


def _mog_psf():
    return GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                              np.array([[[SIGMA ** 2, 0.0], [0.0, SIGMA ** 2]]]))


def _pixelized_psf(n=21):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * SIGMA ** 2))
    return PixelizedPSF((g / g.sum()).astype(np.float32))


def _fit(psf):
    catalog = Catalog(*[PointSource(PixPos(x, y), Flux(f)) for x, y, f in TRUTH])
    image = Image(data=np.zeros((H, W)), inverr=np.full((H, W), 1.0 / NOISE),
                  psf=psf)
    tractor = Tractor([image], catalog)
    clean = np.asarray(tractor.getModelImage(0))
    rng = np.random.default_rng(0)
    tractor.images[0].data = clean + rng.normal(0.0, NOISE, (H, W))
    fluxes, variances = optimize_fluxes(tractor, solver="eigfloor",
                                        return_variances=True,
                                        use_sharding=False)[0]
    return np.asarray(fluxes), np.asarray(variances)


@pytest.mark.parametrize("name", ["ncircular", "mog", "pixelized", "hybrid"])
def test_every_psf_type_is_fitted(name):
    psf = {"ncircular": lambda: NCircularGaussianPSF([SIGMA], [1.0]),
           "mog": _mog_psf,
           "pixelized": _pixelized_psf,
           "hybrid": lambda: HybridPixelizedPSF(_pixelized_psf(),
                                                gauss=_mog_psf())}[name]()
    fluxes, variances = _fit(psf)
    truth = np.array([f for _, _, f in TRUTH])

    # the bug returned exactly this, so assert against it explicitly
    assert not np.allclose(fluxes, 0.0), "PSF fell through to a zero template"
    assert np.all(np.isfinite(variances))
    assert np.allclose(fluxes, truth, rtol=0.15)


def test_psf_types_agree_with_each_other():
    """Same Gaussian expressed four ways must give the same fluxes."""
    ncirc = _fit(NCircularGaussianPSF([SIGMA], [1.0]))[0]
    mog = _fit(_mog_psf())[0]
    assert np.allclose(ncirc, mog, rtol=1e-6)


def test_psf_kind_classifies():
    assert psf_kind(_pixelized_psf())[0] == "pixelized"
    assert psf_kind(HybridPixelizedPSF(_pixelized_psf(),
                                       gauss=_mog_psf()))[0] == "pixelized"
    for psf in (_mog_psf(), NCircularGaussianPSF([SIGMA], [1.0])):
        kind, mog = psf_kind(psf)
        assert kind == "mog" and hasattr(mog, "amp")


def test_unsupported_psf_raises_instead_of_returning_zeros():
    class NotAPsf:
        pass

    with pytest.raises(TypeError, match="unsupported PSF"):
        psf_kind(NotAPsf())
