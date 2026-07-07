"""Engine tiling mode (optimize_fluxes(use_tiling=True)).

Contract: tiled results come back in the same per-image, catalog-layout
format as the untiled path — each source read from the tile whose core box
owns it — and must match the untiled solve.
"""

import numpy as np

from tractor_jax import (
    Tractor, Image, PointSource, PixPos, Flux, ConstantSky, NullWCS,
    GaussianMixturePSF,
)
from tractor_jax.jax.optimizer import optimize_fluxes


def _make_scene(fit_background=False, bg_level=0.0, seed=0):
    """A 60x60 image with point sources spread across tile boundaries."""
    H = W = 60
    sigma = 1.5
    psf = GaussianMixturePSF(
        np.array([1.0]), np.zeros((1, 2)),
        np.array([[[sigma**2, 0.0], [0.0, sigma**2]]]),
    )

    true_fluxes = [900.0, 700.0, 500.0, 800.0, 600.0]
    positions = [(10.0, 10.0), (30.0, 30.0), (50.0, 12.0),
                 (19.5, 40.0),   # near a tile-core boundary for tile_size=20
                 (41.0, 52.0)]

    catalog = [PointSource(PixPos(x, y), Flux(f))
               for (x, y), f in zip(positions, true_fluxes)]

    img = Image(data=np.zeros((H, W), dtype=np.float32),
                inverr=np.ones((H, W), dtype=np.float32),
                psf=psf, wcs=NullWCS(), sky=ConstantSky(0.0))
    tractor = Tractor([img], catalog)

    rng = np.random.default_rng(seed)
    data = np.asarray(tractor.getModelImage(0)) + bg_level
    data = data + rng.normal(0.0, 1e-3, (H, W))
    tractor.images[0].data = data.astype(np.float32)
    if fit_background:
        # start the background estimate away from the truth
        tractor.images[0].sky = ConstantSky(0.0)
    return tractor, np.array(true_fluxes)


def _fluxes(result):
    return np.asarray(result[:len(result)])


def test_tiled_matches_untiled():
    tractor, truth = _make_scene()
    untiled = optimize_fluxes(tractor, use_sharding=False)[0]

    tractor2, _ = _make_scene()
    tiled = optimize_fluxes(tractor2, use_tiling=True, tile_size=20,
                            use_sharding=False)[0]

    assert tiled.shape == untiled.shape
    np.testing.assert_allclose(tiled, untiled, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(tiled, truth, rtol=1e-2)


def test_tiled_variances_and_update_catalog():
    tractor, truth = _make_scene()
    res = optimize_fluxes(tractor, use_tiling=True, tile_size=20,
                          return_variances=True, update_catalog=True,
                          use_sharding=False)
    fluxes, variances = res[0]
    assert fluxes.shape == variances.shape == truth.shape
    assert np.all(variances[:len(truth)] > 0)
    np.testing.assert_allclose(fluxes, truth, rtol=1e-2)

    # update_catalog wrote the merged fluxes back onto the sources
    cat_fluxes = np.array([src.brightness.getValue()
                           for src in tractor.catalog])
    np.testing.assert_allclose(cat_fluxes, fluxes[:len(truth)], rtol=1e-6)


def test_tiled_background():
    bg = 5.0
    tractor, truth = _make_scene(fit_background=True, bg_level=bg)
    res = optimize_fluxes(tractor, use_tiling=True, tile_size=20,
                          fit_background=True, use_sharding=False)
    merged = res[0]
    assert merged.shape == (len(truth) + 1,)
    np.testing.assert_allclose(merged[:-1], truth, rtol=1e-2)
    # merged background = core-area weighted mean of per-tile backgrounds
    np.testing.assert_allclose(merged[-1], bg, rtol=1e-2)
