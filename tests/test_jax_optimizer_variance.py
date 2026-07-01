
import numpy as np
import jax.numpy as jnp
from tractor_jax.jax.optimizer import optimize_fluxes
from tractor_jax.pointsource import PointSource
from tractor_jax.brightness import Flux
from tractor_jax.psf import GaussianMixturePSF
from tractor_jax.mixture_profiles import MixtureOfGaussians

# Mock classes
class MockImage:
    def __init__(self, data, invvar, psf, wcs):
        self.data = data
        self.invvar = invvar
        self.psf = psf
        self.wcs = wcs
        self.shape = data.shape

    def getImage(self): return self.data
    def getInvError(self): return np.sqrt(self.invvar)
    def getPsf(self): return self.psf
    def getWcs(self): return self.wcs

class MockWCS:
    def positionToPixel(self, pos, src=None):
        return pos[0], pos[1]
    def cdInverseAtPixel(self, x, y):
        return np.eye(2)

class MockTractor:
    def __init__(self, images, catalog):
        self.images = images
        self.catalog = catalog

def test_jax_optimizer_variance_integration():
    print("Testing integration of variance calculation...")

    # Setup data
    H, W = 20, 20
    data = np.zeros((H, W))
    invvar = np.ones((H, W))

    # PSF: Simple MoG
    amp = np.array([1.0])
    mean = np.array([[0.0, 0.0]])
    var = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    # Pass arrays directly, not MoG object
    psf = GaussianMixturePSF(amp, mean, var)

    # WCS
    wcs = MockWCS()

    img = MockImage(data, invvar, psf, wcs)

    # Source
    src = PointSource(
        None,
        Flux(100.0)
    )
    # Monkey patch getPosition/getSourceType
    src.getPosition = lambda: np.array([10.0, 10.0])
    # Flux object
    src.brightness = Flux(100.0)

    catalog = [src]
    tractor = MockTractor([img], catalog)

    # Run optimization with variance
    try:
        results = optimize_fluxes(tractor, return_variances=True)
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        raise e

    assert isinstance(results, list)
    assert len(results) == 1

    fluxes, variances = results[0]

    print("Optimized flux:", fluxes)
    print("Variances:", variances)

    # Check return type
    assert isinstance(variances, np.ndarray)
    assert len(variances) == 1

    # Basic check: Variance should be positive
    assert variances[0] > 0.0

    # Check non-variance call still works
    results_novar = optimize_fluxes(tractor, return_variances=False)
    assert isinstance(results_novar, list)
    assert isinstance(results_novar[0], np.ndarray)

    print("Integration test passed!")

if __name__ == "__main__":
    test_jax_optimizer_variance_integration()
