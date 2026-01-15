
import jax.numpy as jnp
import numpy as np
from tractor.engine import Tractor
from tractor.pointsource import PointSource
from tractor.brightness import Flux
from tractor.psf import PixelizedPSF
from tractor.jax.optimizer import optimize_fluxes
import dataclasses

# Mock Classes to avoid complex dependencies
class MockImage:
    def __init__(self, data, invvar, psf, wcs, sky):
        self.data = data
        self.invvar = invvar
        self.psf = psf
        self.wcs = wcs
        self.sky = sky
        self.shape = data.shape

    def getImage(self): return self.data
    def getInvError(self): return np.sqrt(self.invvar)
    def getPsf(self): return self.psf
    def getWcs(self): return self.wcs
    def getSky(self): return self.sky

class MockWCS:
    def positionToPixel(self, pos, src=None):
        return pos[0], pos[1] # Identity for simplicity
    def cdInverseAtPixel(self, x, y):
        return np.eye(2)

class MockSky:
    def __init__(self, val=0.): self.val = val
    def getConstant(self): return self.val

class MockPSF(PixelizedPSF):
    def __init__(self, img):
        self.img = img
        self.sampling = 1.0

def test_multi_band_behavior():
    # 1. Setup Data
    # Two images, 10x10.
    # Source at (5, 5).
    # Image 1: Flux ~ 100
    # Image 2: Flux ~ 200
    # Desired behavior: Should fit 100 and 200 separately.

    psf_img = np.zeros((5, 5))
    psf_img[2, 2] = 1.0
    psf = MockPSF(psf_img)

    wcs = MockWCS()
    sky = MockSky(0.)

    # Create Data
    shape = (11, 11)

    # Img 1
    data1 = np.zeros(shape)
    data1[5, 5] = 100.0
    invvar1 = np.ones(shape)
    img1 = MockImage(data1, invvar1, psf, wcs, sky)

    # Img 2
    data2 = np.zeros(shape)
    data2[5, 5] = 200.0
    invvar2 = np.ones(shape)
    img2 = MockImage(data2, invvar2, psf, wcs, sky)

    # Source
    # Pos (5, 5) corresponds to pixel (5, 5) in our Identity WCS
    src = PointSource(np.array([5., 5.]), Flux(1.0))

    tractor = Tractor([img1, img2], [src])

    print("Running optimize_fluxes...")
    try:
        results = optimize_fluxes(tractor, return_variances=True)
        print("Optimization finished.")

        # results is list of (fluxes, vars)
        if len(results) != 2:
            print(f"FAILURE: Expected 2 results, got {len(results)}")
            return

        res1 = results[0]
        res2 = results[1]

        # flux is array, we want first element
        flux1 = res1[0][0]
        flux2 = res2[0][0]

        print(f"Flux Image 1: {flux1}")
        print(f"Flux Image 2: {flux2}")

        if abs(flux1 - 100.0) < 1.0 and abs(flux2 - 200.0) < 1.0:
            print("Confirmed: Behavior is now independent multi-band fitting.")
        else:
             print(f"FAILURE: Expected 100 and 200, got {flux1} and {flux2}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_band_behavior()
