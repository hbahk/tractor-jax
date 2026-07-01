import unittest
import numpy as np
import jax.numpy as jnp
from tractor_jax.jax_optimize import optimize_batch, render_model
from tractor_jax.image import Image
from tractor_jax.pointsource import PointSource
from tractor_jax.wcs import PixPos
from tractor_jax.brightness import Flux
from tractor_jax.psf import GaussianMixturePSF

class TestJaxOptimize(unittest.TestCase):
    def test_optimization_two_sources(self):
        # Create a synthetic image
        H, W = 40, 40
        sigma = 2.0

        # Two overlapping sources
        src1_x, src1_y, src1_flux = 15.0, 15.0, 1000.0
        src2_x, src2_y, src2_flux = 18.0, 18.0, 800.0 # 3 pixels away, roughly 1.5 sigma

        # Create synthetic data
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        def gaussian(x, y, flux):
            return flux * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

        model = gaussian(src1_x, src1_y, src1_flux) + gaussian(src2_x, src2_y, src2_flux)
        noise = np.random.normal(0, 1, size=(H, W))
        data = model + noise
        invvar = np.ones_like(data) # weight

        # Create Tractor objects
        psf = GaussianMixturePSF([1.0], [[0.0, 0.0]], [[[sigma**2, 0.0], [0.0, sigma**2]]])
        img = Image(data=data, inverr=np.sqrt(invvar), psf=psf)

        # Initial guess (slightly offset)
        src1 = PointSource(PixPos(14.0, 16.0), Flux(900.0))
        src2 = PointSource(PixPos(19.0, 17.0), Flux(700.0))

        # Optimize
        res = optimize_batch([img], [src1, src2])

        print("Optimization result:", res)

        # Extract results
        # optimize_batch packs parameters as [flux1, x1, y1, flux2, x2, y2]
        # Order depends on catalog order.

        est1_flux = res.x[0]
        est1_x = res.x[1]
        est1_y = res.x[2]

        est2_flux = res.x[3]
        est2_x = res.x[4]
        est2_y = res.x[5]

        print(f"Src1 True: flux={src1_flux}, x={src1_x}, y={src1_y}")
        print(f"Src1 Est:  flux={est1_flux}, x={est1_x}, y={est1_y}")

        print(f"Src2 True: flux={src2_flux}, x={src2_x}, y={src2_y}")
        print(f"Src2 Est:  flux={est2_flux}, x={est2_x}, y={est2_y}")

        self.assertTrue(np.abs(est1_x - src1_x) < 0.5)
        self.assertTrue(np.abs(est1_y - src1_y) < 0.5)
        self.assertTrue(np.abs(est1_flux - src1_flux) < 50.0)

        self.assertTrue(np.abs(est2_x - src2_x) < 0.5)
        self.assertTrue(np.abs(est2_y - src2_y) < 0.5)
        self.assertTrue(np.abs(est2_flux - src2_flux) < 50.0)

if __name__ == '__main__':
    unittest.main()
