import jax
import jax.numpy as jnp
import numpy as np
import unittest
from tractor_jax import Tractor, Image, Catalog
from tractor_jax.basics import NullWCS, ConstantSky
from tractor_jax.pointsource import PointSource
from tractor_jax.brightness import Flux
from tractor_jax.wcs import RaDecPos
from tractor_jax.psf import PixelizedPSF
from tractor_jax.jax.optimizer import optimize_fluxes, JaxOptimizer
import tractor_jax.jax.optimizer as jax_opt

class TestJaxBucketing(unittest.TestCase):
    def setUp(self):
        # Create a set of images with different sizes
        self.images = []
        self.catalog = Catalog()

        # Define 3 images with distinct sizes
        sizes = [(64, 64), (100, 100), (32, 128)]

        # Simple PSF
        psf_img = np.zeros((11, 11))
        psf_img[5, 5] = 1.0
        self.psf = PixelizedPSF(psf_img)

        for i, (h, w) in enumerate(sizes):
            img_data = np.zeros((h, w))
            invvar = np.ones((h, w))
            wcs = NullWCS(pixscale=1.0) # Simple WCS

            # Create Image object
            img = Image(data=img_data, inverr=np.sqrt(invvar),
                       psf=self.psf, wcs=wcs, sky=ConstantSky(0.0))
            self.images.append(img)

        # Add a source that overlaps all (roughly)
        # Position centered in the first image (32, 32)
        # In NullWCS(pixscale=1), (32, 32) is pixel coordinates.
        # Let's put a source at (32, 32) which is valid for all.
        # NullWCS(pixscale=1) maps (0,0) pix to (0,0) world if dx=dy=0.
        # But wait, NullWCS implementation:
        # pos.x + dx, pos.y + dy
        # It expects 'pos' to have .x, .y if it's PixPos, or .ra, .dec if RaDecPos?
        # Actually NullWCS.positionToPixel implementation:
        # return pos.x + self.dx, pos.y + self.dy
        # It assumes pos has .x and .y!
        # So I should use PixPos, not RaDecPos for NullWCS.

        from tractor_jax.basics import PixPos
        src = PointSource(PixPos(32.0, 32.0), Flux(100.0))

        self.catalog.append(src)
        self.tractor = Tractor(self.images, self.catalog)

    def test_default_behavior(self):
        # Should run without error using default padding (max of all)
        # Currently max_H=100, max_W=128 -> Pad to (100+pad, 128+pad)
        results = optimize_fluxes(self.tractor, oversample_rendering=False)
        self.assertEqual(len(results), 3)
        # Check result shape (1 flux param)
        self.assertEqual(len(results[0]), 1)

    def test_bucketing_interface(self):
        # Test independent bucketing
        # Expected:
        # Img 3 (32, 128) -> (43, 139) -> Quantized (64, 160)
        # If independent, should be (64, 160)

        results = optimize_fluxes(
            self.tractor,
            bucket_shape_mode="independent",
            bucket_mode="auto",
            bucket_base=32
        )
        self.assertEqual(len(results), 3)

        # Test fixed buckets
        # Force everything into 256x256
        results_fixed = optimize_fluxes(
            self.tractor,
            bucket_mode="fixed",
            bucket_sizes=[256]
        )
        self.assertEqual(len(results_fixed), 3)

        # Test JaxOptimizer interface pass-through
        opt = JaxOptimizer()
        dlnp, X, alpha = opt.optimize(
            self.tractor,
            bucket_mode="fixed",
            bucket_sizes=[256]
        )
        # Just check it ran
        self.assertTrue(len(X) > 0)

if __name__ == "__main__":
    unittest.main()
