import unittest
import jax
import jax.numpy as jnp
import numpy as np
from tractor_jax import Tractor, Image, PointSource, NullWCS, ConstantSky
from tractor_jax.brightness import Flux
from tractor_jax.wcs import PixPos
from tractor_jax.psf import PixelizedPSF
from tractor_jax.jax.optimizer import optimize_fluxes

class TestJaxSharding(unittest.TestCase):
    def test_sharding_execution(self):
        """
        Verifies that optimize_fluxes runs with use_sharding=True.
        """
        print("\nTesting JAX optimization with sharding...")

        # 1. Setup Scene with Multiple Images
        N_img = 4 # Enough to shard if we had devices, or just batch on 1 device
        H, W = 20, 20

        # Simple Gaussian PSF
        psf_h, psf_w = 11, 11
        y, x = np.indices((psf_h, psf_w))
        sigma = 2.0
        psf_img = np.exp(-((x - 5)**2 + (y - 5)**2) / (2 * sigma**2))
        psf_img /= psf_img.sum()
        psf = PixelizedPSF(psf_img, sampling=1.0) # Simple sampling

        # Source
        src_pos = PixPos(10.0, 10.0)
        true_flux = 1000.0
        src_flux = Flux(true_flux)
        src = PointSource(src_pos, src_flux)

        # Create Images with Noise
        images = []
        for i in range(N_img):
            # Simulate image
            data = np.zeros((H, W))

            # Put source at (10, 10)
            # Use rendering.render_point_source_pixelized logic roughly
            cy, cx = 10, 10
            y0 = cy - 5
            x0 = cx - 5

            # Clip
            h_slice = slice(max(0, y0), min(H, y0 + psf_h))
            w_slice = slice(max(0, x0), min(W, x0 + psf_w))

            # Psf slice
            py0 = max(0, -y0)
            px0 = max(0, -x0)
            py1 = py0 + (h_slice.stop - h_slice.start)
            px1 = px0 + (w_slice.stop - w_slice.start)

            data[h_slice, w_slice] += true_flux * psf_img[py0:py1, px0:px1]

            # Add noise
            noise = np.random.normal(0, 1.0, (H, W))
            data += noise

            tim = Image(
                data=data,
                inverr=np.ones((H, W)), # Sigma=1
                psf=psf,
                wcs=NullWCS(),
                sky=ConstantSky(0.0),
            )
            images.append(tim)

        tractor = Tractor(images, [src])

        # Perturb initial flux
        src.brightness.setParams([500.0]) # Start far from 1000

        # 2. Run Optimization with Sharding
        # This calls prepare_sharded_inputs internally
        # We assume use_sharding=True is default or passed explicitly.
        # Passing explicitly to test the arg.
        results = optimize_fluxes(
            tractor,
            oversample_rendering=False,
            return_variances=True,
            fit_background=False,
            update_catalog=True,
            vmap_images=True,
            use_sharding=True
        )

        # 3. Verify Results
        print("Optimization Results:")
        for i, (flux, var) in enumerate(results):
            print(f"Image {i}: Flux={flux[0]:.2f}, Var={var[0]:.2f}")
            # Flux should be close to 1000. Variance should be roughly 1/sum(psf^2) ?
            # sum(psf^2) is roughly 1/(4*pi*sigma^2) approx 1/50.
            # var approx 50?
            self.assertTrue(abs(flux[0] - true_flux) < 50.0, f"Flux failed to converge: {flux[0]}")

        # Check if catalog was updated (Should NOT be for multiple images)
        curr_flux = src.brightness.getParams()[0]
        print(f"Catalog Flux (should be unchanged): {curr_flux}")
        self.assertEqual(curr_flux, 500.0)

        # 4. Run Optimization with Single Image to verify update
        tractor1 = Tractor([images[0]], [src])
        src.brightness.setParams([500.0])
        optimize_fluxes(
            tractor1,
            update_catalog=True,
            vmap_images=True,
            use_sharding=True
        )
        curr_flux_1 = src.brightness.getParams()[0]
        print(f"Catalog Flux (should be updated): {curr_flux_1:.2f}")
        self.assertTrue(abs(curr_flux_1 - true_flux) < 50.0)

if __name__ == "__main__":
    unittest.main()
