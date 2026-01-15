
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from tractor import Tractor, Image, PointSource, NullWCS, ConstantSky
from tractor.brightness import Flux
from tractor.wcs import PixPos
from tractor.psf import PixelizedPSF
from tractor.jax.optimizer import extract_model_data, render_image

class TestJaxShift(unittest.TestCase):
    def test_shift_precision(self):
        print("\nTesting JAX rendering shift precision...")

        # 1. Setup simple image
        H, W = 11, 11
        # Sampling 0.1 means 1 img pixel = 10 psf pixels.
        psf_h, psf_w = 21, 21
        y, x = np.indices((psf_h, psf_w))
        # Use a wide sigma to minimize aliasing/undersampling errors in centroid measurement
        # Sigma = 5.0 high-res pixels -> 0.5 low-res pixels.
        # FWHM = 2.355 * 0.5 = 1.17 pixels. Nyquist sampled.
        sigma = 5.0
        psf_img = np.exp(-((x - 10)**2 + (y - 10)**2) / (2 * sigma**2))
        psf_img /= psf_img.sum() # Normalize

        psf = PixelizedPSF(psf_img, sampling=0.1)

        # 2. Source at various positions
        src_pos = PixPos(5.0, 5.0)
        src_flux = Flux(100.0)
        src = PointSource(src_pos, src_flux)

        # 3. Tractor Image
        tim = Image(
            data=np.zeros((H, W)),
            inverr=np.ones((H, W)),
            psf=psf,
            wcs=NullWCS(),
            sky=ConstantSky(0.0),
        )

        tractor = Tractor([tim], [src])

        offsets = [0.0, 0.25, 0.5, 0.75]
        max_error = 0.0

        for off in offsets:
            p = 5.0 + off
            src.pos = PixPos(p, 5.0)

            # Extract and Render
            images_data, batches, initial_fluxes = extract_model_data(
                tractor, oversample_rendering=True
            )

            # Slice for first image
            s_img_data = {
                "data": images_data["data"][0],
                "invvar": images_data["invvar"][0],
                "psf": {
                    "type_code": images_data["psf"]["type_code"][0],
                    "sampling": images_data["psf"]["sampling"][0],
                    "fft": images_data["psf"]["fft"][0],
                    "amp": images_data["psf"]["amp"][0],
                    "mean": images_data["psf"]["mean"][0],
                    "var": images_data["psf"]["var"][0],
                }
            }
            s_batches = {}
            if "PointSource" in batches:
                s_batches["PointSource"] = {
                    "flux_idx": batches["PointSource"]["flux_idx"],
                    "pos_pix": batches["PointSource"]["pos_pix"][0],
                }

            fluxes = initial_fluxes[0]
            model = render_image(fluxes, s_img_data, s_batches)

            # Measure centroid
            y, x = np.indices(model.shape)
            sum_m = np.sum(model)
            cx = np.sum(x * model) / sum_m

            error = cx - p
            print(f"Pos: {p:.2f}, Centroid X: {cx:.4f}, Error: {error:.4f}")

            # With sigma=0.5, error should be small (< 0.01)
            self.assertTrue(abs(error) < 0.01, f"Centroid error too large: {error} at pos {p}")
            max_error = max(max_error, abs(error))

            # Verify Flux Conservation
            self.assertTrue(np.isclose(sum_m, 100.0, rtol=1e-3), f"Flux not conserved: {sum_m}")

        print(f"Max centroid error: {max_error:.5f}")

if __name__ == "__main__":
    unittest.main()
