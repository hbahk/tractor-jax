
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from tractor_jax import Tractor, Image, Catalog, PointSource, PixelizedPSF, GaussianMixturePSF, ConstantSky, Flux, NullWCS, PixPos
from tractor_jax.galaxy import ExpGalaxy, GalaxyShape, JaxGalaxy
from tractor_jax.jax.optimizer import extract_model_data, optimize_fluxes, render_image
from tractor_jax import mixture_profiles as mp

class TestUndersampledRendering(unittest.TestCase):
    def test_anisotropic_undersampled(self):
        print("\n--- Test Undersampled Anisotropic Rendering ---")

        # 1. Setup Scene
        # Small image, coarse pixels
        H, W = 20, 20
        data = np.zeros((H, W))
        invvar = np.ones((H, W))
        inverr = np.sqrt(invvar)

        # 2. Create Anisotropic Gaussian PSF
        # Define Gaussian parameters
        # Sigma = [0.3, 0.1] pixels (very small, anisotropic)
        # In variance: sigma^2
        sigma_x = 0.3
        sigma_y = 0.1
        var_x = sigma_x**2
        var_y = sigma_y**2

        # Create GaussianMixturePSF (Analytic)
        amp = np.array([1.0])
        mean = np.zeros((1, 2))
        var = np.array([[[var_x, 0.0], [0.0, var_y]]])

        psf_mog = GaussianMixturePSF(amp, mean, var)

        # Create PixelizedPSF (Oversampled)
        # Sampling = 0.2 (5x oversampling)
        sampling = 0.2

        # Render PSF onto fine grid
        # Fine pixel size = 0.2 image pixels
        # Grid size: enough to cover PSF
        # 5 sigma ~ 1.5 pixels.
        # Fine grid size ~ 1.5 / 0.2 ~ 7.5 pixels radius.
        # Let's make it 21x21 fine pixels.
        ph_fine, pw_fine = 21, 21

        # Evaluate MoG on fine grid
        # Grid coordinates in image pixels
        # Center is at (0, 0) relative to PSF center

        # Pixel coordinates on fine grid
        # Center index
        cx, cy = pw_fine // 2, ph_fine // 2

        # Generate grid
        # y, x indices
        y_idx, x_idx = np.indices((ph_fine, pw_fine))

        # Convert to offsets in image pixels
        # pixel (i, j) corresponds to offset ( (j - cx)*sampling, (i - cy)*sampling )
        y_off = (y_idx - cy) * sampling
        x_off = (x_idx - cx) * sampling

        # Evaluate Gaussian
        # 1 / (2pi sqrt(det)) * exp(-0.5 * r^T S^-1 r)
        det = var_x * var_y
        inv_var_x = 1.0 / var_x
        inv_var_y = 1.0 / var_y

        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        exponent = -0.5 * (x_off**2 * inv_var_x + y_off**2 * inv_var_y)
        psf_fine_img = norm * np.exp(exponent)

        # Normalize sum
        psf_fine_img /= np.sum(psf_fine_img)

        psf_pix = PixelizedPSF(psf_fine_img)
        psf_pix.sampling = sampling

        # 3. Create Sheared Galaxy
        # Pos: Center of image
        pos_x, pos_y = 10.0, 10.0
        flux_val = 1000.0

        # Shape: re=0.5 pixels (small), ab=0.5 (elliptical), phi=45 (rotated)
        # Note: re is effective radius. For Exp, half-light radius.
        # GalaxyShape params order: re, ab, phi
        shape = GalaxyShape(0.5, 0.5, 45.0)

        # Source 1: Using MoG PSF (Analytic)
        src1 = ExpGalaxy(PixPos(pos_x, pos_y), Flux(flux_val), shape)
        img1 = Image(data=data, inverr=inverr, psf=psf_mog, wcs=NullWCS(), sky=ConstantSky(0.0))
        tractor1 = Tractor([img1], [src1])

        # Source 2: Using Pixelized PSF (FFT, Oversampled)
        src2 = ExpGalaxy(PixPos(pos_x, pos_y), Flux(flux_val), shape)
        img2 = Image(data=data, inverr=inverr, psf=psf_pix, wcs=NullWCS(), sky=ConstantSky(0.0))
        tractor2 = Tractor([img2], [src2])

        # 4. Render
        print("Rendering Analytic (MoG)...")
        # For MoG, extract_model_data will create MoG batches and render using render_galaxy_mog
        images_data1, batches1, fluxes1 = extract_model_data(tractor1, oversample_rendering=False)
        # Need to slice batches for render_image
        batch1_single = self.slice_batch(batches1, 0)
        model1 = render_image(fluxes1[0], jax.tree_util.tree_map(lambda x: x[0], images_data1), batch1_single)

        print("Rendering FFT (Pixelized, Oversampled)...")
        # For Pixelized, extract_model_data will create FFT batches.
        # sampling=0.2 means max_factor=5.
        # It should use high-res grid.
        images_data2, batches2, fluxes2 = extract_model_data(tractor2, oversample_rendering=True)
        batch2_single = self.slice_batch(batches2, 0)
        model2 = render_image(fluxes2[0], jax.tree_util.tree_map(lambda x: x[0], images_data2), batch2_single)

        # 5. Compare
        # Compare images
        print(f"Model1 shape: {model1.shape}")
        print(f"Model2 shape: {model2.shape}")

        # If model2 is padded (due to FFT padding), crop it
        if model2.shape != model1.shape:
            H, W = model1.shape
            # The padding is added to the end (right/bottom)
            model2 = model2[:H, :W]

        sum1 = np.sum(model1)
        sum2 = np.sum(model2)

        print(f"Flux Analytic: {sum1}")
        print(f"Flux FFT: {sum2}")
        print(f"Expected Flux: {flux_val}")

        # Visual check (print small patch)
        cx, cy = int(pos_x), int(pos_y)
        sl = (slice(cy-2, cy+3), slice(cx-2, cx+3))
        print("\nAnalytic Patch:")
        print(model1[sl])
        print("\nFFT Patch:")
        print(model2[sl])

        # Validation
        # The FFT model should conserve flux significantly better than the point-sampled analytic model
        # for this undersampled case.

        # Check FFT flux conservation
        # Allow 1% error
        rel_err_fft = abs(sum2 - flux_val) / flux_val
        print(f"FFT Relative Flux Error: {rel_err_fft}")
        self.assertTrue(rel_err_fft < 0.01, f"FFT Flux not conserved: {sum2} vs {flux_val}")

        # Check Peak Position
        # Should be at (10, 10)
        peak_idx = np.unravel_index(np.argmax(model2), model2.shape)
        print(f"Peak Location: {peak_idx}")
        self.assertEqual(peak_idx, (10, 10), "Peak location incorrect")

    def slice_batch(self, batches, idx):
        single = {}
        for k, v in batches.items():
            single[k] = {}
            for subk, subv in v.items():
                if subk in ['pos_pix', 'wcs_cd_inv']:
                    single[k][subk] = subv[idx]
                else:
                    single[k][subk] = subv # shared
        return single

    def test_batch_rendering(self):
        print("\n--- Test Batch Undersampled Rendering ---")

        # Parameters
        N_batch = 5
        H, W = 30, 30
        sampling = 0.5 # Undersampled (PSF is higher res)

        # We need uniform shapes for batching
        # Global Max PSF size (in high-res pixels)
        max_psf_size = 21

        # Setup Data
        tractors = []
        expected_fluxes = []

        for i in range(N_batch):
            # 1. Create Data
            data = np.zeros((H, W))
            invvar = np.ones((H, W))
            inverr = np.sqrt(invvar)

            # 2. VARY PSF Size and Shape
            psf_dim = max_psf_size

            # Sigma varies
            sigma = 1.0 + i * 0.5 # 1.0, 1.5, ... (High res pixels)

            # Gaussian
            y, x = np.indices((psf_dim, psf_dim))
            cy, cx = psf_dim // 2, psf_dim // 2
            r2 = (x - cx)**2 + (y - cy)**2
            psf_val = np.exp(-0.5 * r2 / sigma**2)
            psf_val /= np.sum(psf_val)

            psf_img = psf_val

            psf = PixelizedPSF(psf_img)
            psf.sampling = sampling # 0.5

            # 3. Create Source
            # Vary positions
            if i == 0:
                pos = [15., 15.]
            elif i == 1:
                pos = [2., 15.] # Edge
            elif i == 2:
                pos = [28., 15.] # Edge
            elif i == 3:
                pos = [15., 2.] # Edge
            else:
                pos = [15., 28.] # Edge

            flux_val = 1000.0 + i * 100.0
            expected_fluxes.append(flux_val)

            src = PointSource(PixPos(pos[0], pos[1]), Flux(flux_val))

            img = Image(data=data, inverr=inverr, psf=psf, wcs=NullWCS(), sky=ConstantSky(0.0))
            tractors.append(Tractor([img], [src]))

        # 4. Extract & Stack
        images_data_list = []
        batches_list = []
        fluxes_list = []

        print("Extracting data...")
        for trac in tractors:
            # oversample_rendering=True triggers the resizing/padding logic
            img_data, batch, flux = extract_model_data(trac, oversample_rendering=True)
            images_data_list.append(img_data)
            batches_list.append(batch)
            fluxes_list.append(flux)

        # Stack
        def stack_leaves(leaves):
            return jnp.stack(leaves)

        print("Stacking...")
        images_data_batched = jax.tree_util.tree_map(lambda *x: stack_leaves(x), *images_data_list)
        batches_batched = jax.tree_util.tree_map(lambda *x: stack_leaves(x), *batches_list)
        fluxes_batched = jnp.stack(fluxes_list)

        # 5. Render Batch
        print("Rendering Batch...")

        sample_batches = batches_list[0]
        batches_in_axes = {}
        if "PointSource" in sample_batches:
            batches_in_axes["PointSource"] = {
                "flux_idx": 0,
                "pos_pix": 0,
            }

        # Wrapper to slice N_img=0
        def render_wrapper(fluxes, img_data, batch):
            # flux: (N_img=1, N_param) -> (N_param)
            f = fluxes[0]

            # img_data: e.g. data is (N_img=1, H, W) -> slice 0
            single_img_data = jax.tree_util.tree_map(lambda x: x[0], img_data)

            # batch: e.g. pos_pix is (N_img=1, N_src, 2) -> slice 0
            single_batch = {}
            for k, v in batch.items():
                single_batch[k] = {}
                for sk, sv in v.items():
                    if sk in ['pos_pix', 'wcs_cd_inv']:
                        single_batch[k][sk] = sv[0]
                    else:
                        single_batch[k][sk] = sv # shared

            return render_image(f, single_img_data, single_batch)

        # Vmap over Batch axis (0)
        render_batch_fn = jax.vmap(render_wrapper, in_axes=(0, 0, batches_in_axes))

        models = render_batch_fn(fluxes_batched, images_data_batched, batches_batched)

        print(f"Models shape: {models.shape}")

        # 6. Verify
        print(f"Model shape (padded): {models.shape[1:]}")

        for i in range(N_batch):
            model = models[i]
            total_flux = jnp.sum(model)
            expected = expected_fluxes[i]
            rel_err = abs(total_flux - expected) / expected

            print(f"Batch {i}: Pos={tractors[i].catalog[0].pos}, Flux={total_flux:.2f}, Exp={expected:.2f}, Err={rel_err:.4f}")

            self.assertTrue(rel_err < 0.01, f"Flux not conserved in batch {i}")

            # Check edge artifacts
            # e.g. i=1 pos=[2, 15] (Left). Right edge of VALID IMAGE should be empty.
            if i == 1:
                # Valid image is 30x30.
                valid_w = W
                # Check right edge of valid image. e.g. index 25-30.
                valid_right_edge = model[:, valid_w-5 : valid_w]

                print(f"Valid Right edge max: {jnp.max(valid_right_edge)}")
                self.assertTrue(jnp.max(valid_right_edge) < 1e-3, "Wrap around artifact detected in valid region")

if __name__ == '__main__':
    unittest.main()
