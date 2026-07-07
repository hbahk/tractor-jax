
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from tractor_jax import Tractor, Image, Catalog, PointSource, PixelizedPSF, GaussianMixturePSF, ConstantSky, Flux, NullWCS, PixPos
from tractor_jax.galaxy import ExpGalaxy, GalaxyShape, JaxGalaxy
from tractor_jax.jax.optimizer import extract_model_data, optimize_fluxes, render_image
from tractor_jax import mixture_profiles as mp

class TestUndersampledRendering(unittest.TestCase):
    def slice_batch(self, batches, idx):
        # extract_model_data returns every array leaf with a leading N_img
        # axis (matching the optimizer's vmap in_axes=0), so slice them all;
        # only scalar leaves (e.g. Background flux_idx) are shared.
        return jax.tree_util.tree_map(
            lambda x: x[idx] if getattr(x, 'ndim', 0) else x, batches)

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
                "mask": 0,
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
