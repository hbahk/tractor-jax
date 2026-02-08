
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tractor import Tractor, Image, PointSource, Catalog, NullWCS, ConstantSky
from tractor.brightness import Flux
from tractor.wcs import PixPos
from tractor.jax.optimizer import JaxOptimizer, extract_model_data, render_image
from tractor.psf import PixelizedPSF

def test_jax_optimizer_accuracy():
    # Setup
    H, W = 20, 20
    img_data = np.zeros((H, W), dtype=np.float64)
    invvar = np.ones((H, W), dtype=np.float64) / (0.1**2)

    # Create two separated sources to avoid degeneracy and isolate JAX accuracy
    x1, y1 = 4.85, 5.32
    x2, y2 = 14.97, 15.14
    true_flux1 = 1000.0
    true_flux2 = 500.0

    # Create PSF with sampling < 1.0
    yy, xx = np.mgrid[-5:6, -5:6]
    sigma = 10.0
    psf_img = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf_img /= psf_img.sum()
    # sampling=0.5
    psf = PixelizedPSF(psf_img, sampling=0.1)

    wcs = NullWCS(pixscale=1.0)
    sky = ConstantSky(0.0)

    img = Image(data=img_data, inverr=np.sqrt(invvar), psf=psf, wcs=wcs, sky=sky)
    img.name = "test_img"

    src1 = PointSource(PixPos(x1, y1), Flux(true_flux1))
    src2 = PointSource(PixPos(x2, y2), Flux(true_flux2))
    cat = Catalog(src1, src2)

    tractor = Tractor([img], cat)

    # Generate Synthetic Data using JAX to ensure consistency
    # (Since CPU generation is slightly flawed in this env)

    images_data, batches, initial_fluxes = extract_model_data(tractor, oversample_rendering=True)

    # Check PSF FFT
    psf_fft = images_data['psf']['fft']
    print("PSF FFT shape:", psf_fft.shape)
    print("PSF FFT sum:", jnp.sum(jnp.abs(psf_fft)))

    # Slice for render
    single_image_data = jax.tree_util.tree_map(lambda x: x[0], images_data)
    single_batches = {}
    if 'PointSource' in batches:
        single_batches['PointSource'] = {
            'flux_idx': batches['PointSource']['flux_idx'][0],
            'pos_pix': batches['PointSource']['pos_pix'][0],
            'mask': batches['PointSource']['mask'][0] if 'mask' in batches['PointSource'] else None
        }

    # Debug render
    fluxes = initial_fluxes[0]
    print("Fluxes:", fluxes)
    pos = single_batches['PointSource']['pos_pix']
    print("Positions:", pos)

    true_model = render_image(initial_fluxes[0], single_image_data, single_batches)
    print("Raw Model Sum:", jnp.sum(true_model))
    print("Raw Model Mean:", jnp.mean(true_model))
    print("Raw Model Max:", jnp.max(true_model))

    # Crop padding
    true_model = true_model[:H, :W]
    img.data += np.array(true_model)

    print(f"True Model Sum (Cropped): {true_model.sum():.4f}")

    if true_model.sum() == 0:
        print("FAIL: Model is zero.")
        return

    # Perturb fluxes
    src1.brightness.setParams([800.0])
    src2.brightness.setParams([600.0])

    print("Initial Fluxes:", src1.brightness.getParams(), src2.brightness.getParams())

    # Use JaxOptimizer
    tractor.optimizer = JaxOptimizer()

    # Run optimization
    dchisq = 1e-10
    for step in range(50):
        # We need variance=True to match user loop signature, though we don't use it here
        dlnp, X, alpha, var = tractor.optimize(variance=True, shared_params=False)
        # print(f"Step {step}: dlnp={dlnp}")
        if dlnp < dchisq:
            break

    f1 = tractor.catalog[0].brightness.getParams()[0]
    f2 = tractor.catalog[1].brightness.getParams()[0]

    print(f"Final Fluxes: {f1:.6f}, {f2:.6f}")
    print(f"Error: {f1 - true_flux1:.6f}, {f2 - true_flux2:.6f}")

    if abs(f1 - true_flux1) < 1e-4 and abs(f2 - true_flux2) < 1e-4:
        print("PASS: JAX Optimizer recovered fluxes accurately.")
    else:
        print("FAIL: JAX Optimizer failed to recover fluxes.")

if __name__ == "__main__":
    test_jax_optimizer_accuracy()
