import jax
import jax.numpy as jnp
import jax.scipy.optimize
from tractor.engine import Tractor, Catalog, Images
from tractor.batch_mixture_profiles import BatchMixtureOfGaussians
from tractor.batch_psf import BatchPixelizedPSF
from tractor.batch_galaxy import getShearedProfileGPU

def render_model(params, image_data_shape, source_configs):
    """
    Render a model image from parameters using JAX.

    Args:
        params: Flat JAX array of parameters.
        image_data_shape: Shape of the image (H, W).
        source_configs: List of configuration dictionaries for each source.
                        Each dict should contain keys: 'type', 'psf_sigma', etc.

    Returns:
        JAX array of shape image_data_shape.
    """
    model_image = jnp.zeros(image_data_shape)
    H, W = image_data_shape

    # We iterate over sources. In a real JAX application, we would vectorise this.
    # For now, we assume params is structured as [src1_params, src2_params, ...].
    # And we loop (Python loop) over sources, adding them to the image.

    # This assumes we know the number of params per source.
    # Let's assume for this simple implementation:
    # PointSource: [flux, x, y]

    param_idx = 0
    for config in source_configs:
        if config['type'] == 'PointSource':
            flux = params[param_idx]
            x = params[param_idx + 1]
            y = params[param_idx + 2]
            param_idx += 3

            # Simple Gaussian PSF rendering
            # In real tractor, we use the PSF object. Here we simulate it with a Gaussian.
            sigma = config.get('psf_sigma', 1.0)

            # Create a grid
            xx, yy = jnp.meshgrid(jnp.arange(W), jnp.arange(H))

            # Gaussian
            g = flux * jnp.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)) / (2 * jnp.pi * sigma**2)

            model_image = model_image + g

    return model_image

def optimize_batch(images, catalog):
    """
    Simultaneously fit adjacent sources using JAX.

    Args:
        images: list of Image objects (must contain data and invvar)
        catalog: list of Source objects

    Returns:
        Optimized parameters
    """

    if isinstance(catalog, list):
        catalog = Catalog(*catalog)
    if isinstance(images, list):
        images = Images(*images)

    tractor = Tractor(images, catalog)

    # Extract initial parameters and configuration
    p0_list = []
    source_configs = []

    # We only support PointSource for this demonstration
    # Note: Params are global for sources, but fluxes might be band-dependent if we had MultiParams source.
    # The standard PointSource(pos, brightness) assumes brightness is a Flux object (single value) or similar.
    # If we have multiple images, usually we want to fit flux per image or a spectrum.
    # Here we assume the source brightness is a single parameter valid for all images (e.g. same filter) OR
    # we are fitting positions only and flux is shared (which is weird).
    #
    # Actually, standard Tractor PointSource has brightness.
    # If we have multiple images in different bands, we usually use a Source that has SED or multiple fluxes.
    # Here, for simplicity, we assume the catalog sources have parameters that we want to optimize.
    # If the images are different bands, the flux parameter in PointSource is usually interpreted as flux in some reference band or we need a different source model.
    #
    # For this demo, we assume a single flux parameter.

    for src in catalog:
        if src.getSourceType() == 'PointSource':
            pos = src.pos.getParams()
            br = src.brightness.getParams()

            # Tractor order is [x, y, flux] usually.
            # We will use [flux, x, y] for our simple render model
            p0_list.extend([br[0], pos[0], pos[1]])

            # Get PSF sigma from image (assuming first image's PSF is GaussianMixturePSF)
            # We assume PSF is roughly same for all images or we take average/first.
            # Ideally we should pass list of PSFs.
            psf = images[0].getPsf()
            sigma = 1.0
            if hasattr(psf, 'mog'):
                 # Approximate sigma from first component variance
                 sigma = jnp.sqrt(psf.mog.var[0,0,0])

            source_configs.append({'type': 'PointSource', 'psf_sigma': sigma})
        else:
            raise NotImplementedError(f"Source type {src.getSourceType()} not supported in this JAX demo")

    p0 = jnp.array(p0_list)

    # Prepare data arrays
    data_list = []
    invvar_list = []

    for img in images:
        data_list.append(jnp.array(img.getImage()))
        invvar_list.append(jnp.array(img.getInvError())**2)

    # Define loss function
    def loss_fn(params):
        chi2_total = 0.0
        # Iterate over images
        for i, data in enumerate(data_list):
            invvar = invvar_list[i]
            # In a real scenario, we might have different PSF/WCS per image.
            # Render model for this image
            # Note: render_model uses source_configs which has fixed psf_sigma.
            # This is a simplification.
            model = render_model(params, data.shape, source_configs)
            chi2 = jnp.sum((data - model)**2 * invvar)
            chi2_total += chi2
        return chi2_total

    # Optimize
    # We use BFGS
    res = jax.scipy.optimize.minimize(loss_fn, p0, method='BFGS')

    return res
