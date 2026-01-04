import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit, value_and_grad, vmap
import numpy as np

from tractor.engine import Tractor
from tractor.pointsource import PointSource
from tractor.galaxy import (
    Galaxy,
    ExpGalaxy,
    DevGalaxy,
    CompositeGalaxy,
    FixedCompositeGalaxy,
)
from tractor.psf import PixelizedPSF, GaussianMixturePSF
from tractor.jax.rendering import (
    render_pixelized_psf,
    render_galaxy_fft,
    render_point_source_pixelized,
    render_galaxy_mog,
    render_point_source_mog,
    render_point_source_fft,
)


def extract_model_data(tractor_obj):
    """
    Extracts all necessary data from a Tractor object for JAX optimization,
    grouping sources into batches (PointSource, Galaxy) for vectorized rendering.

    Returns:
        images_data: list of dicts (data, invvar, psf_data, shape, ...)
        batches: dict containing batched source data:
            {
                'PointSource': {
                    'flux_idx': (N_ps,),
                    'pos_pix': (N_ps, N_img, 2),
                },
                'Galaxy': {
                    'flux_idx': (N_gal,),
                    'pos_pix': (N_gal, N_img, 2),
                    'wcs_cd_inv': (N_gal, N_img, 2, 2),
                    'shapes': (N_gal, 3),
                    'profile': { 'amp': ..., 'mean': ..., 'var': ... }
                }
            }
        initial_fluxes: JAX array of all fluxes.
    """
    images = tractor_obj.images
    catalog = tractor_obj.catalog

    # 1. Extract Image Data & Precompute PSF FFTs
    images_data = []
    for img in images:
        h, w = img.shape
        data = jnp.array(img.getImage())
        invvar = jnp.array(img.getInvError()) ** 2

        # PSF Extraction
        psf = img.getPsf()
        psf_data = {}
        if isinstance(psf, PixelizedPSF):
            psf_data["type"] = "pixelized"
            psf_data["img"] = jnp.array(psf.img)

            # Precompute FFT for full image size (for FFT convolution)
            ph, pw = psf.img.shape
            pad_img = jnp.zeros((h, w))
            # Place PSF in center
            cy, cx = h // 2, w // 2
            y0 = cy - ph // 2
            x0 = cx - pw // 2
            pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(jnp.array(psf.img))
            # Shift to (0,0) for FFT
            pad_img = jnp.fft.ifftshift(pad_img)
            psf_data["fft"] = jfft.rfft2(pad_img)

        elif isinstance(psf, GaussianMixturePSF):
            psf_data["type"] = "mog"
            mog = psf.mog
            psf_data["amp"] = jnp.array(mog.amp)
            psf_data["mean"] = jnp.array(mog.mean)
            psf_data["var"] = jnp.array(mog.var)

        images_data.append(
            {
                "data": data,
                "invvar": invvar,
                "psf": psf_data,
                "shape": (h, w),
                # We don't strictly need WCS object here anymore for rendering,
                # as we precompute pixel positions.
            }
        )

    # 2. Extract Source Data & Group into Batches
    initial_fluxes = []

    # Collectors
    ps_flux_idx = []
    ps_pos_pix = [] # (N_ps, N_img, 2)

    gal_flux_idx = []
    gal_pos_pix = []
    gal_wcs_cd_inv = [] # (N_gal, N_img, 2, 2)
    gal_shapes = []
    gal_profiles = [] # List of dicts

    flux_offset = 0

    for src in catalog:
        # Flux
        br = src.brightness.getParams()
        n_flux = len(br)
        initial_fluxes.extend(br)
        # We assume 1 flux param for the optimization target for now (single band/brightness)
        # If n_flux > 1, we might need to handle band indices.
        # For this implementation, we take the index of the first flux param.
        f_idx = flux_offset
        flux_offset += n_flux

        # Precompute Position & WCS for all images
        pos_pix_list = []
        cd_inv_list = []

        for img in images:
            wcs = img.getWcs()
            # Calculate pixel position (x, y)
            x, y = wcs.positionToPixel(src.getPosition(), src)
            pos_pix_list.append([x, y])

            # Calculate CD Inverse matrix at that pixel
            cd_inv = wcs.cdInverseAtPixel(x, y)
            cd_inv_list.append(cd_inv)

        pos_pix = np.array(pos_pix_list) # (N_img, 2)
        cd_inv = np.array(cd_inv_list)   # (N_img, 2, 2)

        src_type = src.getSourceType()

        # Grouping Logic
        if src_type == 'PointSource':
            ps_flux_idx.append(f_idx)
            ps_pos_pix.append(pos_pix)

        elif isinstance(src, Galaxy) or hasattr(src, 'getProfile'):
            # Generic Galaxy handling
            gal_flux_idx.append(f_idx)
            gal_pos_pix.append(pos_pix)
            gal_wcs_cd_inv.append(cd_inv)

            # Shape (re, ab, phi)
            gal_shapes.append(src.shape.getParams())

            # Profile (MoG)
            prof = src.getProfile()
            if prof is None:
                 # Fallback?
                 pass
            else:
                gal_profiles.append({
                    'amp': np.array(prof.amp),
                    'mean': np.array(prof.mean),
                    'var': np.array(prof.var)
                })
        else:
            print(f"Warning: Unknown source type {src_type} in JAX optimization")

    # 3. Assemble Batches with Padding
    batches = {}

    # -- Point Sources --
    if ps_flux_idx:
        batches['PointSource'] = {
            'flux_idx': jnp.array(ps_flux_idx, dtype=jnp.int32),
            'pos_pix': jnp.array(np.stack(ps_pos_pix), dtype=jnp.float32),
        }

    # -- Galaxies --
    if gal_flux_idx:
        # Zero Padding for MoG profiles
        max_K = 0
        for p in gal_profiles:
            max_K = max(max_K, len(p['amp']))

        amp_list, mean_list, var_list = [], [], []

        for p in gal_profiles:
            K = len(p['amp'])
            pad_len = max_K - K

            amp = p['amp']
            mean = p['mean']
            var = p['var']

            if pad_len > 0:
                amp = np.pad(amp, (0, pad_len), constant_values=0)
                mean = np.pad(mean, ((0, pad_len), (0, 0)), constant_values=0)
                # Pad var with identity (1.0) to avoid potential singularities in linear algebra if used
                var = np.pad(var, ((0, pad_len), (0, 0), (0, 0)), constant_values=1.0)

            amp_list.append(amp)
            mean_list.append(mean)
            var_list.append(var)

        batches['Galaxy'] = {
            'flux_idx': jnp.array(gal_flux_idx, dtype=jnp.int32),
            'pos_pix': jnp.array(np.stack(gal_pos_pix), dtype=jnp.float32),
            'wcs_cd_inv': jnp.array(np.stack(gal_wcs_cd_inv), dtype=jnp.float32),
            'shapes': jnp.array(np.stack(gal_shapes), dtype=jnp.float32),
            'profile': {
                'amp': jnp.array(np.stack(amp_list), dtype=jnp.float32),
                'mean': jnp.array(np.stack(mean_list), dtype=jnp.float32),
                'var': jnp.array(np.stack(var_list), dtype=jnp.float32),
            }
        }

    return images_data, batches, jnp.array(initial_fluxes, dtype=jnp.float32)


def render_batch_point_sources(fluxes, pos_pix, psf_data, img_shape):
    """
    Renders a batch of Point Sources.
    """
    if psf_data['type'] == 'pixelized':
        psf_fft = psf_data['fft']

        # Vmap over sources
        # fluxes: (N_ps,)
        # pos_pix: (N_ps, 2)
        # psf_fft: (H, W) -> broadcasted or passed as is (not mapped)

        render_fn = vmap(render_point_source_fft, in_axes=(0, 0, None, None))

        stamps = render_fn(fluxes, pos_pix, psf_fft, img_shape)
        # Sum over sources
        return jnp.sum(stamps, axis=0)

    elif psf_data['type'] == 'mog':
        # Vmap over sources
        # render_point_source_mog(flux, pos, psf_mix, image_shape)
        psf_mix = (psf_data['amp'], psf_data['mean'], psf_data['var'])

        render_fn = vmap(render_point_source_mog, in_axes=(0, 0, None, None))

        stamps = render_fn(fluxes, pos_pix, psf_mix, img_shape)
        return jnp.sum(stamps, axis=0)

    return jnp.zeros(img_shape)


def render_batch_galaxies(fluxes, pos_pix, wcs_cd_inv, shapes, profiles, psf_data, img_shape):
    """
    Renders a batch of Galaxies.
    """
    if psf_data['type'] == 'pixelized':
        psf_fft = psf_data['fft']

        # Unpack profiles
        # profiles is dict of (N_gal, ...)
        gal_mix = (profiles['amp'], profiles['mean'], profiles['var'])

        # Vmap over sources
        # render_galaxy_fft(galaxy_mix, psf_fft, shape_params, wcs_cd_inv, subpixel_offset, image_shape)

        # render_galaxy_fft does not take flux argument, so we multiply by flux after.
        # But wait, render_galaxy_fft is expensive.
        # vmap over: gal_mix components, shapes, wcs_cd_inv, pos_pix

        # in_axes:
        # gal_mix: (0, 0, 0)
        # psf_fft: None
        # shape_params: 0
        # wcs_cd_inv: 0
        # pos_pix: 0
        # image_shape: None

        render_fn = vmap(render_galaxy_fft, in_axes=((0, 0, 0), None, 0, 0, 0, None))

        stamps = render_fn(gal_mix, psf_fft, shapes, wcs_cd_inv, pos_pix, img_shape)

        # Multiply by fluxes (N_gal, 1, 1) broadcast
        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]

        return jnp.sum(weighted_stamps, axis=0)

    elif psf_data['type'] == 'mog':
        psf_mix = (psf_data['amp'], psf_data['mean'], psf_data['var'])
        gal_mix = (profiles['amp'], profiles['mean'], profiles['var'])

        render_fn = vmap(render_galaxy_mog, in_axes=((0, 0, 0), None, 0, 0, 0, None))

        stamps = render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix, img_shape)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]

        return jnp.sum(weighted_stamps, axis=0)

    return jnp.zeros(img_shape)


def render_scene(fluxes, images_data, batches):
    """
    Renders the scene for all images using batched processing.
    """
    model_images = []

    for img_idx, img_dat in enumerate(images_data):
        H, W = img_dat["shape"]
        img_model = jnp.zeros((H, W))

        # 1. Render Point Sources
        if 'PointSource' in batches:
            batch = batches['PointSource']
            # Get data for this image
            # pos_pix: (N_ps, N_img, 2) -> select img_idx
            pos_pix = batch['pos_pix'][:, img_idx, :]

            # Fluxes
            f_idx = batch['flux_idx']
            batch_fluxes = fluxes[f_idx]

            ps_model = render_batch_point_sources(batch_fluxes, pos_pix, img_dat['psf'], (H, W))
            img_model = img_model + ps_model

        # 2. Render Galaxies
        if 'Galaxy' in batches:
            batch = batches['Galaxy']
            pos_pix = batch['pos_pix'][:, img_idx, :]
            wcs_cd_inv = batch['wcs_cd_inv'][:, img_idx, :, :]
            shapes = batch['shapes']
            profiles = batch['profile']

            f_idx = batch['flux_idx']
            batch_fluxes = fluxes[f_idx]

            gal_model = render_batch_galaxies(batch_fluxes, pos_pix, wcs_cd_inv, shapes, profiles, img_dat['psf'], (H, W))
            img_model = img_model + gal_model

        model_images.append(img_model)

    return model_images


def optimize_fluxes(tractor_obj):
    """
    Optimizes fluxes for forced photometry using JAX.

    Args:
        tractor_obj: Tractor object with images and catalog.

    Returns:
        New Tractor object with optimized fluxes.
    """
    # 1. Precompute/Extract data
    images_data, batches, initial_fluxes = extract_model_data(tractor_obj)

    # 3. Define Loss Function
    def loss_fn(fluxes):
        model_images = render_scene(fluxes, images_data, batches)
        chi2 = 0.0
        for i, model in enumerate(model_images):
            data = images_data[i]["data"]
            invvar = images_data[i]["invvar"]
            diff = data - model
            chi2 += jnp.sum(diff**2 * invvar)
        return chi2

    # 4. Optimize
    # Use Matrix-Free Newton-CG for linear least squares.

    # 1. Compute Gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(initial_fluxes)

    # 2. Define Hessian-Vector Product (HVP)
    def matvec(v):
        return jax.jvp(grad_fn, (initial_fluxes,), (v,))[1]

    # 3. Solve H * step = -grads using CG
    step, info = jax.scipy.sparse.linalg.cg(
        matvec, -grads, maxiter=50
    )  # 50 steps usually enough for convergence

    # 4. Apply Update
    # For a purely quadratic function, one Newton step is exact.
    optimized_fluxes = initial_fluxes + step

    # 5. Update Tractor object
    # We must ensure optimized_fluxes is a numpy array (cpu) before setting params
    # because tractor objects expect numpy usually.
    optimized_fluxes_np = np.array(optimized_fluxes)

    flux_idx = 0
    catalog = tractor_obj.catalog
    for src in catalog:
        n_flux = len(src.brightness.getParams())
        new_flux = optimized_fluxes_np[flux_idx : flux_idx + n_flux]
        src.brightness.setParams(new_flux)
        flux_idx += n_flux

    return tractor_obj
