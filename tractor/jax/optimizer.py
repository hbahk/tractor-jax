import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit, value_and_grad, vmap
import numpy as np
from functools import partial
import jax.image

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
    downsample_image,
)


def extract_model_data(tractor_obj, oversample_rendering=False, fit_background=False):
    """
    Extracts all necessary data from a Tractor object for JAX optimization,
    grouping sources into batches (PointSource, Galaxy) for vectorized rendering.

    Args:
        tractor_obj: Tractor object.
        oversample_rendering: If True, handles oversampled PixelizedPSF by rendering at high resolution.
        fit_background: If True, includes background level in optimization parameters (one scalar per image).

    Returns:
        images_data: list of dicts (data, invvar, psf_data, shape, ...)
        batches: dict containing batched source data.
        initial_fluxes: JAX array of all fluxes.
    """
    from tractor import ConstantSky
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
        # We store type as an integer enum or boolean flag to avoid JAX string issues during JIT/vmap

        if isinstance(psf, PixelizedPSF):
            psf_data["type_code"] = jnp.array(0, dtype=jnp.int32)  # 0 for pixelized

            # Determine sampling
            # In tractor, psf.sampling < 1 usually means oversampling.
            # We convert this to an oversampling factor > 1.
            psf_sampling_raw = getattr(psf, "sampling", 1.0)
            if psf_sampling_raw < 1.0:
                 factor = 1.0 / psf_sampling_raw
            else:
                 factor = 1.0

            if oversample_rendering and factor > 1.0:
                # Store sampling factor (python float for static use)
                psf_data["sampling"] = float(factor)

                # Calculate high-res grid size
                # Assuming integer oversampling roughly
                factor_int = int(round(factor))
                H_hr = h * factor_int
                W_hr = w * factor_int

                # Pad in high-res grid
                ph, pw = psf.img.shape
                pad_img = jnp.zeros((H_hr, W_hr))

                # Place PSF in center
                cy, cx = H_hr // 2, W_hr // 2
                y0 = cy - ph // 2
                x0 = cx - pw // 2

                # Ensure bounds
                pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(jnp.array(psf.img))

                # Shift to (0,0) for FFT
                pad_img = jnp.fft.ifftshift(pad_img)
                psf_data["fft"] = jfft.rfft2(pad_img)

            elif factor > 1.0 and not oversample_rendering:
                 # Downsample PSF to native resolution first
                 psf_data["sampling"] = 1.0

                 # Resize using jax.image.resize (or similar logic)
                 # We want to downsample psf.img from oversampled to native.
                 # Factor is 'factor'.
                 ph, pw = psf.img.shape
                 target_ph = int(round(ph / factor))
                 target_pw = int(round(pw / factor))

                 # Using lanczos3 resize
                 # Note: psf.img is numpy array here, but we can use jax.image.resize on it if we convert to jax array
                 img_jax = jnp.array(psf.img)

                 # Resize expects (H, W, C) or (H, W).
                 # We need to conserve flux.
                 # jax.image.resize interpolates.
                 # downsample_image logic:
                 scale_y = ph / target_ph
                 scale_x = pw / target_pw

                 resized_psf = jax.image.resize(img_jax, (target_ph, target_pw), method='lanczos3')
                 # Scale flux
                 resized_psf = resized_psf * (scale_y * scale_x)

                 # Now pad and FFT at native resolution
                 pad_img = jnp.zeros((h, w))
                 cy, cx = h // 2, w // 2
                 y0 = cy - target_ph // 2
                 x0 = cx - target_pw // 2

                 pad_img = pad_img.at[y0 : y0 + target_ph, x0 : x0 + target_pw].set(resized_psf)
                 pad_img = jnp.fft.ifftshift(pad_img)
                 psf_data["fft"] = jfft.rfft2(pad_img)

            else:
                # Standard path (sampling=1 or ignored)
                psf_data["sampling"] = 1.0

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
            psf_data["type_code"] = jnp.array(1, dtype=jnp.int32)  # 1 for mog
            psf_data["sampling"] = 1.0
            mog = psf.mog
            psf_data["amp"] = jnp.array(mog.amp)
            psf_data["mean"] = jnp.array(mog.mean)
            psf_data["var"] = jnp.array(mog.var)
        else:
             # Unknown PSF
             psf_data["sampling"] = 1.0

        images_data.append(
            {
                "data": data,
                "invvar": invvar,
                "psf": psf_data,
                "shape": (h, w),
            }
        )

    # 2. Extract Source Data & Group into Batches
    initial_fluxes = []

    # Collectors
    ps_flux_idx = []
    ps_pos_pix = []  # (N_ps, N_img, 2)

    gal_flux_idx = []
    gal_pos_pix = []
    gal_wcs_cd_inv = []  # (N_gal, N_img, 2, 2)
    gal_shapes = []
    gal_profiles = []  # List of dicts

    flux_offset = 0

    for src in catalog:
        # Determine Type and potential Profile
        # Handle getSourceType missing
        if hasattr(src, "getSourceType"):
            src_type = src.getSourceType()
        else:
            # Fallback based on class
            if isinstance(src, PointSource):
                src_type = "PointSource"
            elif isinstance(src, Galaxy):
                src_type = "Galaxy"
            else:
                src_type = "Unknown"

        prof = None
        is_galaxy = False

        # Check if source is supported (PointSource, ExpGalaxy, DevGalaxy)
        # Skip CompositeGalaxy explicitly or handle it if possible.
        if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
            print(f"Warning: Skipping CompositeGalaxy {src} in JAX optimization")
            continue

        # Flux (always extract to keep alignment)
        # Assuming src has brightness attribute (True for PointSource, Galaxy subclasses except Composite)
        if hasattr(src, "brightness"):
            br = src.brightness.getParams()
            n_flux = len(br)
            initial_fluxes.extend(br)
            f_idx = flux_offset
            flux_offset += n_flux
        else:
            print(f"Warning: Source {src} has no brightness attribute, skipping.")
            continue

        if isinstance(src, Galaxy) or hasattr(src, "getProfile"):
            is_galaxy = True
            if hasattr(src, "getProfile"):
                prof = src.getProfile()
            # If profile is None (e.g. CompositeGalaxy), we can't batch it as a single profile galaxy.
            if prof is None:
                is_galaxy = False  # Treat as unsupported for batching

        # Precompute Position & WCS for all images (only if we are going to batch it)
        if src_type == "PointSource" or (is_galaxy and prof is not None):
            pos_pix_list = []
            cd_inv_list = []

            for img in images:
                wcs = img.getWcs()
                x, y = wcs.positionToPixel(src.getPosition(), src)
                pos_pix_list.append([x, y])
                cd_inv = wcs.cdInverseAtPixel(x, y)
                cd_inv_list.append(cd_inv)

            pos_pix = np.array(pos_pix_list)  # (N_img, 2)
            cd_inv = np.array(cd_inv_list)  # (N_img, 2, 2)

            if src_type == "PointSource":
                ps_flux_idx.append(f_idx)
                ps_pos_pix.append(pos_pix)

            elif is_galaxy and prof is not None:
                # Safe to append
                gal_flux_idx.append(f_idx)
                gal_pos_pix.append(pos_pix)
                gal_wcs_cd_inv.append(cd_inv)
                gal_shapes.append(src.shape.getAllParams())

                gal_profiles.append(
                    {
                        "amp": np.array(prof.amp),
                        "mean": np.array(prof.mean),
                        "var": np.array(prof.var),
                    }
                )

    # 3. Assemble Batches with Padding
    batches = {}

    # -- Point Sources --
    if ps_flux_idx:
        batches["PointSource"] = {
            "flux_idx": jnp.array(ps_flux_idx, dtype=jnp.int32),
            "pos_pix": jnp.array(np.stack(ps_pos_pix), dtype=jnp.float32),
        }

    # -- Galaxies --
    if gal_flux_idx:
        # Zero Padding for MoG profiles
        max_K = 0
        for p in gal_profiles:
            max_K = max(max_K, len(p["amp"]))

        amp_list, mean_list, var_list = [], [], []

        for p in gal_profiles:
            K = len(p["amp"])
            pad_len = max_K - K

            amp = p["amp"]
            mean = p["mean"]
            var = p["var"]

            if pad_len > 0:
                amp = np.pad(amp, (0, pad_len), constant_values=0)
                mean = np.pad(mean, ((0, pad_len), (0, 0)), constant_values=0)
                # Pad var with identity (1.0) to avoid potential singularities in linear algebra if used
                var = np.pad(var, ((0, pad_len), (0, 0), (0, 0)), constant_values=1.0)

            amp_list.append(amp)
            mean_list.append(mean)
            var_list.append(var)

        batches["Galaxy"] = {
            "flux_idx": jnp.array(gal_flux_idx, dtype=jnp.int32),
            "pos_pix": jnp.array(np.stack(gal_pos_pix), dtype=jnp.float32),
            "wcs_cd_inv": jnp.array(np.stack(gal_wcs_cd_inv), dtype=jnp.float32),
            "shapes": jnp.array(np.stack(gal_shapes), dtype=jnp.float32),
            "profile": {
                "amp": jnp.array(np.stack(amp_list), dtype=jnp.float32),
                "mean": jnp.array(np.stack(mean_list), dtype=jnp.float32),
                "var": jnp.array(np.stack(var_list), dtype=jnp.float32),
            },
        }

    # -- Background --
    if fit_background:
        # Append background parameters to initial_fluxes
        bg_flux_idx = []
        for img in images:
            sky = img.getSky()
            # If sky is ConstantSky, use its value. Else start from 0.
            if hasattr(sky, "val"):
                 val = sky.val
            elif hasattr(sky, "getConstant"):
                 val = sky.getConstant()
            else:
                 val = 0.0

            initial_fluxes.append(val)
            bg_flux_idx.append(flux_offset)
            flux_offset += 1

        batches["Background"] = {
            "flux_idx": jnp.array(bg_flux_idx, dtype=jnp.int32)
        }

    return images_data, batches, jnp.array(initial_fluxes, dtype=jnp.float32)


def render_batch_point_sources(fluxes, pos_pix, psf_data, img_shape):
    """
    Renders a batch of Point Sources.
    """
    sampling = psf_data.get("sampling", 1.0)

    if "fft" in psf_data:
        psf_fft = psf_data["fft"]

        # Prepare inputs for high-res rendering if sampling > 1
        H, W = img_shape
        if sampling > 1.0:
            factor = int(round(sampling))
            H_hr = H * factor
            W_hr = W * factor
            render_shape = (H_hr, W_hr)

            # Scale parameters
            # Position: data pixels -> high res pixels
            # Add offset to align pixel centers.
            # pos_hr = pos_lr * factor + (factor - 1) / 2.0
            pos_pix_scaled = pos_pix * sampling + (sampling - 1.0) / 2.0
        else:
            render_shape = img_shape
            pos_pix_scaled = pos_pix

        # Vmap over sources
        # Use partial to bind any static args if needed?
        # render_point_source_fft takes (flux, pos, psf_fft, image_shape)

        # NOTE: render_point_source_fft assumes image_shape is static tuple.
        # But here it is passed as argument.

        # vmap over fluxes and pos
        # psf_fft is broadcast (None)
        # image_shape is static (None) - wait, if I pass it as argument to vmap, JAX might trace it.
        # It's better to closure it or use partial.

        render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))

        stamps = render_fn(fluxes, pos_pix_scaled, psf_fft)

        # Sum over sources
        combined_img = jnp.sum(stamps, axis=0)

        # Downsample if needed
        if sampling > 1.0:
            combined_img = downsample_image(combined_img, img_shape)

        return combined_img

    else:
        # MoG
        # Vmap over sources
        # render_point_source_mog(flux, pos, psf_mix, image_shape)
        psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])

        render_fn = vmap(partial(render_point_source_mog, image_shape=img_shape), in_axes=(0, 0, None))

        stamps = render_fn(fluxes, pos_pix, psf_mix)
        return jnp.sum(stamps, axis=0)


def render_batch_galaxies(
    fluxes, pos_pix, wcs_cd_inv, shapes, profiles, psf_data, img_shape
):
    """
    Renders a batch of Galaxies.
    """
    sampling = psf_data.get("sampling", 1.0)

    if "fft" in psf_data:
        psf_fft = psf_data["fft"]

        H, W = img_shape
        if sampling > 1.0:
            factor = int(round(sampling))
            H_hr = H * factor
            W_hr = W * factor
            render_shape = (H_hr, W_hr)

            # Scale parameters
            pos_pix_scaled = pos_pix * sampling + (sampling - 1.0) / 2.0
            wcs_cd_inv_scaled = wcs_cd_inv * sampling
        else:
            render_shape = img_shape
            pos_pix_scaled = pos_pix
            wcs_cd_inv_scaled = wcs_cd_inv

        # Unpack profiles
        # profiles is dict of (N_gal, ...)
        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        # Vmap over sources
        # render_galaxy_fft(galaxy_mix, psf_fft, shape_params, wcs_cd_inv, subpixel_offset, image_shape)

        # in_axes:
        # gal_mix: (0, 0, 0)
        # psf_fft: None
        # shape_params: 0
        # wcs_cd_inv: 0
        # pos_pix: 0

        render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))

        stamps = render_fn(gal_mix, psf_fft, shapes, wcs_cd_inv_scaled, pos_pix_scaled)

        # Multiply by fluxes (N_gal, 1, 1) broadcast
        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]

        combined_img = jnp.sum(weighted_stamps, axis=0)

        # Downsample if needed
        if sampling > 1.0:
            combined_img = downsample_image(combined_img, img_shape)

        return combined_img

    else:
        # MoG
        psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        render_fn = vmap(partial(render_galaxy_mog, image_shape=img_shape), in_axes=((0, 0, 0), None, 0, 0, 0))

        stamps = render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]

        return jnp.sum(weighted_stamps, axis=0)


def render_scene(fluxes, images_data, batches):
    """
    Renders the scene for all images using batched processing.
    """
    model_images = []

    for img_idx, img_dat in enumerate(images_data):
        # Determine output shape
        # If 'data' is available, use its shape.
        if "data" in img_dat:
            H, W = img_dat["data"].shape
        elif "shape" in img_dat:
            H, W = img_dat["shape"]
        else:
            raise ValueError("Cannot determine image shape")

        img_model = jnp.zeros((H, W))

        # 1. Render Point Sources
        if "PointSource" in batches:
            batch = batches["PointSource"]
            # Get data for this image
            # pos_pix: (N_ps, N_img, 2) -> select img_idx
            pos_pix = batch["pos_pix"][:, img_idx, :]

            # Fluxes
            f_idx = batch["flux_idx"]
            batch_fluxes = fluxes[f_idx]

            ps_model = render_batch_point_sources(
                batch_fluxes, pos_pix, img_dat["psf"], (H, W)
            )
            img_model = img_model + ps_model

        # 2. Render Galaxies
        if "Galaxy" in batches:
            batch = batches["Galaxy"]
            pos_pix = batch["pos_pix"][:, img_idx, :]
            wcs_cd_inv = batch["wcs_cd_inv"][:, img_idx, :, :]
            shapes = batch["shapes"]
            profiles = batch["profile"]

            f_idx = batch["flux_idx"]
            batch_fluxes = fluxes[f_idx]

            gal_model = render_batch_galaxies(
                batch_fluxes,
                pos_pix,
                wcs_cd_inv,
                shapes,
                profiles,
                img_dat["psf"],
                (H, W),
            )
            img_model = img_model + gal_model

        # 3. Background
        if "Background" in batches:
            batch = batches["Background"]
            f_idx = batch["flux_idx"][img_idx]
            bg_val = fluxes[f_idx]
            img_model = img_model + bg_val

        model_images.append(img_model)

    return model_images


def compute_fisher_diagonal(images_data, batches, n_flux):
    """
    Computes the diagonal of the Fisher Information Matrix.
    F_ss = sum_pixels ( (dModel/dFlux_s)^2 * invvar )
    This avoids computing the full Hessian.
    """
    fisher_diag = jnp.zeros(n_flux)

    for img_idx, img_dat in enumerate(images_data):
        # Determine output shape
        if "data" in img_dat:
            H, W = img_dat["data"].shape
        elif "shape" in img_dat:
            H, W = img_dat["shape"]
        else:
            raise ValueError("Cannot determine image shape")

        invvar = img_dat["invvar"] # (H, W)

        # 1. Point Sources
        if "PointSource" in batches:
            batch = batches["PointSource"]
            pos_pix = batch["pos_pix"][:, img_idx, :] # (N_ps, 2)
            f_idx = batch["flux_idx"]

            # Unit fluxes for derivatives
            N_ps = pos_pix.shape[0]
            unit_fluxes = jnp.ones(N_ps)

            psf_data = img_dat["psf"]
            sampling = psf_data.get("sampling", 1.0)

            stamps = None

            if "fft" in psf_data:
                psf_fft = psf_data["fft"]

                if sampling > 1.0:
                    factor = int(round(sampling))
                    H_hr = H * factor
                    W_hr = W * factor
                    render_shape = (H_hr, W_hr)
                    pos_pix_scaled = pos_pix * sampling + (sampling - 1.0) / 2.0
                else:
                    render_shape = (H, W)
                    pos_pix_scaled = pos_pix

                render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))
                stamps = render_fn(unit_fluxes, pos_pix_scaled, psf_fft) # (N_ps, H_r, W_r)

            else:
                # MoG
                psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
                render_fn = vmap(partial(render_point_source_mog, image_shape=(H, W)), in_axes=(0, 0, None))
                stamps = render_fn(unit_fluxes, pos_pix, psf_mix)

            # Downsample if needed
            if sampling > 1.0:
                 # vmap downsample
                 ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                 stamps = ds_fn(stamps)

            # Compute contribution: sum(stamp^2 * invvar)
            contrib = jnp.sum(stamps**2 * invvar[jnp.newaxis, :, :], axis=(1, 2))
            fisher_diag = fisher_diag.at[f_idx].add(contrib)

        # 2. Galaxies
        if "Galaxy" in batches:
            batch = batches["Galaxy"]
            pos_pix = batch["pos_pix"][:, img_idx, :]
            wcs_cd_inv = batch["wcs_cd_inv"][:, img_idx, :, :]
            shapes = batch["shapes"]
            profiles = batch["profile"]
            f_idx = batch["flux_idx"]

            psf_data = img_dat["psf"]
            sampling = psf_data.get("sampling", 1.0)

            stamps = None

            if "fft" in psf_data:
                psf_fft = psf_data["fft"]
                if sampling > 1.0:
                    factor = int(round(sampling))
                    H_hr = H * factor
                    W_hr = W * factor
                    render_shape = (H_hr, W_hr)
                    pos_pix_scaled = pos_pix * sampling + (sampling - 1.0) / 2.0
                    wcs_cd_inv_scaled = wcs_cd_inv * sampling
                else:
                    render_shape = (H, W)
                    pos_pix_scaled = pos_pix
                    wcs_cd_inv_scaled = wcs_cd_inv

                gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
                render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
                stamps = render_fn(gal_mix, psf_fft, shapes, wcs_cd_inv_scaled, pos_pix_scaled)

            else:
                # MoG
                psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
                gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
                render_fn = vmap(partial(render_galaxy_mog, image_shape=(H, W)), in_axes=((0, 0, 0), None, 0, 0, 0))
                stamps = render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix)

            # Downsample
            if sampling > 1.0:
                 ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                 stamps = ds_fn(stamps)

            contrib = jnp.sum(stamps**2 * invvar[jnp.newaxis, :, :], axis=(1, 2))
            fisher_diag = fisher_diag.at[f_idx].add(contrib)

        # 3. Background
        if "Background" in batches:
            batch = batches["Background"]
            f_idx = batch["flux_idx"][img_idx]

            # Derivative of model w.r.t background is 1.0 everywhere.
            # Contribution is sum(1.0^2 * invvar) = sum(invvar)
            contrib = jnp.sum(invvar)
            fisher_diag = fisher_diag.at[f_idx].add(contrib)

    return fisher_diag


def solve_fluxes_core(initial_fluxes, images_data, batches, return_variances=False):
    """
    Pure JAX core optimization logic.
    Can be vmapped over batches of images/sources if data is stacked.

    Args:
        initial_fluxes: JAX array (N_flux,)
        images_data: list of dicts (for N_bands), each leaf is JAX array.
        batches: dict of batched source data (pytree leaves are JAX arrays).
        return_variances: bool, if True, calculate and return variances.

    Returns:
        optimized_fluxes: JAX array (N_flux,)
        variances: JAX array (N_flux,) (only if return_variances=True)
    """

    def loss_fn(fluxes):
        model_images = render_scene(fluxes, images_data, batches)
        chi2 = 0.0
        for i, model in enumerate(model_images):
            data = images_data[i]["data"]
            invvar = images_data[i]["invvar"]
            diff = data - model
            chi2 += jnp.sum(diff**2 * invvar)
        return chi2

    # Use Matrix-Free Newton-CG for linear least squares.
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(initial_fluxes)

    def matvec(v):
        return jax.jvp(grad_fn, (initial_fluxes,), (v,))[1]

    step, info = jax.scipy.sparse.linalg.cg(matvec, -grads, maxiter=50)

    optimized_fluxes = initial_fluxes + step

    if return_variances:
        # Compute Fisher Diagonal directly (Scalar Fisher)
        # Avoids computing full Hessian.
        fisher_diag = compute_fisher_diagonal(images_data, batches, len(initial_fluxes))

        # Variance = 1.0 / F_ss (since Hessian approx = 2 * F_ss, and Variance = 2 / Hessian)

        # Avoid division by zero
        fisher_diag = jnp.where(fisher_diag <= 0, 1e-12, fisher_diag)

        variances = 1.0 / fisher_diag

        return optimized_fluxes, variances

    return optimized_fluxes


def optimize_fluxes(tractor_obj, oversample_rendering=False, return_variances=False, fit_background=False):
    """
    Optimizes fluxes for forced photometry using JAX.

    Args:
        tractor_obj: Tractor object with images and catalog.
        oversample_rendering: bool, if True use oversampled rendering for PixelizedPSF with sampling != 1.
        return_variances: bool, if True, return variances of fluxes.
        fit_background: bool, if True, includes background level in optimization parameters.

    Returns:
        New Tractor object with optimized fluxes.
        (Optional) variances array if return_variances=True.
    """
    from tractor import ConstantSky

    # 1. Precompute/Extract data
    images_data, batches, initial_fluxes = extract_model_data(
        tractor_obj,
        oversample_rendering=oversample_rendering,
        fit_background=fit_background
    )

    # 2. Run JAX Optimization Core
    # We use the pure function here.
    if return_variances:
        optimized_fluxes, variances = solve_fluxes_core(initial_fluxes, images_data, batches, return_variances=True)
    else:
        optimized_fluxes = solve_fluxes_core(initial_fluxes, images_data, batches, return_variances=False)

    # 3. Update Tractor object
    optimized_fluxes_np = np.array(optimized_fluxes)

    flux_idx = 0
    catalog = tractor_obj.catalog
    for src in catalog:
        # Skip sources that were skipped in extraction (Composite)
        if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
            continue
        # Also ensure it has brightness
        if hasattr(src, "brightness"):
            n_flux = len(src.brightness.getParams())
            new_flux = optimized_fluxes_np[flux_idx : flux_idx + n_flux]
            src.brightness.setParams(new_flux)
            flux_idx += n_flux

    # Update Background if fitted
    if fit_background and "Background" in batches:
        bg_flux_idx = batches["Background"]["flux_idx"]
        # bg_flux_idx is array of indices for each image
        # indices are JAX array, convert to numpy
        bg_flux_idx = np.array(bg_flux_idx)

        for i, idx in enumerate(bg_flux_idx):
            bg_val = optimized_fluxes_np[idx]
            img = tractor_obj.images[i]
            # Update sky
            if isinstance(img.sky, ConstantSky):
                img.sky.val = bg_val
            else:
                # Replace with ConstantSky
                img.sky = ConstantSky(bg_val)

    if return_variances:
        return tractor_obj, np.array(variances)

    return tractor_obj
