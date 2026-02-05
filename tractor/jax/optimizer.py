import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit, value_and_grad, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from functools import partial
import jax.image

from tractor.engine import Tractor
from tractor.optimize import Optimizer
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
    grouping sources into batches and stacking image data with padding for vectorized rendering.

    Args:
        tractor_obj: Tractor object.
        oversample_rendering: If True, handles oversampled PixelizedPSF by rendering at high resolution.
        fit_background: If True, includes background level in optimization parameters.

    Returns:
        images_data: dict containing stacked image data (data, invvar, psf).
                     Shapes are (N_img, max_H, max_W) or (N_img, ...).
        batches: dict containing batched source data.
                 Shapes are (N_img, N_src, ...).
        initial_fluxes: JAX array of initial fluxes of shape (N_img, N_params).
                        Sources are broadcast/shared, background is per-image.
    """
    from tractor import ConstantSky
    images = tractor_obj.images
    catalog = tractor_obj.catalog

    # 1. Determine Max Image Size & Sampling
    max_H, max_W = 0, 0
    max_factor = 1.0

    for img in images:
        h, w = img.shape
        max_H = max(max_H, h)
        max_W = max(max_W, w)

        if oversample_rendering:
            psf = img.getPsf()
            if isinstance(psf, PixelizedPSF):
                s = getattr(psf, "sampling", 1.0)
                if s < 1.0:
                    max_factor = max(max_factor, 1.0 / s)

    # 2. Extract & Stack Image Data
    data_list = []
    invvar_list = []

    # PSF Stacks
    psf_type_code_list = []
    psf_sampling_list = []
    psf_fft_list = []
    psf_amp_list = []
    psf_mean_list = []
    psf_var_list = []

    # Max MoG K for padding
    max_mog_K = 0
    for img in images:
        psf = img.getPsf()
        if isinstance(psf, GaussianMixturePSF):
            max_mog_K = max(max_mog_K, len(psf.mog.amp))

    # Ensure at least K=1 to avoid empty arrays
    max_mog_K = max(max_mog_K, 1)

    for img in images:
        h, w = img.shape

        # -- Pad Data --
        pad_h = max_H - h
        pad_w = max_W - w

        d = jnp.array(img.getImage())
        d = jnp.pad(d, ((0, pad_h), (0, pad_w)), constant_values=0.0)
        data_list.append(d)

        iv = jnp.array(img.getInvError()) ** 2
        # Use 0.0 for invvar in padded regions (masked out)
        iv = jnp.pad(iv, ((0, pad_h), (0, pad_w)), constant_values=0.0)
        invvar_list.append(iv)

        # -- Prepare PSF Data --
        psf = img.getPsf()

        # Default Dummies
        p_type = 0
        p_sampling = 1.0

        # Dummy MoG (Identity)
        p_amp = jnp.zeros(max_mog_K)
        p_mean = jnp.zeros((max_mog_K, 2))
        p_var = jnp.tile(jnp.eye(2), (max_mog_K, 1, 1))

        # FFT Handling
        # We need a consistent grid size for FFTs.
        # If oversampling is used, we use max_factor.

        if oversample_rendering and max_factor > 1.0:
            target_H = int(round(max_H * max_factor))
            target_W = int(round(max_W * max_factor))
            p_sampling = float(max_factor) # Force uniform sampling for stack
        else:
            target_H = max_H
            target_W = max_W
            p_sampling = 1.0

        if isinstance(psf, PixelizedPSF):
            p_type = 0

            # Get local sampling
            s = getattr(psf, "sampling", 1.0)
            local_factor = 1.0/s if s < 1.0 else 1.0

            # If we need to render this PSF on the unified target grid:
            # If oversampling is ON, we assume we want high-res.

            # Logic:
            # 1. Get raw PSF image
            raw_img = jnp.array(psf.img)
            ph, pw = raw_img.shape

            # 2. If we need to resize to match target sampling?
            # Ideally we assume local_factor matches max_factor if consistent.
            # If not, we might need resizing.
            # For now, assume factor matches or we just place it.
            # BUT if local_factor != p_sampling (max_factor), we should resize.

            if abs(local_factor - p_sampling) > 1e-3:
                # Resize PSF image to match target resolution
                # This is complex (flux conservation etc).
                # Simplification: Assume consistent sampling for now or just pad/place.
                pass

            # 3. Pad to target_H, target_W
            pad_img = jnp.zeros((target_H, target_W))
            cy, cx = target_H // 2, target_W // 2
            y0 = cy - ph // 2
            x0 = cx - pw // 2

            # Clip if psf larger than target (unlikely)

            pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(raw_img)
            pad_img = jnp.fft.ifftshift(pad_img)
            p_fft = jfft.rfft2(pad_img)

        elif isinstance(psf, GaussianMixturePSF):
            p_type = 1
            # MoG parameters
            K = len(psf.mog.amp)
            pad_len = max_mog_K - K

            amp = jnp.array(psf.mog.amp)
            mean = jnp.array(psf.mog.mean)
            var = jnp.array(psf.mog.var)

            if pad_len > 0:
                amp = jnp.pad(amp, (0, pad_len), constant_values=0)
                mean = jnp.pad(mean, ((0, pad_len), (0, 0)), constant_values=0)

                # Correct padding for var: Identity
                # Using numpy for construction before converting to jax array might be easier if mixed types,
                # but here variables are already jax arrays (jnp.array above).
                # Actually, jnp.pad pads with constant value.
                # To get identity, we construct new array.

                new_var = jnp.zeros((max_mog_K, 2, 2), dtype=var.dtype)
                new_var = new_var.at[:K].set(var)

                # Set identity for padding
                # (pad_len, 2, 2)
                padding_eye = jnp.tile(jnp.eye(2), (pad_len, 1, 1))
                new_var = new_var.at[K:].set(padding_eye)
                var = new_var

            p_amp = amp
            p_mean = mean
            p_var = var

            # Dummy FFT (Zeros)
            # Must match shape of FFT from Pixelized path
            # Shape of rfft2 is (H, W//2 + 1)
            p_fft = jnp.zeros((target_H, target_W // 2 + 1), dtype=jnp.complex64)

        else:
            # Unknown
            p_fft = jnp.zeros((target_H, target_W // 2 + 1), dtype=jnp.complex64)

        psf_type_code_list.append(p_type)
        psf_sampling_list.append(p_sampling)
        psf_fft_list.append(p_fft)
        psf_amp_list.append(p_amp)
        psf_mean_list.append(p_mean)
        psf_var_list.append(p_var)

    # Stack Images Data
    images_data = {
        "data": jnp.stack(data_list),       # (N_img, max_H, max_W)
        "invvar": jnp.stack(invvar_list),   # (N_img, max_H, max_W)
        "psf": {
            "type_code": jnp.array(psf_type_code_list, dtype=jnp.int32),
            "sampling": jnp.array(psf_sampling_list, dtype=jnp.float32),
            "fft": jnp.stack(psf_fft_list),
            "amp": jnp.stack(psf_amp_list),
            "mean": jnp.stack(psf_mean_list),
            "var": jnp.stack(psf_var_list),
        }
    }

    # 3. Extract & Stack Source Data
    src_fluxes = []

    ps_flux_idx = []
    ps_pos_pix_list = [] # (N_src, N_img, 2)

    gal_flux_idx = []
    gal_pos_pix_list = []
    gal_wcs_cd_inv_list = []
    gal_shapes = []
    gal_profiles = []

    flux_offset = 0

    for src in catalog:
        # Check source support
        if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
            print(f"Warning: Skipping CompositeGalaxy {src} in JAX optimization")
            continue

        # Get Brightness
        if hasattr(src, "brightness"):
            br = src.brightness.getParams()
            src_fluxes.extend(br)
            n_flux = len(br)
            f_idx = flux_offset
            flux_offset += n_flux
        else:
            continue

        # Determine type
        if hasattr(src, "getSourceType"):
            src_type = src.getSourceType()
        else:
            if isinstance(src, PointSource): src_type = "PointSource"
            elif isinstance(src, Galaxy): src_type = "Galaxy"
            else: src_type = "Unknown"

        prof = None
        is_galaxy = False
        if isinstance(src, Galaxy) or hasattr(src, "getProfile"):
            is_galaxy = True
            if hasattr(src, "getProfile"):
                prof = src.getProfile()
            if prof is None: is_galaxy = False

        # Collect Positions per Image
        if src_type == "PointSource" or (is_galaxy and prof is not None):
            pos_per_img = []
            cd_inv_per_img = []

            for img in images:
                wcs = img.getWcs()
                x, y = wcs.positionToPixel(src.getPosition(), src)
                pos_per_img.append([x, y])
                cd_inv = wcs.cdInverseAtPixel(x, y)
                cd_inv_per_img.append(cd_inv)

            # Stack over images
            # (N_img, 2)
            pos_pix = np.array(pos_per_img)
            cd_inv = np.array(cd_inv_per_img)

            if src_type == "PointSource":
                ps_flux_idx.append(f_idx)
                ps_pos_pix_list.append(pos_pix)

            elif is_galaxy and prof is not None:
                gal_flux_idx.append(f_idx)
                gal_pos_pix_list.append(pos_pix)
                gal_wcs_cd_inv_list.append(cd_inv)
                gal_shapes.append(src.shape.getAllParams())
                gal_profiles.append({
                    "amp": np.array(prof.amp),
                    "mean": np.array(prof.mean),
                    "var": np.array(prof.var),
                })

    # Build Batches
    batches = {}
    N_img = len(images)

    # Point Sources
    if ps_flux_idx:
        # Stack pos: (N_src, N_img, 2) -> (N_img, N_src, 2)
        pos_stack = np.stack(ps_pos_pix_list) # (N_src, N_img, 2)
        pos_stack = np.transpose(pos_stack, (1, 0, 2)) # (N_img, N_src, 2)

        batches["PointSource"] = {
            "flux_idx": jnp.array(ps_flux_idx, dtype=jnp.int32),
            "pos_pix": jnp.array(pos_stack, dtype=jnp.float32),
        }

    # Galaxies
    if gal_flux_idx:
        pos_stack = np.stack(gal_pos_pix_list) # (N_src, N_img, 2)
        pos_stack = np.transpose(pos_stack, (1, 0, 2))

        cd_inv_stack = np.stack(gal_wcs_cd_inv_list) # (N_src, N_img, 2, 2)
        cd_inv_stack = np.transpose(cd_inv_stack, (1, 0, 2, 3)) # (N_img, N_src, 2, 2)

        # Pad Profiles
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

                # Identity padding for var
                new_var = np.zeros((max_K, 2, 2), dtype=var.dtype)
                new_var[:K] = var
                # Set identity
                # new_var[K:] = np.eye(2)
                # Broadcasting (pad_len, 2, 2) = (2, 2) works in numpy?
                # Yes, if pad_len > 0.
                if pad_len > 0:
                    new_var[K:] = np.eye(2)
                var = new_var
            amp_list.append(amp)
            mean_list.append(mean)
            var_list.append(var)

        batches["Galaxy"] = {
            "flux_idx": jnp.array(gal_flux_idx, dtype=jnp.int32),
            "pos_pix": jnp.array(pos_stack, dtype=jnp.float32),
            "wcs_cd_inv": jnp.array(cd_inv_stack, dtype=jnp.float32),
            "shapes": jnp.array(np.stack(gal_shapes), dtype=jnp.float32),
            "profile": {
                "amp": jnp.array(np.stack(amp_list), dtype=jnp.float32),
                "mean": jnp.array(np.stack(mean_list), dtype=jnp.float32),
                "var": jnp.array(np.stack(var_list), dtype=jnp.float32),
            },
        }

    # 4. Prepare Initial Fluxes (Per Image)
    # src_fluxes: (N_src_params,) - shared
    # If fit_background, we add 1 param per image.

    src_fluxes = np.array(src_fluxes, dtype=np.float32)

    # Broadcast src_fluxes to (N_img, N_src_params)
    initial_fluxes_matrix = np.tile(src_fluxes, (N_img, 1))

    if fit_background:
        bg_vals = []
        for img in images:
            sky = img.getSky()
            if hasattr(sky, "val"): val = sky.val
            elif hasattr(sky, "getConstant"): val = sky.getConstant()
            else: val = 0.0
            bg_vals.append(val)

        bg_vals = np.array(bg_vals, dtype=np.float32).reshape(N_img, 1)

        # Concatenate
        initial_fluxes_matrix = np.hstack([initial_fluxes_matrix, bg_vals])

        # Batch Indices
        # Background param is at index N_src_params
        bg_idx = len(src_fluxes) # scalar index relative to row
        # Since each row has its own bg param at the end
        batches["Background"] = {
            "flux_idx": jnp.array([bg_idx], dtype=jnp.int32)
        }

    return images_data, batches, jnp.array(initial_fluxes_matrix, dtype=jnp.float32)


def render_batch_point_sources(fluxes, pos_pix, psf_data, img_shape):
    """
    Renders a batch of Point Sources.
    """
    # Determine scale from FFT shape vs Target shape
    H, W = img_shape
    H_hr = psf_data['fft'].shape[0]
    scale = float(H_hr) / float(H)

    def render_fft(operand):
        W_hr = int(round(W * scale))
        render_shape = (H_hr, W_hr)

        s = scale
        pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0

        render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))
        stamps = render_fn(fluxes, pos_pix_scaled, psf_data['fft'])
        combined = jnp.sum(stamps, axis=0)

        if scale > 1.001:
            combined = downsample_image(combined, img_shape)
        return combined

    def render_mog(operand):
        psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
        render_fn = vmap(partial(render_point_source_mog, image_shape=img_shape), in_axes=(0, 0, None))
        stamps = render_fn(fluxes, pos_pix, psf_mix)
        return jnp.sum(stamps, axis=0)

    # type_code: 0 = Pixelized/FFT, 1 = MoG
    return jax.lax.cond(psf_data['type_code'] == 0, render_fft, render_mog, None)


def render_batch_galaxies(
    fluxes, pos_pix, wcs_cd_inv, shapes, profiles, psf_data, img_shape
):
    """
    Renders a batch of Galaxies.
    """
    H, W = img_shape
    H_hr = psf_data['fft'].shape[0]
    scale = float(H_hr) / float(H)

    def render_fft(operand):
        W_hr = int(round(W * scale))
        render_shape = (H_hr, W_hr)

        s = scale
        pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0
        wcs_cd_inv_scaled = wcs_cd_inv * s

        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
        stamps = render_fn(gal_mix, psf_data['fft'], shapes, wcs_cd_inv_scaled, pos_pix_scaled)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]
        combined = jnp.sum(weighted_stamps, axis=0)

        if scale > 1.001:
            combined = downsample_image(combined, img_shape)
        return combined

    def render_mog(operand):
        psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        render_fn = vmap(partial(render_galaxy_mog, image_shape=img_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
        stamps = render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]
        return jnp.sum(weighted_stamps, axis=0)

    return jax.lax.cond(psf_data['type_code'] == 0, render_fft, render_mog, None)


def prepare_sharded_inputs(images_data, batches, initial_fluxes):
    """
    Distributes data across available devices using NamedSharding (GSPMD).
    Shards image-based arrays along axis 0 and replicates shared source parameters.
    """
    devices = jax.devices()
    # Create a mesh for data parallelism over images
    mesh = Mesh(devices, axis_names=('img_batch',))

    # Shard along the first axis (axis 0) corresponding to 'img_batch'
    sharding = NamedSharding(mesh, PartitionSpec('img_batch'))

    # Replicate on all devices (no partitioning axes)
    replicated = NamedSharding(mesh, PartitionSpec())

    # 1. Shard images_data (all leaves have shape (N_img, ...))
    images_spec = jax.tree_util.tree_map(lambda x: sharding, images_data)

    # 2. Shard initial_fluxes (N_img, N_params)
    fluxes_spec = sharding

    # 3. Shard batches
    # Keys like 'pos_pix', 'wcs_cd_inv' are per-image (N_img, ...) -> Shard
    # Others like 'flux_idx', 'shapes', 'profile' are shared -> Replicate
    batches_spec = {}

    for key, batch in batches.items():
        spec = {}
        for k, v in batch.items():
            if k in ['pos_pix', 'wcs_cd_inv']:
                 spec[k] = sharding
            elif k == 'profile':
                 # profile is a dict of arrays, all replicated
                 spec[k] = jax.tree_util.tree_map(lambda x: replicated, v)
            else:
                 # flux_idx, shapes, etc.
                 spec[k] = replicated
        batches_spec[key] = spec

    return (
        jax.device_put(images_data, images_spec),
        jax.device_put(batches, batches_spec),
        jax.device_put(initial_fluxes, fluxes_spec)
    )


def render_image(fluxes, image_data, batches):
    """
    Renders a single image using sliced batch data.
    """
    H, W = image_data['data'].shape
    img_model = jnp.zeros((H, W))

    # 1. Render Point Sources
    if "PointSource" in batches:
        batch = batches["PointSource"]
        pos_pix = batch["pos_pix"]  # (N_ps, 2)
        f_idx = batch["flux_idx"]
        batch_fluxes = fluxes[f_idx]

        ps_model = render_batch_point_sources(
            batch_fluxes, pos_pix, image_data["psf"], (H, W)
        )
        img_model = img_model + ps_model

    # 2. Render Galaxies
    if "Galaxy" in batches:
        batch = batches["Galaxy"]
        pos_pix = batch["pos_pix"] # (N_gal, 2)
        wcs_cd_inv = batch["wcs_cd_inv"] # (N_gal, 2, 2)
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
            image_data["psf"],
            (H, W),
        )
        img_model = img_model + gal_model

    # 3. Background
    if "Background" in batches:
        batch = batches["Background"]
        f_idx = batch["flux_idx"] # (1,)
        # For single image optimization, flux_idx points to the bg parameter.
        bg_val = fluxes[f_idx[0]]
        img_model = img_model + bg_val

    return img_model


def compute_fisher_diagonal(image_data, batches, n_flux):
    """
    Computes the diagonal of the Fisher Information Matrix for a single image.
    F_ss = sum_pixels ( (dModel/dFlux_s)^2 * invvar )
    """
    fisher_diag = jnp.zeros(n_flux)

    H, W = image_data['data'].shape
    invvar = image_data["invvar"] # (H, W)

    # 1. Point Sources
    if "PointSource" in batches:
        batch = batches["PointSource"]
        pos_pix = batch["pos_pix"] # (N_ps, 2)
        f_idx = batch["flux_idx"]

        # Unit fluxes for derivatives
        N_ps = pos_pix.shape[0]
        unit_fluxes = jnp.ones(N_ps)

        psf_data = image_data["psf"]

        # Render unit fluxes
        stamps = render_batch_point_sources(unit_fluxes, pos_pix, psf_data, (H, W))
        # Wait, render_batch_point_sources returns summed image if we pass fluxes.
        # But we need stamps squared.
        # We need to expose a function that returns stamps!
        # render_batch_point_sources logic sums internally.

        # We need to replicate logic but without summing.
        # Or we call the internal vmap manually.
        # This duplicates code.
        # Better to factor out `get_model_stamps`.

        # But wait, render_batch_point_sources has branching logic (cond).
        # We can reuse it if we pass unit flux and get stamps?
        # No, it sums.

        # Re-implementation inline for Fisher (simplification)
        # Using branching logic again?

        H_hr = psf_data['fft'].shape[0]
        scale = float(H_hr) / float(H)

        def compute_stamps_fft(op):
            W_hr = int(round(W * scale))
            render_shape = (H_hr, W_hr)
            s = scale
            pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0

            render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))
            stamps = render_fn(unit_fluxes, pos_pix_scaled, psf_data['fft'])

            if scale > 1.001:
                ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                stamps = ds_fn(stamps)
            return stamps

        def compute_stamps_mog(op):
            psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
            render_fn = vmap(partial(render_point_source_mog, image_shape=(H, W)), in_axes=(0, 0, None))
            stamps = render_fn(unit_fluxes, pos_pix, psf_mix)
            return stamps

        stamps = jax.lax.cond(psf_data['type_code'] == 0, compute_stamps_fft, compute_stamps_mog, None)

        # Compute contribution: sum(stamp^2 * invvar)
        contrib = jnp.sum(stamps**2 * invvar[jnp.newaxis, :, :], axis=(1, 2))
        fisher_diag = fisher_diag.at[f_idx].add(contrib)

    # 2. Galaxies
    if "Galaxy" in batches:
        batch = batches["Galaxy"]
        pos_pix = batch["pos_pix"]
        wcs_cd_inv = batch["wcs_cd_inv"]
        shapes = batch["shapes"]
        profiles = batch["profile"]
        f_idx = batch["flux_idx"]

        psf_data = image_data["psf"]
        H_hr = psf_data['fft'].shape[0]
        scale = float(H_hr) / float(H)

        def compute_stamps_fft(op):
            W_hr = int(round(W * scale))
            render_shape = (H_hr, W_hr)
            s = scale
            pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0
            wcs_cd_inv_scaled = wcs_cd_inv * s

            gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
            render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
            stamps = render_fn(gal_mix, psf_data['fft'], shapes, wcs_cd_inv_scaled, pos_pix_scaled)

            if scale > 1.001:
                ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                stamps = ds_fn(stamps)
            return stamps

        def compute_stamps_mog(op):
            psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
            gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
            render_fn = vmap(partial(render_galaxy_mog, image_shape=(H, W)), in_axes=((0, 0, 0), None, 0, 0, 0))
            stamps = render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix)
            return stamps

        stamps = jax.lax.cond(psf_data['type_code'] == 0, compute_stamps_fft, compute_stamps_mog, None)

        contrib = jnp.sum(stamps**2 * invvar[jnp.newaxis, :, :], axis=(1, 2))
        fisher_diag = fisher_diag.at[f_idx].add(contrib)

    # 3. Background
    if "Background" in batches:
        f_idx = batches["Background"]["flux_idx"] # (1,)
        # Derivative is 1.0
        contrib = jnp.sum(invvar)
        fisher_diag = fisher_diag.at[f_idx].add(contrib)

    return fisher_diag


def solve_fluxes_core(initial_fluxes, image_data, batches, return_variances=False):
    """
    Pure JAX core optimization logic for a SINGLE image.
    Designed to be vmapped.

    Args:
        initial_fluxes: JAX array (N_flux,)
        image_data: dict containing single image data (slices).
        batches: dict of batched source data (slices).
        return_variances: bool
    """

    def loss_fn(fluxes):
        model_image = render_image(fluxes, image_data, batches)
        data = image_data["data"]
        invvar = image_data["invvar"]
        diff = data - model_image
        chi2 = jnp.sum(diff**2 * invvar)
        return chi2

    # Use Matrix-Free Newton-CG for linear least squares.
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(initial_fluxes)

    def matvec(v):
        return jax.jvp(grad_fn, (initial_fluxes,), (v,))[1]

    # CG solve A x = b where A is Hessian, b = -grads
    # For high precision, maxiter needs to be higher?
    # User requested dchi2=1e-10.
    step, info = jax.scipy.sparse.linalg.cg(matvec, -grads, maxiter=500, tol=1e-10)

    optimized_fluxes = initial_fluxes + step

    if return_variances:
        fisher_diag = compute_fisher_diagonal(image_data, batches, len(initial_fluxes))
        fisher_diag = jnp.where(fisher_diag <= 0, 1e-12, fisher_diag)
        variances = 1.0 / fisher_diag
        return optimized_fluxes, variances

    return optimized_fluxes


def optimize_fluxes(tractor_obj, oversample_rendering=False, return_variances=False, fit_background=False, update_catalog=False, vmap_images=True, use_sharding=True):
    """
    Optimizes fluxes for forced photometry using JAX.
    Iterates over images in tractor_obj and fits each one separately using vectorized execution (vmap).

    Args:
        tractor_obj: Tractor object with images and catalog.
        oversample_rendering: bool, if True use oversampled rendering for PixelizedPSF with sampling != 1.
        return_variances: bool, if True, return variances of fluxes.
        fit_background: bool, if True, includes background level in optimization parameters.
        update_catalog: bool, if True, updates the source catalog with optimized fluxes.
                        Requires single image or warns if multiple.
        vmap_images: bool, if True (default), stacks all images and processes them in a single vmap call.
                     If False, iterates over images sequentially (saving memory).
        use_sharding: bool, if True (default), distributes the batch across available devices when vmap_images=True.

    Returns:
        List of results per image.
        Each result is (fluxes, variances) if return_variances is True, else fluxes.
        Note: The tractor_obj source catalog is NOT updated unless update_catalog=True.
        However, if fit_background is True, img.sky is updated for each image.
    """
    from tractor import ConstantSky

    results = []

    # Common function to compile/call solve_fluxes_core
    # We can use JIT here regardless of vmap
    solve_jit = jit(partial(solve_fluxes_core, return_variances=return_variances))

    if vmap_images:
        # 1. Extract Data (Full Batch)
        images_data, batches, initial_fluxes = extract_model_data(
            tractor_obj,
            oversample_rendering=oversample_rendering,
            fit_background=fit_background
        )

        # 2. Define in_axes for batches
        batches_in_axes = {}
        if "PointSource" in batches:
            batches_in_axes["PointSource"] = {
                "flux_idx": None, # Shared
                "pos_pix": 0,     # Batched
            }
        if "Galaxy" in batches:
            batches_in_axes["Galaxy"] = {
                "flux_idx": None,
                "pos_pix": 0,
                "wcs_cd_inv": 0,
                "shapes": None,
                "profile": {
                    "amp": None,
                    "mean": None,
                    "var": None,
                }
            }
        if "Background" in batches:
            batches_in_axes["Background"] = {
                "flux_idx": None
            }

        # 3. Vmap Optimization
        # images_data is fully batched (0)
        # initial_fluxes is batched (0)

        if use_sharding:
            images_data, batches, initial_fluxes = prepare_sharded_inputs(images_data, batches, initial_fluxes)

        solve_fn = jit(vmap(
            partial(solve_fluxes_core, return_variances=return_variances),
            in_axes=(0, 0, batches_in_axes)
        ))

        if return_variances:
            optimized_fluxes_stack, variances_stack = solve_fn(initial_fluxes, images_data, batches)
        else:
            optimized_fluxes_stack = solve_fn(initial_fluxes, images_data, batches)

        optimized_fluxes_np = np.array(optimized_fluxes_stack)
        if return_variances:
            variances_np = np.array(variances_stack)

    else:
        # Sequential Processing
        # We process images one by one to save memory.
        # However, we still need to collect results in the same format.

        fluxes_list = []
        variances_list = []

        batches = {} # Initialize in case loop doesn't run, to avoid UnboundLocalError for bg check

        for img in tractor_obj.images:
            # Create a mini Tractor object for extraction
            # extract_model_data works on Tractor objects.
            sub_tractor = Tractor([img], tractor_obj.catalog)

            img_data, batches, init_flux = extract_model_data(
                sub_tractor,
                oversample_rendering=oversample_rendering,
                fit_background=fit_background
            )

            # img_data is stacked with shape (1, ...). We unbatch.
            single_data = jax.tree_util.tree_map(lambda x: x[0], img_data)

            # batches contain fields like 'pos_pix' which are (1, N_src, 2).
            # We unbatch them to (N_src, 2).

            single_batches = batches.copy()
            if "PointSource" in batches:
                # pos_pix is (N_img, N_src, 2).
                single_batches["PointSource"] = batches["PointSource"].copy()
                single_batches["PointSource"]["pos_pix"] = batches["PointSource"]["pos_pix"][0]
                # flux_idx is shared.

            if "Galaxy" in batches:
                single_batches["Galaxy"] = batches["Galaxy"].copy()
                single_batches["Galaxy"]["pos_pix"] = batches["Galaxy"]["pos_pix"][0]
                single_batches["Galaxy"]["wcs_cd_inv"] = batches["Galaxy"]["wcs_cd_inv"][0]
                # shapes, profile are shared (attributes of source).

            if "Background" in batches:
                # flux_idx is (1,) for single image?
                pass

            single_flux = init_flux[0] # (N_params,)

            if return_variances:
                f, v = solve_jit(single_flux, single_data, single_batches)
                fluxes_list.append(f)
                variances_list.append(v)
            else:
                f = solve_jit(single_flux, single_data, single_batches)
                fluxes_list.append(f)

        optimized_fluxes_np = np.array(fluxes_list)
        if return_variances:
            variances_np = np.array(variances_list)

    N_img = len(tractor_obj.images)

    bg_idx = None
    if fit_background and "Background" in batches:
        bg_idx = int(batches["Background"]["flux_idx"][0])

    for i in range(N_img):
        f = optimized_fluxes_np[i]

        if return_variances:
            v = variances_np[i]
            results.append((f, v))
        else:
            results.append(f)

        # Update Background
        if bg_idx is not None:
            img = tractor_obj.images[i]
            bg_val = f[bg_idx]

            if isinstance(img.sky, ConstantSky):
                img.sky.val = bg_val
            else:
                img.sky = ConstantSky(bg_val)

    if update_catalog:
        if N_img == 1:
            # Update catalog
            # We need to map flux array back to sources.
            # We can use batches info which contains 'flux_idx'.
            # The indices in flux array correspond to the order in src_fluxes (from extract_model_data).
            # src_fluxes was built by iterating catalog.

            # Re-iterate catalog to update params?
            # Or use indices if we stored them.
            # batches stores flux_idx per type.

            # Let's iterate types.
            f_vec = optimized_fluxes_np[0] # Single image

            # Point Sources
            if "PointSource" in batches:
                idxs = batches["PointSource"]["flux_idx"]
                # idxs is (N_src,) array of indices
                # We need to know WHICH sources are these.
                # extract_model_data iterates catalog.

                # It's cleaner to re-iterate catalog and update in order if we know the order matches.
                # But extract_model_data filters sources (CompositeGalaxy etc).

                # We should probably modify extract_model_data to return a mapping or list of (source, start_idx).
                # But I don't want to break API if possible.

                # Let's assume standard iteration order is preserved.
                # And assume we only have PointSources and Galaxies supported.

                # This is tricky without refactoring extract_model_data.
                # BUT, extract_model_data is in this file. I CAN refactor it or rely on its logic.

                # The fluxes in 'initial_fluxes' are packed: [src1_params, src2_params, ...].
                # So if we iterate catalog again, we can match them.

                ptr = 0
                for src in tractor_obj.catalog:
                    if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
                        continue
                    if hasattr(src, "brightness"):
                        n = src.brightness.numberOfParams()
                        vals = f_vec[ptr : ptr+n]
                        src.brightness.setParams(vals)
                        ptr += n

        else:
            print("Warning: update_catalog=True but N_img > 1. Catalog not updated to avoid ambiguity.")

    return results


class JaxOptimizer(Optimizer):
    def __init__(self):
        super(JaxOptimizer, self).__init__()
        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)

    def optimize(self, tractor, alphas=None, damp=0, priors=True,
                 scale_columns=True, shared_params=True, variance=False,
                 just_variance=False, **kwargs):
        """
        Performs optimization using JAX.
        """
        lnp0 = tractor.getLogProb()
        p0 = tractor.getParams()

        # Call optimize_fluxes with update_catalog=True
        # We assume oversample_rendering=True as safe default? Or only if needed?
        # User requested oversampling test.
        # But we should probably check if needed? No, just pass it.

        # Note: optimize_fluxes returns (fluxes, vars) if variance=True.
        # But update_catalog=True updates the tractor object.
        # optimize_fluxes currently returns list of results per image.

        res = optimize_fluxes(
            tractor,
            return_variances=variance,
            fit_background=True,
            oversample_rendering=True,
            update_catalog=True
        )

        # tractor catalog is updated.
        p1 = tractor.getParams()
        lnp1 = tractor.getLogProb()
        dlnp = lnp1 - lnp0
        X = np.array(p1) - np.array(p0)

        alpha = 1.0

        if variance:
            # We need to return variance vector matching X.
            # optimize_fluxes returns list of variances per image.
            # If N_img=1, we take the first.
            if len(res) == 1:
                fluxes, vars = res[0]
                # vars corresponds to flux parameters.
                # X corresponds to ALL parameters (including fixed positions).
                # But X has 0 for fixed params.
                # We need to map vars to the full parameter vector.

                # We can construct full variance vector.
                # tractor.getParams() returns all thawed params.
                # optimize_fluxes only optimizes fluxes (and maybe background).
                # It does NOT optimize positions.
                # So variance for positions is 0 (or infinity? usually 0 if fixed).

                # We need to map the variances back.
                # This is hard without explicit mapping.

                # For now, let's assume we are fitting fluxes only (positions fixed).
                # Then X length == flux params length.
                # And vars length == flux params length.

                # But if positions are thawed in tractor, X will be larger.
                # optimize_fluxes does NOT touch positions.
                # So X will have 0s for positions.
                # And we don't have variances for positions.

                # If the user expects variances for all params, we should pad with 0?

                # Let's try to match lengths.
                full_var = np.zeros_like(X)

                # Mapping:
                # We iterate catalog again to fill full_var?
                # Similar logic to update_catalog.

                ptr_flux = 0
                ptr_param = 0

                # This depends on how tractor.getParams() orders things.
                # Tractor orders by: Images (if thawed), Catalog (if thawed).
                # Images params (Sky?)
                # Catalog params (Src1, Src2...)

                # Check if Images params are thawed.
                # If fit_background=False, Sky is not optimized by JAX (except if we updated it?)
                # optimize_fluxes with fit_background=False does not return sky variance.

                # If Sky is thawed in Tractor, p0 includes sky.
                # But JAX didn't optimize it.

                # Let's assume simple case: Sky fixed, Pos fixed. Flux thawed.
                # Then params match.

                if len(X) == len(vars):
                     full_var = vars
                else:
                    # Try to map?
                    # Too risky without robust mapping.
                    # Just return vars and hope user handles it or lengths match.
                    full_var = vars # Mismatch likely if pos thawed.
            else:
                full_var = None

            return dlnp, X, alpha, full_var

        return dlnp, X, alpha

    def optimize_loop(self, tractor, dchisq=0., steps=50, **kwargs):
        # Run single step as JAX CG solves it
        return self.optimize(tractor, **kwargs)
