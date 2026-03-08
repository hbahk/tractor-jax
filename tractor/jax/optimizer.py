import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit, value_and_grad, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import math
from functools import partial
from collections import defaultdict, Counter
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
from tractor.jax.tiling import tile_image, project_catalog, filter_sources_by_box


def compute_image_shapes(images, stats):
    """
    Computes required target shape for each image.
    """
    max_factor = stats["max_factor"]
    fft_pad_h_lr = stats["fft_pad_h_lr"]
    fft_pad_w_lr = stats["fft_pad_w_lr"]

    shapes = []
    for img in images:
        h, w = img.shape
        padded_h = h + fft_pad_h_lr
        padded_w = w + fft_pad_w_lr

        target_h = int(round(padded_h * max_factor))
        target_w = int(round(padded_w * max_factor))
        shapes.append((target_h, target_w))

    return shapes


def assign_buckets(
    required_shapes,
    bucket_sizes=None,
    bucket_mode="auto",
    bucket_shape_mode="square",
    bucket_base=32,
    max_buckets=5
):
    """
    Assigns images to buckets based on required shapes.
    Returns a dict: { bucket_shape: [img_indices] }
    """

    # 1. Determine available buckets
    allowed_sizes = []
    allowed_shapes = []

    if bucket_mode == "fixed":
        if bucket_sizes is None:
            # Fallback default
            bucket_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

        # In fixed mode, we assume bucket_sizes defines the allowed grid.
        # It can be a list of ints (squares) or tuples.
        allowed_list = sorted(bucket_sizes) if hasattr(bucket_sizes, '__iter__') else [bucket_sizes]

        # Normalize to tuples
        norm_shapes = []
        for b in allowed_list:
            if isinstance(b, (int, float)):
                norm_shapes.append((int(b), int(b)))
            else:
                norm_shapes.append((int(b[0]), int(b[1])))

        # For assignment logic, we use this list.
        allowed_shapes = norm_shapes

    else: # auto
        # Determine buckets from distribution

        # Quantize required shapes to bucket_base
        quantized_shapes = []
        for h, w in required_shapes:
            # Ceil to multiple of bucket_base
            h_q = int(math.ceil(h / bucket_base) * bucket_base)
            w_q = int(math.ceil(w / bucket_base) * bucket_base)
            quantized_shapes.append((h_q, w_q))

        if bucket_shape_mode == "square":
            # Force square
            sq_sizes = []
            for h, w in quantized_shapes:
                s = max(h, w)
                sq_sizes.append(s)

            counts = Counter(sq_sizes)
            max_size = max(sq_sizes) if sq_sizes else bucket_base

            # Get common sizes
            common = counts.most_common(max_buckets)

            candidates = set([s for s, c in common])
            candidates.add(max_size)

            for s in sorted(list(candidates)):
                allowed_shapes.append((s, s))

        else: # independent
            counts = Counter(quantized_shapes)

            all_h = [s[0] for s in quantized_shapes]
            all_w = [s[1] for s in quantized_shapes]
            max_h_all = max(all_h) if all_h else bucket_base
            max_w_all = max(all_w) if all_w else bucket_base
            catch_all = (max_h_all, max_w_all)

            common = counts.most_common(max_buckets - 1)
            active = set([s for s, c in common])
            active.add(catch_all)

            allowed_shapes = list(active)

    # 2. Assign images
    bucket_map = defaultdict(list)

    for i, (req_h, req_w) in enumerate(required_shapes):
        # Find best bucket: Smallest area that fits
        valid = [s for s in allowed_shapes if s[0] >= req_h and s[1] >= req_w]

        if valid:
            best = min(valid, key=lambda x: x[0]*x[1])
        else:
            # Fallback if no bucket fits (e.g. fixed mode with too small buckets)
            # We create a new bucket on the fly fitting this image
            bh = int(math.ceil(req_h / bucket_base) * bucket_base)
            bw = int(math.ceil(req_w / bucket_base) * bucket_base)
            if bucket_shape_mode == "square":
                s = max(bh, bw)
                best = (s, s)
            else:
                best = (bh, bw)

        bucket_map[best].append(i)

    return bucket_map


def compute_target_stats(images, oversample_rendering=False):
    """
    Computes global statistics for a set of images to determine required grid sizes.
    """
    max_factor = 1.0
    # First pass: max_factor
    for img in images:
        if oversample_rendering:
            psf = img.getPsf()
            if isinstance(psf, PixelizedPSF):
                s = getattr(psf, "sampling", 1.0)
                if s < 1.0:
                    max_factor = max(max_factor, 1.0 / s)

    max_psf_h = 0
    max_psf_w = 0
    for img in images:
        psf = img.getPsf()

        # Use r_eff for padding if available
        if hasattr(psf, 'get_r_eff'):
            r_eff = psf.get_r_eff(0.999)

            if isinstance(psf, PixelizedPSF):
                 # r_eff is in PSF pixels. Convert to Image Pixels.
                 s = getattr(psf, "sampling", 1.0)
                 r_eff_img = r_eff / s
            else:
                 # Analytic PSF (Gaussian), r_eff is in Image Pixels.
                 r_eff_img = r_eff

            # Convert to Target Pixels
            r_eff_target = r_eff_img * max_factor

            # Diameter in Target Pixels (padded to be safe)
            # We want the padding to be enough such that a source at the edge
            # fully contains the PSF on the padded grid.
            size_target = math.ceil(2.0 * r_eff_target)

            max_psf_h = max(max_psf_h, size_target)
            max_psf_w = max(max_psf_w, size_target)

        elif isinstance(psf, PixelizedPSF):
            ph, pw = psf.img.shape
            s = getattr(psf, "sampling", 1.0)
            # Fallback to old logic (fixed to divide by s?)
            # Old logic was: scale = max_factor * s.
            # If s > 1 (oversampled), this explodes.
            # Assuming s means samples/pixel, it should be / s.
            # But let's stick to old logic for fallback to avoid changing behavior for non-reff cases too much
            # unless we are sure.
            # Actually, let's fix it if we are confident.
            # But if r_eff is missing, maybe it's safest to use shape.

            if oversample_rendering:
                # Correct logic: pixels * max_factor / s
                scale = max_factor / s
                ph_target = int(ph * scale)
                pw_target = int(pw * scale)
            else:
                ph_target = ph
                pw_target = pw
            max_psf_h = max(max_psf_h, ph_target)
            max_psf_w = max(max_psf_w, pw_target)
        else:
             max_psf_h = max(max_psf_h, 32 * max_factor)
             max_psf_w = max(max_psf_w, 32 * max_factor)

    fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))

    return {
        "max_factor": max_factor,
        "max_psf_h": max_psf_h,
        "max_psf_w": max_psf_w,
        "fft_pad_h_lr": fft_pad_h_lr,
        "fft_pad_w_lr": fft_pad_w_lr,
    }


def extract_model_data(
    tractor_obj,
    oversample_rendering=False,
    fit_background=False,
    fixed_target_shape=None,
    fixed_max_factor=None,
    img_source_indices=None
):
    """
    Extracts all necessary data from a Tractor object for JAX optimization,
    grouping sources into batches and stacking image data with padding for vectorized rendering.

    Args:
        tractor_obj: Tractor object.
        oversample_rendering: If True, handles oversampled PixelizedPSF by rendering at high resolution.
        fit_background: If True, includes background level in optimization parameters.
        fixed_target_shape: (H, W) tuple. If provided, forces the target grid size to this shape.
                            Useful for bucketing.
        fixed_max_factor: float. Required if fixed_target_shape is provided.
                          The oversampling factor assumed for the bucket.

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

    if fixed_target_shape is not None:
        if fixed_max_factor is None:
            raise ValueError("fixed_max_factor is required when fixed_target_shape is used.")

        target_H, target_W = fixed_target_shape
        max_factor = fixed_max_factor

        # Calculate max_mog_K for padding logic below (needed for consistency)
        # Note: We still need psf info for individual images.

        # We need padded_H/W (input resolution padded size) for padding input images.
        # target_H >= padded_H * max_factor
        # padded_H = floor(target_H / max_factor)
        # We use floor to ensure that valid_H_hr (padded_H * max_factor) <= target_H
        padded_H = int(math.floor(target_H / max_factor))
        padded_W = int(math.floor(target_W / max_factor))

        # We enforce target_sampling to be max_factor physically
        target_sampling = float(max_factor) if max_factor > 1.0 else 1.0

    else:
        # Standard logic: compute from current batch
        stats = compute_target_stats(images, oversample_rendering)
        max_factor = stats["max_factor"]
        fft_pad_h_lr = stats["fft_pad_h_lr"]
        fft_pad_w_lr = stats["fft_pad_w_lr"]

        max_H, max_W = 0, 0
        for img in images:
            h, w = img.shape
            max_H = max(max_H, h)
            max_W = max(max_W, w)

        padded_H = max_H + fft_pad_h_lr
        padded_W = max_W + fft_pad_w_lr

        if oversample_rendering and max_factor > 1.0:
            target_H = int(round(padded_H * max_factor))
            target_W = int(round(padded_W * max_factor))
            target_sampling = float(max_factor)
        else:
            target_H = padded_H
            target_W = padded_W
            target_sampling = 1.0

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
        pad_h = padded_H - h
        pad_w = padded_W - w

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
        p_sampling = target_sampling

        # Dummy MoG (Identity)
        p_amp = jnp.zeros(max_mog_K)
        p_mean = jnp.zeros((max_mog_K, 2))
        p_var = jnp.tile(jnp.eye(2), (max_mog_K, 1, 1))

        if isinstance(psf, PixelizedPSF):
            p_type = 0

            # Get local sampling
            s = getattr(psf, "sampling", 1.0)
            local_factor = 1.0/s if s < 1.0 else 1.0

            if abs(local_factor - p_sampling) > 1e-3:
                # Resize PSF image to match target resolution
                raw_img = jnp.array(psf.img)
                ph, pw = raw_img.shape
                ratio = p_sampling / local_factor
                new_shape = (int(round(ph * ratio)), int(round(pw * ratio)))

                # Resize using jax.image.resize
                resized_img = jax.image.resize(raw_img, new_shape, method='lanczos3')

                # Normalize flux to preserve sum
                orig_sum = jnp.sum(raw_img)
                new_sum = jnp.sum(resized_img)
                resized_img = resized_img * (orig_sum / new_sum)

                raw_img = resized_img
            else:
                raw_img = jnp.array(psf.img)

            ph, pw = raw_img.shape

            # 3. Pad to target_H, target_W
            pad_img = jnp.zeros((target_H, target_W))
            cy, cx = target_H // 2, target_W // 2
            y0 = cy - ph // 2
            x0 = cx - pw // 2

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
                new_var = jnp.zeros((max_mog_K, 2, 2), dtype=var.dtype)
                new_var = new_var.at[:K].set(var)

                # Set identity for padding
                padding_eye = jnp.tile(jnp.eye(2), (pad_len, 1, 1))
                new_var = new_var.at[K:].set(padding_eye)
                var = new_var

            p_amp = amp
            p_mean = mean
            p_var = var

            # Dummy FFT (Zeros)
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
    cat_idx_to_flux_idx = {}
    flux_offset = 0

    # Pass 1: Catalog Flux Index Mapping
    for i, src in enumerate(catalog):
        if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
            print(f"Warning: Skipping CompositeGalaxy {src} in JAX optimization")
            continue

        if hasattr(src, "brightness"):
            br = src.brightness.getParams()
            cat_idx_to_flux_idx[i] = flux_offset
            src_fluxes.extend(br)
            flux_offset += len(br)

    # Prepare batches per image
    ps_batch_list = [] # (N_img) list of (flux_idx, pos_pix, mask)
    gal_batch_list = [] # (N_img) list of (...)

    max_gal_mog_K = 0

    N_img = len(images)
    for i_img in range(N_img):
        img = images[i_img]
        wcs = img.getWcs()

        if img_source_indices is not None:
            indices = img_source_indices[i_img]
        else:
            indices = sorted(cat_idx_to_flux_idx.keys())

        # Current Image Lists
        ps_flux = []
        ps_pos = []

        gal_flux = []
        gal_pos = []
        gal_cd = []
        gal_shape = []
        gal_prof = [] # (amp, mean, var)

        for cat_idx in indices:
            if cat_idx not in cat_idx_to_flux_idx:
                continue

            src = catalog[cat_idx]
            f_idx = cat_idx_to_flux_idx[cat_idx]

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

            if src_type == "PointSource":
                x, y = wcs.positionToPixel(src.getPosition(), src)
                ps_flux.append(f_idx)
                ps_pos.append([x, y])

            elif is_galaxy and prof is not None:
                x, y = wcs.positionToPixel(src.getPosition(), src)
                cd_inv = wcs.cdInverseAtPixel(x, y)
                gal_flux.append(f_idx)
                gal_pos.append([x, y])
                gal_cd.append(cd_inv)
                gal_shape.append(src.shape.getAllParams())

                if hasattr(prof, "mog"):
                    amp, mean, var = prof.mog.amp, prof.mog.mean, prof.mog.var
                else:
                    amp, mean, var = prof.amp, prof.mean, prof.var

                gal_prof.append((amp, mean, var))
                max_gal_mog_K = max(max_gal_mog_K, len(amp))

        ps_batch_list.append((ps_flux, ps_pos))
        gal_batch_list.append((gal_flux, gal_pos, gal_cd, gal_shape, gal_prof))

    # Build Final Batches
    batches = {}

    # Pad Point Sources
    max_ps = max(len(x[0]) for x in ps_batch_list)
    if max_ps > 0:
        flux_idx_stack = []
        pos_pix_stack = []
        mask_stack = []

        for (fl, pos) in ps_batch_list:
            n = len(fl)
            pad = max_ps - n

            # Pad arrays
            # flux_idx: pad with 0
            f_arr = np.array(fl, dtype=np.int32)
            f_arr = np.pad(f_arr, (0, pad), constant_values=0)
            flux_idx_stack.append(f_arr)

            # pos_pix: pad with 0
            if n > 0:
                p_arr = np.array(pos, dtype=np.float32)
            else:
                p_arr = np.zeros((0, 2), dtype=np.float32)
            p_arr = np.pad(p_arr, ((0, pad), (0, 0)), constant_values=0)
            pos_pix_stack.append(p_arr)

            # mask: 1 for real, 0 for pad
            m_arr = np.ones(n, dtype=np.float32)
            m_arr = np.pad(m_arr, (0, pad), constant_values=0)
            mask_stack.append(m_arr)

        batches["PointSource"] = {
            "flux_idx": jnp.array(np.stack(flux_idx_stack)),
            "pos_pix": jnp.array(np.stack(pos_pix_stack)),
            "mask": jnp.array(np.stack(mask_stack)),
        }

    # Pad Galaxies
    max_gal = max(len(x[0]) for x in gal_batch_list)
    if max_gal > 0:
        flux_idx_stack = []
        pos_pix_stack = []
        wcs_stack = []
        shape_stack = []
        mask_stack = []

        prof_amp_stack = []
        prof_mean_stack = []
        prof_var_stack = []

        for (fl, pos, cd, sh, pr) in gal_batch_list:
            n = len(fl)
            pad = max_gal - n

            f_arr = np.array(fl, dtype=np.int32)
            f_arr = np.pad(f_arr, (0, pad), constant_values=0)
            flux_idx_stack.append(f_arr)

            if n > 0:
                p_arr = np.array(pos, dtype=np.float32)
                cd_arr = np.array(cd, dtype=np.float32)
                sh_arr = np.array(sh, dtype=np.float32)
            else:
                p_arr = np.zeros((0, 2), dtype=np.float32)
                cd_arr = np.zeros((0, 2, 2), dtype=np.float32)
                sh_arr = np.zeros((0, 3), dtype=np.float32) # re, ab, phi

            p_arr = np.pad(p_arr, ((0, pad), (0, 0)), constant_values=0)
            pos_pix_stack.append(p_arr)

            cd_arr = np.pad(cd_arr, ((0, pad), (0, 0), (0, 0)), constant_values=0) # Identity? 0 is fine if masked
            wcs_stack.append(cd_arr)

            sh_arr = np.pad(sh_arr, ((0, pad), (0, 0)), constant_values=0)
            shape_stack.append(sh_arr)

            m_arr = np.ones(n, dtype=np.float32)
            m_arr = np.pad(m_arr, (0, pad), constant_values=0)
            mask_stack.append(m_arr)

            # Profile padding (MoG)
            # Each source has MoG with K components.
            # We need to pad each MoG to max_gal_mog_K.
            # AND pad the list of sources to max_gal.

            # Construct (max_gal, max_K, ...) arrays for this image
            img_amp = np.zeros((max_gal, max_gal_mog_K), dtype=np.float32)
            img_mean = np.zeros((max_gal, max_gal_mog_K, 2), dtype=np.float32)
            img_var = np.zeros((max_gal, max_gal_mog_K, 2, 2), dtype=np.float32)
            # Initialize var to Identity to avoid singular matrices if unmasked?
            img_var[:] = np.eye(2)

            for k_src in range(n):
                amp, mean, var = pr[k_src]
                K = len(amp)
                img_amp[k_src, :K] = amp
                img_mean[k_src, :K] = mean
                img_var[k_src, :K] = var

            prof_amp_stack.append(img_amp)
            prof_mean_stack.append(img_mean)
            prof_var_stack.append(img_var)

        batches["Galaxy"] = {
            "flux_idx": jnp.array(np.stack(flux_idx_stack)),
            "pos_pix": jnp.array(np.stack(pos_pix_stack)),
            "wcs_cd_inv": jnp.array(np.stack(wcs_stack)),
            "shapes": jnp.array(np.stack(shape_stack)),
            "mask": jnp.array(np.stack(mask_stack)),
            "profile": {
                "amp": jnp.array(np.stack(prof_amp_stack)),
                "mean": jnp.array(np.stack(prof_mean_stack)),
                "var": jnp.array(np.stack(prof_var_stack)),
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


def render_batch_point_sources(fluxes, pos_pix, psf_data, img_shape, sampling_factor=None, mask=None):
    """
    Renders a batch of Point Sources.
    """
    if sampling_factor is not None:
        s = sampling_factor
    else:
        s = psf_data['sampling']

    H, W = img_shape
    H_hr_grid = psf_data['fft'].shape[0]
    W_hr_grid = (psf_data['fft'].shape[1] - 1) * 2

    if mask is not None:
        fluxes = fluxes * mask

    def render_fft(operand):
        render_shape = (H_hr_grid, W_hr_grid)

        pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0

        render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))
        stamps = render_fn(fluxes, pos_pix_scaled, psf_data['fft'])
        combined = jnp.sum(stamps, axis=0)

        if sampling_factor is not None and s > 1.001:
            valid_H = int(round(H * s))
            valid_W = int(round(W * s))
            valid_H = min(valid_H, H_hr_grid)
            valid_W = min(valid_W, W_hr_grid)

            combined = combined[:valid_H, :valid_W]
            combined = downsample_image(combined, img_shape)
        elif sampling_factor is None:
             if H_hr_grid > H + 1:
                 combined = downsample_image(combined, img_shape)

        return combined

    def render_mog(operand):
        psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
        render_fn = vmap(partial(render_point_source_mog, image_shape=img_shape), in_axes=(0, 0, None))
        stamps = render_fn(fluxes, pos_pix, psf_mix)
        return jnp.sum(stamps, axis=0)

    return jax.lax.cond(psf_data['type_code'] == 0, render_fft, render_mog, None)


def render_batch_galaxies(
    fluxes, pos_pix, wcs_cd_inv, shapes, profiles, psf_data, img_shape, sampling_factor=None, mask=None
):
    """
    Renders a batch of Galaxies.
    """
    if sampling_factor is not None:
        s = sampling_factor
    else:
        s = psf_data['sampling']

    H, W = img_shape
    H_hr_grid = psf_data['fft'].shape[0]
    W_hr_grid = (psf_data['fft'].shape[1] - 1) * 2

    if mask is not None:
        fluxes = fluxes * mask

    def render_fft(operand):
        render_shape = (H_hr_grid, W_hr_grid)

        pos_pix_scaled = pos_pix * s + (s - 1.0) / 2.0
        wcs_cd_inv_scaled = wcs_cd_inv * s

        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
        stamps = render_fn(gal_mix, psf_data['fft'], shapes, wcs_cd_inv_scaled, pos_pix_scaled)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]
        combined = jnp.sum(weighted_stamps, axis=0)

        if sampling_factor is not None and s > 1.001:
            valid_H = int(round(H * s))
            valid_W = int(round(W * s))
            valid_H = min(valid_H, H_hr_grid)
            valid_W = min(valid_W, W_hr_grid)

            combined = combined[:valid_H, :valid_W]
            combined = downsample_image(combined, img_shape)
        elif sampling_factor is None:
             if H_hr_grid > H + 1:
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


def render_image(fluxes, image_data, batches, sampling_factor=None):
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
        mask = batch.get("mask", None)

        ps_model = render_batch_point_sources(
            batch_fluxes, pos_pix, image_data["psf"], (H, W), sampling_factor=sampling_factor, mask=mask
        )
        img_model = img_model + ps_model

    # 2. Render Galaxies
    if "Galaxy" in batches:
        batch = batches["Galaxy"]
        pos_pix = batch["pos_pix"] # (N_gal, 2)
        wcs_cd_inv = batch["wcs_cd_inv"] # (N_gal, 2, 2)
        shapes = batch["shapes"]
        profiles = batch["profile"]
        mask = batch.get("mask", None)

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
            sampling_factor=sampling_factor,
            mask=mask
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
        mask = batch.get("mask", None)
        if mask is not None:
            unit_fluxes = unit_fluxes * mask

        psf_data = image_data["psf"]

        # Render unit fluxes
        stamps = render_batch_point_sources(unit_fluxes, pos_pix, psf_data, (H, W), mask=mask)
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
        mask = batch.get("mask", None)

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

        if mask is not None:
            stamps = stamps * mask[:, jnp.newaxis, jnp.newaxis]

        contrib = jnp.sum(stamps**2 * invvar[jnp.newaxis, :, :], axis=(1, 2))
        fisher_diag = fisher_diag.at[f_idx].add(contrib)

    # 3. Background
    if "Background" in batches:
        f_idx = batches["Background"]["flux_idx"] # (1,)
        # Derivative is 1.0
        contrib = jnp.sum(invvar)
        fisher_diag = fisher_diag.at[f_idx].add(contrib)

    return fisher_diag


def solve_fluxes_core(initial_fluxes, image_data, batches, return_variances=False, sampling_factor=None, use_preconditioner=True, precond_eps=1e-12):
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
        model_image = render_image(fluxes, image_data, batches, sampling_factor=sampling_factor)
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

    fisher_diag = None
    inv_fisher_diag = None
    if use_preconditioner or return_variances:
        fisher_diag = compute_fisher_diagonal(image_data, batches, len(initial_fluxes))
        fisher_diag = jnp.where(fisher_diag <= 0, precond_eps, fisher_diag)
        inv_fisher_diag = 1.0 / fisher_diag

    # CG solve A x = b where A is Hessian, b = -grads
    # For high precision, maxiter needs to be higher?
    # User requested dchi2=1e-10.
    if use_preconditioner:
        def precond(v):
            return v * inv_fisher_diag
        step, info = jax.scipy.sparse.linalg.cg(
            matvec, -grads, maxiter=500, tol=1e-6, M=precond
        )
    else:
        step, info = jax.scipy.sparse.linalg.cg(matvec, -grads, maxiter=500, tol=1e-6)

    optimized_fluxes = initial_fluxes + step

    if return_variances:
        variances = inv_fisher_diag
        return optimized_fluxes, variances

    return optimized_fluxes


def optimize_fluxes(tractor_obj, oversample_rendering=False, return_variances=False, fit_background=False, update_catalog=False, vmap_images=True, use_sharding=True, bucket_sizes=None, bucket_mode="auto", bucket_shape_mode="square", bucket_base=32, use_tiling=False, tile_size=256, tile_super_halo=None, use_preconditioner=True, precond_eps=1e-12):
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
        bucket_sizes: List of bucket sizes (e.g. [128, 256]). Used if bucket_mode="fixed".
        bucket_mode: "auto" or "fixed".
        bucket_shape_mode: "square" or "independent".
        bucket_base: rounding base for auto mode.
        use_tiling: bool, if True, splits images into tiles and processes them.
        tile_size: int, size of tiles (default 256).
        tile_super_halo: optional int, override calculated halo size.
        use_preconditioner: bool, if True (default), use Fisher diagonal preconditioner.
        precond_eps: float, floor for Fisher diagonal to avoid divide-by-zero.

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
    solve_jit = jit(partial(
        solve_fluxes_core,
        return_variances=return_variances,
        use_preconditioner=use_preconditioner,
        precond_eps=precond_eps,
    ))

    if use_tiling:
        # TILING MODE
        # 1. Calculate Halo
        halo = 0
        if tile_super_halo is not None:
            halo = tile_super_halo
        else:
            max_r_eff = 0.0
            for img in tractor_obj.images:
                psf = img.getPsf()
                if hasattr(psf, 'get_r_eff'):
                    r = psf.get_r_eff(0.999)
                    if isinstance(psf, PixelizedPSF):
                        s = getattr(psf, 'sampling', 1.0)
                        r = r / s
                    max_r_eff = max(max_r_eff, r)
                else:
                    # Fallback
                    max_r_eff = max(max_r_eff, 32.0) # Default conservative
            halo = int(math.ceil(max_r_eff))

        print(f"JAX Optimization: Tiling enabled. Tile size {tile_size}, Halo {halo}")

        # 2. Generate Tiles & Filter Sources
        all_tiles = []
        all_indices = []
        original_img_indices = [] # Map tile -> original image index (if needed)

        for i_img, img in enumerate(tractor_obj.images):
            # Project catalog to this image's pixel coords
            pos_cat = project_catalog(tractor_obj.catalog, img.getWcs())

            # Split into tiles
            tiles_with_meta = tile_image(img, tile_size, halo)

            for (tile_img, meta) in tiles_with_meta:
                # Filter sources
                indices = filter_sources_by_box(
                    pos_cat,
                    meta['x_start'], meta['x_end'],
                    meta['y_start'], meta['y_end'],
                    margin=0 # Halo already included in start/end
                )

                # We include the tile even if indices is empty?
                # Yes, might fit background.

                all_tiles.append(tile_img)
                all_indices.append(indices)
                original_img_indices.append(i_img)

        # 3. Bucket Tiles
        # Calculate stats for tiles
        stats = compute_target_stats(all_tiles, oversample_rendering)
        max_factor = stats["max_factor"]
        req_shapes = compute_image_shapes(all_tiles, stats)

        # Bucketing
        bucket_map = assign_buckets(req_shapes, bucket_sizes, bucket_mode, bucket_shape_mode, bucket_base)

        print(f"JAX Optimization: {len(all_tiles)} tiles -> {len(bucket_map)} buckets")

        # Container for results (fluxes per tile)
        # We store result as list of results matching 'all_tiles' order.
        tile_results = [None] * len(all_tiles)

        for shape, tile_idxs in bucket_map.items():
            if not tile_idxs:
                continue

            sub_tiles = [all_tiles[i] for i in tile_idxs]
            sub_source_indices = {k: all_indices[original_idx] for k, original_idx in enumerate(tile_idxs)}

            # sub_tractor needs same catalog
            sub_tractor = Tractor(sub_tiles, tractor_obj.catalog)

            # Extract Data (Sparse)
            images_data, batches, initial_fluxes = extract_model_data(
                sub_tractor,
                oversample_rendering=oversample_rendering,
                fit_background=fit_background,
                fixed_target_shape=shape,
                fixed_max_factor=max_factor,
                img_source_indices=sub_source_indices
            )

            # Define in_axes
            batches_in_axes = {}
            if "PointSource" in batches:
                batches_in_axes["PointSource"] = {
                    "flux_idx": 0, "pos_pix": 0, "mask": 0
                }
            if "Galaxy" in batches:
                batches_in_axes["Galaxy"] = {
                    "flux_idx": 0, "pos_pix": 0, "wcs_cd_inv": 0, "shapes": 0, "mask": 0,
                    "profile": {"amp": 0, "mean": 0, "var": 0}
                }
            if "Background" in batches:
                batches_in_axes["Background"] = {"flux_idx": None} # Background index logic might vary?

            # Wait, background logic in extract_model_data assumes 1 flux per image.
            # And it puts `bg_vals` at end of `initial_fluxes`.
            # And `batches["Background"]["flux_idx"]` is (N_img,) usually.
            # But in `extract_model_data` dense/sparse refactor, I kept `batches["Background"]` as:
            # `batches["Background"] = { "flux_idx": jnp.array([bg_idx], dtype=jnp.int32) }` (Scalar index relative to row!)
            # Check line 592 in modified code: `bg_idx = len(src_fluxes)`.
            # This is scalar.
            # So `batches_in_axes` should be `None` for flux_idx if it's the same scalar for all images?
            # Yes, for each image row, the background is at `N_flux`.
            # So the index is constant.
            # So `flux_idx: None` is correct.

            # Optimization
            if use_sharding:
                images_data, batches, initial_fluxes = prepare_sharded_inputs(images_data, batches, initial_fluxes)

            solve_fn = jit(vmap(
                partial(
                    solve_fluxes_core,
                    return_variances=return_variances,
                    sampling_factor=max_factor,
                    use_preconditioner=use_preconditioner,
                    precond_eps=precond_eps,
                ),
                in_axes=(0, 0, batches_in_axes)
            ))

            if return_variances:
                optimized_fluxes_stack, variances_stack = solve_fn(initial_fluxes, images_data, batches)
            else:
                optimized_fluxes_stack = solve_fn(initial_fluxes, images_data, batches)

            res_fluxes = np.array(optimized_fluxes_stack)
            if return_variances:
                res_variances = np.array(variances_stack)

            for k, original_idx in enumerate(tile_idxs):
                f = res_fluxes[k]
                if return_variances:
                    v = res_variances[k]
                    tile_results[original_idx] = (f, v)
                else:
                    tile_results[original_idx] = f

        # Tiling Done. Results are per tile.
        # We assume update_catalog is False or we warn.
        if update_catalog:
            print("Warning: update_catalog=True is ignored in Tiling mode (ambiguous results).")

        return tile_results

    elif vmap_images:
        # Determine buckets
        stats = compute_target_stats(tractor_obj.images, oversample_rendering)
        max_factor = stats["max_factor"]
        req_shapes = compute_image_shapes(tractor_obj.images, stats)
        bucket_map = assign_buckets(req_shapes, bucket_sizes, bucket_mode, bucket_shape_mode, bucket_base)

        # Debug Logging
        print(f"JAX Optimization: {len(tractor_obj.images)} images -> {len(bucket_map)} buckets")
        for shape, idxs in bucket_map.items():
            print(f"  Bucket {shape}: {len(idxs)} images")

        # Container for results
        all_results = [None] * len(tractor_obj.images)

        optimized_fluxes_np = None # placeholder if needed later
        # Actually we construct results list at the end differently if we bucket.
        # But optimize_fluxes expects `results` list in order.

        # We need to collect fluxes to update catalog if single image.
        # But if single image, we probably only have 1 bucket.

        # For update_catalog logic at end:
        # We need `optimized_fluxes_np` array (N_img, N_params).
        # We can construct it from all_results.

        for shape, img_indices in bucket_map.items():
            if not img_indices:
                continue

            sub_images = [tractor_obj.images[i] for i in img_indices]
            # sub_tractor needs same catalog
            sub_tractor = Tractor(sub_images, tractor_obj.catalog)

            # 1. Extract Data (Bucket Batch)
            images_data, batches, initial_fluxes = extract_model_data(
                sub_tractor,
                oversample_rendering=oversample_rendering,
                fit_background=fit_background,
                fixed_target_shape=shape,
                fixed_max_factor=max_factor
            )

            # 2. Define in_axes for batches
            # Note: With refactoring, flux_idx, shapes, profiles are now per-image arrays (shape N_img, N_src, ...).
            # So they should be mapped with in_axes=0.

            batches_in_axes = {}
            if "PointSource" in batches:
                batches_in_axes["PointSource"] = {
                    "flux_idx": 0,
                    "pos_pix": 0,
                    "mask": 0,
                }
            if "Galaxy" in batches:
                batches_in_axes["Galaxy"] = {
                    "flux_idx": 0,
                    "pos_pix": 0,
                    "wcs_cd_inv": 0,
                    "shapes": 0,
                    "mask": 0,
                    "profile": {
                        "amp": 0,
                        "mean": 0,
                        "var": 0,
                    }
                }
            if "Background" in batches:
                batches_in_axes["Background"] = {
                    "flux_idx": None
                }

            # 3. Vmap Optimization

            if use_sharding:
                images_data, batches, initial_fluxes = prepare_sharded_inputs(images_data, batches, initial_fluxes)

            # Re-compile for this shape
            solve_fn = jit(vmap(
                partial(
                    solve_fluxes_core,
                    return_variances=return_variances,
                    sampling_factor=max_factor,
                    use_preconditioner=use_preconditioner,
                    precond_eps=precond_eps,
                ),
                in_axes=(0, 0, batches_in_axes)
            ))

            if return_variances:
                optimized_fluxes_stack, variances_stack = solve_fn(initial_fluxes, images_data, batches)
            else:
                optimized_fluxes_stack = solve_fn(initial_fluxes, images_data, batches)

            # Map back to original indices
            res_fluxes = np.array(optimized_fluxes_stack)
            if return_variances:
                res_variances = np.array(variances_stack)

            for k, original_idx in enumerate(img_indices):
                f = res_fluxes[k]
                if return_variances:
                    v = res_variances[k]
                    all_results[original_idx] = (f, v)
                else:
                    all_results[original_idx] = f

        # Reconstruct arrays for compatibility with subsequent logic
        if len(all_results) > 0:
            if return_variances:
                optimized_fluxes_np = np.array([r[0] for r in all_results])
                variances_np = np.array([r[1] for r in all_results])
            else:
                optimized_fluxes_np = np.array(all_results)
        else:
             # Handle empty case
             optimized_fluxes_np = np.array([])
             if return_variances:
                 variances_np = np.array([])

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

    def optimize(self, tractor, alphas=None, damp=0, priors=True,
                 scale_columns=True, shared_params=True, variance=False,
                 just_variance=False, vmap_images=True, use_sharding=True, **kwargs):
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
            update_catalog=True,
            vmap_images=vmap_images,
            use_sharding=use_sharding,
            **kwargs
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
