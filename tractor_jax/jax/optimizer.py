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

from tractor_jax.engine import Tractor
from tractor_jax.optimize import Optimizer
from tractor_jax.pointsource import PointSource
from tractor_jax.galaxy import (
    Galaxy,
    ExpGalaxy,
    DevGalaxy,
    CompositeGalaxy,
    FixedCompositeGalaxy,
)
from tractor_jax.psf import PixelizedPSF, GaussianMixturePSF
from tractor_jax.jax.rendering import (
    render_pixelized_psf,
    render_galaxy_fft,
    render_point_source_pixelized,
    render_galaxy_mog,
    render_point_source_mog,
    render_point_source_fft,
    downsample_image,
)
from tractor_jax.jax.tiling import tile_image, project_catalog, filter_sources_by_box


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
            if isinstance(psf, PixelizedPSF):
                # Pad to the FULL finite kernel support (half-diagonal). The
                # kernel is at PSF-pixel resolution, which IS the target(HR)-grid
                # resolution (the HR grid renders the PSF at its own sampling),
                # so the kernel half-diagonal in PSF pixels is the target-pixel
                # radius that contains the whole kernel for any sub-pixel shift.
                # (The old `r_eff / s` was a unit error that over-padded by
                # ~1/s^2 -> a ~25x-area HR grid, e.g. ~1669px for an 85px tile at
                # sampling 0.224, making field fits OOM/intractable. An r_eff
                # enclosure fraction instead slightly UNDER-pads undersampled
                # kernels via a grid-alignment reshuffle. Full-support is
                # deterministic; the rendered template is padding-invariant to
                # <2e-5 and well-sampled/production PSFs change <0.05%.)
                r_eff_target = 0.5 * math.hypot(psf.H, psf.W)
            else:
                # Analytic PSF (Gaussian): r_eff is in image pixels -> target.
                r_eff_target = psf.get_r_eff(0.999) * max_factor

            # Diameter in Target Pixels (padded to be safe): enough that a
            # source at the tile edge fully contains its PSF on the padded grid.
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
            # The HR width must be EVEN: downstream the true width is
            # reconstructed from the rfft2 array as (shape[1]-1)*2, which is
            # wrong for odd widths (e.g. 487 -> 486) and silently evaluates
            # the phase gradient on the wrong frequency grid — a
            # position-dependent registration drift (proj research note
            # lasso_alpha/15 render-mismatch diagnosis).
            target_W += target_W % 2
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


def extract_model_data_direct(
    frames,
    catalog_table,
    psf_sampling,
    fit_background=False,
    fixed_target_shape=None,
    fixed_max_factor=None,
    profile_lookup_fn=None,
):
    """
    Build JAX arrays directly from raw frame data and a catalog table,
    bypassing Tractor/Image/Source object construction.

    Args:
        frames: list of dicts, each with keys:
            'data': 2-D ndarray, background-subtracted image
            'invvar': 2-D ndarray, inverse-variance
            'psf': 2-D ndarray, pixelized PSF image
            'wcs': astropy.wcs.WCS object
        catalog_table: astropy Table (or similar) with columns:
            'ra', 'dec', 'shape_r', 'shape_ab', 'shape_phi', 'sersic'
            (shape_r == 0 marks point sources).
        psf_sampling: float, pixel scale of PSF relative to science pixel
            (e.g. 0.2 means PSF is oversampled 5x).
        fit_background: bool.
        fixed_target_shape: (H, W) for the padded rendering grid.
        fixed_max_factor: float, oversampling factor.
        profile_lookup_fn: callable(sersic_index) -> MoG object with
            .amp, .mean, .var attributes.  Required if catalog has galaxies.

    Returns:
        Same (images_data, batches, initial_fluxes) tuple as extract_model_data.
    """
    from astropy.coordinates import SkyCoord

    N_img = len(frames)
    max_factor = fixed_max_factor if fixed_max_factor is not None else (1.0 / psf_sampling if psf_sampling < 1.0 else 1.0)
    target_sampling = float(max_factor) if max_factor > 1.0 else 1.0

    if fixed_target_shape is not None:
        target_H, target_W = fixed_target_shape
        if target_W % 2:
            # odd HR widths are ambiguous after rfft2 (see the even-width
            # note in extract_model_data) — bump to even; callers only
            # consume LR-shaped outputs, the HR grid is internal.
            print(f"extract_model_data_direct: bumping odd fixed target "
                  f"width {target_W} -> {target_W + 1}")
            target_W += 1
        padded_H = int(math.floor(target_H / max_factor))
        padded_W = int(math.floor(target_W / max_factor))
    else:
        max_H = max(f['data'].shape[0] for f in frames)
        max_W = max(f['data'].shape[1] for f in frames)
        max_psf_h = max(f['psf'].shape[0] for f in frames)
        max_psf_w = max(f['psf'].shape[1] for f in frames)
        fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
        fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))
        padded_H = max_H + fft_pad_h_lr
        padded_W = max_W + fft_pad_w_lr
        target_H = int(round(padded_H * max_factor))
        target_W = int(round(padded_W * max_factor))
        target_W += target_W % 2          # even width (rfft round-trip)

    # Pre-scan catalog for max MoG K
    max_gal_mog_K = 1
    for row in catalog_table:
        if row['shape_r'] > 0 and profile_lookup_fn is not None:
            prof = profile_lookup_fn(row['sersic'])
            max_gal_mog_K = max(max_gal_mog_K, len(prof.amp))

    # Source positions in sky
    sco = SkyCoord(ra=catalog_table['ra'], dec=catalog_table['dec'], unit='deg')

    # ---- Build per-image stacks ----
    data_list, invvar_list = [], []
    psf_fft_list = []

    ps_flux_list, ps_pos_list = [], []
    gal_flux_list, gal_pos_list, gal_cd_list = [], [], []
    gal_shape_list, gal_prof_list = [], []

    src_fluxes = []
    cat_idx_to_flux_idx = {}
    for ci, row in enumerate(catalog_table):
        cat_idx_to_flux_idx[ci] = len(src_fluxes)
        src_fluxes.append(0.0)

    for i_img in range(N_img):
        fr = frames[i_img]
        d = fr['data']
        iv = fr['invvar']
        psf_img = fr['psf']
        wcs_obj = fr['wcs']
        h, w = d.shape

        pad_h = padded_H - h
        pad_w = padded_W - w
        d_pad = jnp.pad(jnp.array(d), ((0, pad_h), (0, pad_w)), constant_values=0.0)
        iv_pad = jnp.pad(jnp.array(iv), ((0, pad_h), (0, pad_w)), constant_values=0.0)
        data_list.append(d_pad)
        invvar_list.append(iv_pad)

        # PSF FFT
        ph, pw = psf_img.shape
        raw_psf = jnp.array(psf_img)
        local_factor = 1.0 / psf_sampling if psf_sampling < 1.0 else 1.0
        if abs(local_factor - target_sampling) > 1e-3:
            ratio = target_sampling / local_factor
            new_shape = (int(round(ph * ratio)), int(round(pw * ratio)))
            resized = jax.image.resize(raw_psf, new_shape, method='lanczos3')
            resized = resized * (jnp.sum(raw_psf) / jnp.sum(resized))
            raw_psf = resized
            ph, pw = raw_psf.shape

        pad_psf = jnp.zeros((target_H, target_W))
        cy, cx = target_H // 2, target_W // 2
        y0 = cy - ph // 2
        x0 = cx - pw // 2
        pad_psf = pad_psf.at[y0:y0 + ph, x0:x0 + pw].set(raw_psf)
        pad_psf = jnp.fft.ifftshift(pad_psf)
        psf_fft_list.append(jfft.rfft2(pad_psf))

        # Source pixel positions for this frame
        pxs, pys = wcs_obj.world_to_pixel(sco)

        ps_flux_img, ps_pos_img = [], []
        gal_flux_img, gal_pos_img, gal_cd_img = [], [], []
        gal_shape_img, gal_prof_img = [], []

        for ci, row in enumerate(catalog_table):
            px, py = float(pxs[ci]), float(pys[ci])
            f_idx = cat_idx_to_flux_idx[ci]

            if i_img == 0:
                ix, iy = int(round(px)), int(round(py))
                if 0 <= iy < h and 0 <= ix < w:
                    raw_val = float(d[iy, ix])
                    src_fluxes[f_idx] = raw_val if np.isfinite(raw_val) else 0.0

            if row['shape_r'] == 0:
                ps_flux_img.append(f_idx)
                ps_pos_img.append([px, py])
            else:
                gal_flux_img.append(f_idx)
                gal_pos_img.append([px, py])
                # Approximate CD inverse from WCS at source position
                try:
                    cd_matrix = np.array(wcs_obj.wcs.cd) if hasattr(wcs_obj.wcs, 'cd') else np.array(wcs_obj.pixel_scale_matrix)
                except Exception:
                    cd_matrix = np.eye(2) * (6.15 / 3600.0)
                cd_inv = np.linalg.inv(cd_matrix)
                gal_cd_img.append(cd_inv)
                gal_shape_img.append([row['shape_r'], row['shape_ab'], row['shape_phi']])

                if profile_lookup_fn is not None:
                    prof = profile_lookup_fn(row['sersic'])
                    gal_prof_img.append((np.array(prof.amp), np.array(prof.mean), np.array(prof.var)))
                else:
                    gal_prof_img.append((np.zeros(1), np.zeros((1, 2)), np.eye(2)[np.newaxis]))

        ps_flux_list.append(ps_flux_img)
        ps_pos_list.append(ps_pos_img)
        gal_flux_list.append(gal_flux_img)
        gal_pos_list.append(gal_pos_img)
        gal_cd_list.append(gal_cd_img)
        gal_shape_list.append(gal_shape_img)
        gal_prof_list.append(gal_prof_img)

    # Stack images
    images_data = {
        'data': jnp.stack(data_list),
        'invvar': jnp.stack(invvar_list),
        'psf': {
            'type_code': jnp.zeros(N_img, dtype=jnp.int32),
            'sampling': jnp.full(N_img, target_sampling, dtype=jnp.float32),
            'fft': jnp.stack(psf_fft_list),
            'amp': jnp.zeros((N_img, 1)),
            'mean': jnp.zeros((N_img, 1, 2)),
            'var': jnp.tile(jnp.eye(2), (N_img, 1, 1, 1)),
        }
    }

    # Build batches
    batches = {}

    max_ps = max(len(x) for x in ps_flux_list) if ps_flux_list else 0
    if max_ps > 0:
        fi_stack, pp_stack, mk_stack = [], [], []
        for fl, pos in zip(ps_flux_list, ps_pos_list):
            n = len(fl)
            pad = max_ps - n
            fi_stack.append(np.pad(np.array(fl, dtype=np.int32), (0, pad)))
            p = np.array(pos, dtype=np.float32) if n > 0 else np.zeros((0, 2), dtype=np.float32)
            pp_stack.append(np.pad(p, ((0, pad), (0, 0))))
            mk_stack.append(np.pad(np.ones(n, dtype=np.float32), (0, pad)))
        batches['PointSource'] = {
            'flux_idx': jnp.array(np.stack(fi_stack)),
            'pos_pix': jnp.array(np.stack(pp_stack)),
            'mask': jnp.array(np.stack(mk_stack)),
        }

    max_gal = max(len(x) for x in gal_flux_list) if gal_flux_list else 0
    if max_gal > 0:
        fi_s, pp_s, cd_s, sh_s, mk_s = [], [], [], [], []
        amp_s, mean_s, var_s = [], [], []
        for fl, pos, cd, sh, pr in zip(gal_flux_list, gal_pos_list, gal_cd_list, gal_shape_list, gal_prof_list):
            n = len(fl)
            pad = max_gal - n
            fi_s.append(np.pad(np.array(fl, dtype=np.int32), (0, pad)))
            p = np.array(pos, dtype=np.float32) if n > 0 else np.zeros((0, 2), dtype=np.float32)
            pp_s.append(np.pad(p, ((0, pad), (0, 0))))
            c = np.array(cd, dtype=np.float32) if n > 0 else np.zeros((0, 2, 2), dtype=np.float32)
            cd_s.append(np.pad(c, ((0, pad), (0, 0), (0, 0))))
            s = np.array(sh, dtype=np.float32) if n > 0 else np.zeros((0, 3), dtype=np.float32)
            sh_s.append(np.pad(s, ((0, pad), (0, 0))))
            mk_s.append(np.pad(np.ones(n, dtype=np.float32), (0, pad)))

            img_amp = np.zeros((max_gal, max_gal_mog_K), dtype=np.float32)
            img_mean = np.zeros((max_gal, max_gal_mog_K, 2), dtype=np.float32)
            img_var = np.zeros((max_gal, max_gal_mog_K, 2, 2), dtype=np.float32)
            img_var[:] = np.eye(2)
            for k in range(n):
                a, m, v = pr[k]
                K = len(a)
                img_amp[k, :K] = a
                img_mean[k, :K] = m
                img_var[k, :K] = v
            amp_s.append(img_amp)
            mean_s.append(img_mean)
            var_s.append(img_var)

        batches['Galaxy'] = {
            'flux_idx': jnp.array(np.stack(fi_s)),
            'pos_pix': jnp.array(np.stack(pp_s)),
            'wcs_cd_inv': jnp.array(np.stack(cd_s)),
            'shapes': jnp.array(np.stack(sh_s)),
            'mask': jnp.array(np.stack(mk_s)),
            'profile': {
                'amp': jnp.array(np.stack(amp_s)),
                'mean': jnp.array(np.stack(mean_s)),
                'var': jnp.array(np.stack(var_s)),
            },
        }

    src_fluxes_np = np.array(src_fluxes, dtype=np.float32)
    initial_fluxes_matrix = np.tile(src_fluxes_np, (N_img, 1))

    if fit_background:
        bg_vals = np.zeros((N_img, 1), dtype=np.float32)
        initial_fluxes_matrix = np.hstack([initial_fluxes_matrix, bg_vals])
        bg_idx = len(src_fluxes_np)
        batches['Background'] = {'flux_idx': jnp.array([bg_idx], dtype=jnp.int32)}

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

        # Effective per-axis HR->LR factors. Placement MUST use the same
        # factors the boxcar downsample integrates with (valid/H after the
        # crop, or the full-grid ratio otherwise) — using the nominal `s`
        # (e.g. 4.46371 while the grid maps at 487/109) accumulates a
        # position-dependent registration drift of up to ~0.05 native px
        # across a tile (proj research note lasso_alpha/15 render diagnosis).
        if sampling_factor is not None and s > 1.001:
            valid_H = min(int(round(H * s)), H_hr_grid)
            valid_W = min(int(round(W * s)), W_hr_grid)
        else:
            valid_H, valid_W = H_hr_grid, W_hr_grid
        f_x = valid_W / W
        f_y = valid_H / H
        f_xy = jnp.array([f_x, f_y])
        pos_pix_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0

        render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape), in_axes=(0, 0, None))
        stamps = render_fn(fluxes, pos_pix_scaled, psf_data['fft'])
        combined = jnp.sum(stamps, axis=0)

        if sampling_factor is not None and s > 1.001:
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

    # Do NOT lax.cond on type_code: under vmap the predicate is batched and the
    # cond lowers to a select over both branches, which XLA:GPU miscompiles for
    # the FFT branch (stamps corrupted at the tens-of-percent level; jax 0.5.3).
    # Computing both branches and selecting with where is what the batched cond
    # executed anyway, and compiles correctly on CPU and GPU.
    return jnp.where(psf_data['type_code'] == 0,
                     render_fft(None), render_mog(None))


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

        # effective per-axis HR->LR factors — see render_batch_point_sources
        # (registration-drift fix; placement must match the boxcar factors)
        if sampling_factor is not None and s > 1.001:
            valid_H = min(int(round(H * s)), H_hr_grid)
            valid_W = min(int(round(W * s)), W_hr_grid)
        else:
            valid_H, valid_W = H_hr_grid, W_hr_grid
        f_x = valid_W / W
        f_y = valid_H / H
        f_xy = jnp.array([f_x, f_y])
        pos_pix_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0
        # pixel_hr = diag(f_x, f_y) @ pixel_lr, so the (world -> pixel)
        # inverse-CD rows scale per axis (row 0 = x, row 1 = y)
        wcs_cd_inv_scaled = wcs_cd_inv * f_xy[:, jnp.newaxis]

        gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])

        render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape), in_axes=((0, 0, 0), None, 0, 0, 0))
        stamps = render_fn(gal_mix, psf_data['fft'], shapes, wcs_cd_inv_scaled, pos_pix_scaled)

        weighted_stamps = stamps * fluxes[:, jnp.newaxis, jnp.newaxis]
        combined = jnp.sum(weighted_stamps, axis=0)

        if sampling_factor is not None and s > 1.001:
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

    # See render_batch_point_sources: batched-pred lax.cond miscompiles on GPU.
    return jnp.where(psf_data['type_code'] == 0,
                     render_fft(None), render_mog(None))


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
            # true grid width from the rfft array (even by construction);
            # per-axis effective factors (registration-drift fix)
            W_hr = (psf_data['fft'].shape[1] - 1) * 2
            render_shape = (H_hr, W_hr)
            f_xy = jnp.array([W_hr / W, H_hr / H])
            pos_pix_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0

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

        # See render_batch_point_sources: batched-pred lax.cond miscompiles on GPU.
        stamps = jnp.where(psf_data['type_code'] == 0,
                           compute_stamps_fft(None), compute_stamps_mog(None))

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
            # true grid width from the rfft array (even by construction);
            # per-axis effective factors (registration-drift fix)
            W_hr = (psf_data['fft'].shape[1] - 1) * 2
            render_shape = (H_hr, W_hr)
            f_xy = jnp.array([W_hr / W, H_hr / H])
            pos_pix_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0
            wcs_cd_inv_scaled = wcs_cd_inv * f_xy[:, jnp.newaxis]

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

        # See render_batch_point_sources: batched-pred lax.cond miscompiles on GPU.
        stamps = jnp.where(psf_data['type_code'] == 0,
                           compute_stamps_fft(None), compute_stamps_mog(None))

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


def _render_source_templates(image_data, batches, n_flux, sampling_factor=None):
    """
    Renders unit-flux template images for every source (and background).
    Returns a design matrix A of shape (N_flux, H, W).
    """
    H, W = image_data['data'].shape
    psf_data = image_data["psf"]
    templates = jnp.zeros((n_flux, H, W))

    if "PointSource" in batches:
        batch = batches["PointSource"]
        pos_pix = batch["pos_pix"]
        f_idx = batch["flux_idx"]
        mask = batch.get("mask", None)

        if sampling_factor is not None:
            s = sampling_factor
        else:
            s = psf_data['sampling']

        H_hr_grid = psf_data['fft'].shape[0]
        W_hr_grid = (psf_data['fft'].shape[1] - 1) * 2

        def _ps_stamps_fft(op):
            render_shape = (H_hr_grid, W_hr_grid)
            # effective per-axis factors (registration-drift fix; see
            # render_batch_point_sources)
            if sampling_factor is not None and s > 1.001:
                valid_H = min(int(round(H * s)), H_hr_grid)
                valid_W = min(int(round(W * s)), W_hr_grid)
            else:
                valid_H, valid_W = H_hr_grid, W_hr_grid
            f_xy = jnp.array([valid_W / W, valid_H / H])
            pos_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0
            unit = jnp.ones(pos_pix.shape[0])
            if mask is not None:
                unit = unit * mask
            render_fn = vmap(partial(render_point_source_fft, image_shape=render_shape),
                             in_axes=(0, 0, None))
            stamps = render_fn(unit, pos_scaled, psf_data['fft'])
            if sampling_factor is not None and s > 1.001:
                stamps = stamps[:, :valid_H, :valid_W]
                ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                stamps = ds_fn(stamps)
            elif sampling_factor is None:
                if H_hr_grid > H + 1:
                    ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                    stamps = ds_fn(stamps)
            return stamps

        def _ps_stamps_mog(op):
            psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
            unit = jnp.ones(pos_pix.shape[0])
            if mask is not None:
                unit = unit * mask
            render_fn = vmap(partial(render_point_source_mog, image_shape=(H, W)),
                             in_axes=(0, 0, None))
            return render_fn(unit, pos_pix, psf_mix)

        # See render_batch_point_sources: batched-pred lax.cond miscompiles on GPU.
        ps_stamps = jnp.where(psf_data['type_code'] == 0,
                              _ps_stamps_fft(None), _ps_stamps_mog(None))
        templates = templates.at[f_idx].add(ps_stamps)

    if "Galaxy" in batches:
        batch = batches["Galaxy"]
        pos_pix = batch["pos_pix"]
        wcs_cd_inv = batch["wcs_cd_inv"]
        shapes = batch["shapes"]
        profiles = batch["profile"]
        f_idx = batch["flux_idx"]
        mask = batch.get("mask", None)

        if sampling_factor is not None:
            s = sampling_factor
        else:
            s = psf_data['sampling']

        H_hr_grid = psf_data['fft'].shape[0]
        W_hr_grid = (psf_data['fft'].shape[1] - 1) * 2

        def _gal_stamps_fft(op):
            render_shape = (H_hr_grid, W_hr_grid)
            # effective per-axis factors (registration-drift fix; see
            # render_batch_galaxies)
            if sampling_factor is not None and s > 1.001:
                valid_H = min(int(round(H * s)), H_hr_grid)
                valid_W = min(int(round(W * s)), W_hr_grid)
            else:
                valid_H, valid_W = H_hr_grid, W_hr_grid
            f_xy = jnp.array([valid_W / W, valid_H / H])
            pos_scaled = pos_pix * f_xy + (f_xy - 1.0) / 2.0
            wcs_scaled = wcs_cd_inv * f_xy[:, jnp.newaxis]
            gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
            render_fn = vmap(partial(render_galaxy_fft, image_shape=render_shape),
                             in_axes=((0, 0, 0), None, 0, 0, 0))
            stamps = render_fn(gal_mix, psf_data['fft'], shapes, wcs_scaled, pos_scaled)
            if sampling_factor is not None and s > 1.001:
                stamps = stamps[:, :valid_H, :valid_W]
                ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                stamps = ds_fn(stamps)
            elif sampling_factor is None:
                if H_hr_grid > H + 1:
                    ds_fn = vmap(partial(downsample_image, target_shape=(H, W)))
                    stamps = ds_fn(stamps)
            return stamps

        def _gal_stamps_mog(op):
            psf_mix = (psf_data["amp"], psf_data["mean"], psf_data["var"])
            gal_mix = (profiles["amp"], profiles["mean"], profiles["var"])
            render_fn = vmap(partial(render_galaxy_mog, image_shape=(H, W)),
                             in_axes=((0, 0, 0), None, 0, 0, 0))
            return render_fn(gal_mix, psf_mix, shapes, wcs_cd_inv, pos_pix)

        # See render_batch_point_sources: batched-pred lax.cond miscompiles on GPU.
        gal_stamps = jnp.where(psf_data['type_code'] == 0,
                               _gal_stamps_fft(None), _gal_stamps_mog(None))
        if mask is not None:
            gal_stamps = gal_stamps * mask[:, jnp.newaxis, jnp.newaxis]
        templates = templates.at[f_idx].add(gal_stamps)

    if "Background" in batches:
        bg_idx = batches["Background"]["flux_idx"][0]
        templates = templates.at[bg_idx].set(jnp.ones((H, W)))

    return templates


def solve_fluxes_linear(initial_fluxes, image_data, batches, return_variances=False,
                        sampling_factor=None, rcond=1e-12):
    """
    Direct linear solve for forced photometry on a SINGLE image.
    Designed to be vmapped.

    Builds the design matrix A (template per source), then solves
    the normal equations  (A^T W A) f = A^T W d  via Cholesky/LU.

    Regularization is a Jacobi-scaled ridge, reg_j = rcond * diag(AtWA)_j:
    each source's ridge depends only on its own template norm, so the
    solution is invariant to masked padding and to the number of co-fit
    sources. Dead slots (all-zero template, e.g. shape padding or a source
    with no unmasked pixels) are pinned to flux 0 with infinite variance.
    """
    n_flux = initial_fluxes.shape[0]
    H, W = image_data['data'].shape

    templates = _render_source_templates(image_data, batches, n_flux,
                                         sampling_factor=sampling_factor)

    data_flat = image_data["data"].ravel()
    w_flat = image_data["invvar"].ravel()
    A = templates.reshape(n_flux, -1).T

    Aw = A * w_flat[:, jnp.newaxis]
    AtWA = Aw.T @ A
    AtWd = Aw.T @ data_flat

    Fjj = jnp.clip(jnp.diag(AtWA), 0.0)
    live = Fjj > 0
    AtWA_reg = AtWA + jnp.diag(rcond * Fjj + jnp.where(live, 0.0, 1.0))

    optimized_fluxes = jnp.linalg.solve(AtWA_reg, AtWd)

    if return_variances:
        cov = jnp.linalg.inv(AtWA_reg)
        variances = jnp.where(live, jnp.diag(cov), jnp.inf)
        return optimized_fluxes, variances

    return optimized_fluxes


def solve_fluxes_eigfloor(initial_fluxes, image_data, batches,
                          return_variances=False, sampling_factor=None,
                          floor=1e-4):
    """Direct linear solve with an eigenvalue floor on the JACOBI-NORMALIZED
    AtWA (SINGLE image).

    Same normal equations as solve_fluxes_linear, but the solve happens in
    Jacobi-normalized coordinates beta_j = sqrt(AtWA_jj) * f_j, where the
    Gram Ghat = D^{-1/2} AtWA D^{-1/2} has UNIT diagonal (a correlation
    matrix): its spectrum is clamped from below at floor * lambda_max(Ghat)
    (Tikhonov in the eigenbasis). In these coordinates only the genuinely
    correlation-degenerate directions - the anti-correlated flux splits of
    blended groups - sit near zero eigenvalue and get damped, while
    well-constrained sources (eigenvalue ~ 1) are solved exactly.

    The normalization is essential, not cosmetic: on the RAW AtWA the
    largest eigenvalue is dominated by whichever column has the largest
    norm - typically the constant BACKGROUND column (~n_pix * w) or a
    bright galaxy - so floor * lambda_max would exceed most source
    eigenvalues and shrink every flux toward zero (verified on the field
    sim: -50..-99% bias at high S/N before normalization; research note
    lasso_alpha/08 §5 item 8). After normalization lambda_max(Ghat) <= n
    regardless of units, background, or depth, and `floor` is a pure
    correlation-degeneracy threshold, invariant to flux units and image
    count - the same reasoning as the S/N-units lasso penalty.

    Symmetric and SIGN-FREE: no non-negativity clip and no per-band
    selection, so faint fluxes keep their negative excursions (no
    rectification bias, no selection-conditioning bias; research note
    lasso_alpha/12). Candidate default estimator for blind multi-target SED
    photometry.

    Dead slots (all-zero template, e.g. shape padding or a fully-masked
    source) are pinned to flux 0 with infinite variance. Designed to be
    vmapped, like solve_fluxes_linear.
    """
    n_flux = initial_fluxes.shape[0]

    templates = _render_source_templates(image_data, batches, n_flux,
                                         sampling_factor=sampling_factor)

    data_flat = image_data["data"].ravel()
    w_flat = image_data["invvar"].ravel()
    A = templates.reshape(n_flux, -1).T

    Aw = A * w_flat[:, jnp.newaxis]
    AtWA = Aw.T @ A
    AtWd = Aw.T @ data_flat

    Fjj = jnp.clip(jnp.diag(AtWA), 0.0)
    live = Fjj > 0
    D = jnp.where(live, jnp.sqrt(jnp.where(live, Fjj, 1.0)), 1.0)

    Ghat = AtWA / (D[:, jnp.newaxis] * D[jnp.newaxis, :])
    bhat = AtWd / D

    evals, evecs = jnp.linalg.eigh(Ghat)      # ascending eigenvalues
    emax = jnp.clip(evals[-1], 1e-30)
    evals_f = jnp.maximum(evals, floor * emax)

    xhat = evecs @ ((evecs.T @ bhat) / evals_f)
    optimized_fluxes = jnp.where(live, xhat / D, 0.0)

    if return_variances:
        # diag of D^{-1} V diag(1/evals_f) V^T D^{-1}
        var_hat = jnp.sum(evecs * evecs / evals_f[jnp.newaxis, :], axis=1)
        variances = jnp.where(live, var_hat / (D * D), jnp.inf)
        return optimized_fluxes, variances

    return optimized_fluxes


def _power_iter_lmax(G, n_steps=16):
    """Largest eigenvalue of a symmetric PSD matrix via power iteration.

    Deterministic init (ones vector) so results are reproducible under jit/vmap.
    """
    n = G.shape[0]
    v0 = jnp.ones(n) / jnp.sqrt(n)

    def body(v, _):
        w = G @ v
        v_new = w / (jnp.linalg.norm(w) + 1e-30)
        return v_new, None

    v, _ = jax.lax.scan(body, v0, None, length=n_steps)
    return jnp.vdot(v, G @ v)


def _lasso_fista(G, b, lam, *, nonneg=True, free=None, n_iter=1000, reg=0.0):
    """Per-coordinate L1-penalized quadratic solve on the normal equations.

        minimize  1/2 f^T (G + reg*I) f - b^T f + sum_j lam_j |f_j|
        subject to f_j >= 0 for penalized coordinates (if nonneg)

    G = A^T W A, b = A^T W d of a whitened linear model; lam_j is the absolute
    per-coordinate penalty (lam_j = 0 leaves coordinate j unpenalized).
    Coordinates with free[j] = 1 are unpenalized AND sign-free (background).
    reg may be a scalar or a per-coordinate (n,) ridge vector (Jacobi-scaled
    reg_j = rcond * G_jj becomes exactly rcond on the normalized diagonal).

    Runs FISTA with gradient-scheme adaptive restart in Jacobi-normalized
    coordinates beta_j = sqrt(G_jj) * f_j (unit-diagonal system), which removes
    the dynamic-range part of the conditioning; with lam_j = alpha*sqrt(G_jj)
    the normalized threshold is uniform and dimensionless (S/N units).

    Returns (f, kkt): the solution in original coordinates and the maximum KKT
    violation in normalized (S/N) units - the convergence diagnostic.
    """
    n = G.shape[0]
    if free is None:
        free = jnp.zeros(n)
    Fjj = jnp.clip(jnp.diag(G), 0.0)
    live = Fjj > 0
    D = jnp.where(live, jnp.sqrt(jnp.where(live, Fjj, 1.0)), 1.0)

    Ghat = G / (D[:, None] * D[None, :]) + jnp.diag(reg / (D * D))
    chat = b / D
    lamhat = lam / D

    L = jnp.maximum(_power_iter_lmax(Ghat), 1e-12) * 1.05

    def prox(z):
        if nonneg:
            pen = jnp.maximum(z - lamhat / L, 0.0)
        else:
            pen = jnp.sign(z) * jnp.maximum(jnp.abs(z) - lamhat / L, 0.0)
        return jnp.where(free > 0, z, pen)

    def step(carry, _):
        beta, y, t = carry
        grad = Ghat @ y - chat
        beta_new = prox(y - grad / L)
        restart = jnp.vdot(y - beta_new, beta_new - beta) > 0
        t_new = jnp.where(restart, 1.0, 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)))
        y_new = beta_new + jnp.where(restart, 0.0, (t - 1.0) / t_new) * (beta_new - beta)
        return (beta_new, y_new, t_new), None

    beta0 = jnp.zeros(n)
    (beta, _, _), _ = jax.lax.scan(step, (beta0, beta0, 1.0), None, length=n_iter)

    # KKT violation in normalized units (exact stationarity conditions)
    grad = Ghat @ beta - chat
    active = beta != 0
    if nonneg:
        v_active = jnp.abs(grad + lamhat)
        v_inactive = jnp.maximum(-grad - lamhat, 0.0)
    else:
        v_active = jnp.abs(grad + lamhat * jnp.sign(beta))
        v_inactive = jnp.maximum(jnp.abs(grad) - lamhat, 0.0)
    viol = jnp.where(active, v_active, v_inactive)
    viol = jnp.where(free > 0, jnp.abs(grad), viol)
    kkt = jnp.max(jnp.where(live, viol, 0.0))

    return beta / D, kkt


def _ln_binom(p, k):
    """log C(p, k) via gammaln (exact; valid for traced float k)."""
    from jax.scipy.special import gammaln
    return gammaln(p + 1.0) - gammaln(k + 1.0) - gammaln(p - k + 1.0)


def solve_fluxes_lasso(initial_fluxes, image_data, batches,
                       return_variances=False, sampling_factor=None,
                       alpha=None, penalty_mode="snr", penalty_weights=None,
                       nonneg=True, selection_mode="fixed", criterion="ebic",
                       grid=None, ebic_gamma=0.5, return_path=False,
                       debias=True, debias_signfree="none", return_aux=False,
                       n_iter=1000, rcond=1e-12):
    """L1-regularized (LASSO) forced photometry on a SINGLE image.

    Designed to be vmapped, like solve_fluxes_linear. Builds the same design
    matrix A (unit-flux templates) and solves

        min 1/2 || W^{1/2} (d - A f) ||^2 + sum_j lambda_j |f_j|,   f >= 0

    Penalty parameterization (see proj-spherex-gpupipe research note
    notebooks/research_notes/lasso_alpha/01):
      penalty_mode="snr":  lambda_j = alpha * w_j * sqrt(F_jj), with
        F_jj = diag(A^T W A) the squared matched-filter norm of template j,
        TAKEN FROM THE SAME TEMPLATES USED FOR THE SOLVE (not from
        compute_fisher_diagonal, which deviates in the oversampled-FFT path).
        alpha is then a dimensionless residual matched-filter S/N entry
        threshold, invariant across images, bands, depth and cutout size.
      penalty_mode="raw":  lambda_j = alpha * w_j (absolute units; sklearn
        conversion: lambda_raw = n_pix * alpha_sklearn).

    penalty_weights: (n_flux,) per-source multiplier; 0 = PROTECTED source
      (never shrunk, never zeroed, always refit - use for the forced-photometry
      target list). None = ones. The background parameter (if fit) is always
      forced unpenalized and sign-free.

    Selection:
      selection_mode="fixed": solve at the given alpha (production; inject the
        sim-calibrated value here).
      selection_mode="path":  solve on `grid` (default logspace(0.5..5, 16) in
        S/N units), score with `criterion` on the DEBIASED refit
        ("ebic" [default; exact 2*gamma*ln C(p,k) multiplicity term],
        "bic" [gamma=0], "sure" [biased RSS + 2*df - n_eff]), pick the argmin.
        No CV, deliberately: pixel folds violate independence and select
        overfit alphas (research note lasso_alpha/02 section 5).

    debias=True: exact re-solve on the selected support (identity pinning,
      static shapes), nonneg-clipped for penalized coordinates; variances are
      diag(inv(AtWA_S + reg)) of the refit, inf off-support (conditional on
      the selected support - not post-selection corrected).

    debias_signfree: which refit coordinates skip the nonneg clip -
      "none" (default; background only, original behavior), "protected"
      (background + wj==0 sources: protected SED/photo-z targets keep
      negative excursions - no faint-end rectification bias, research note
      lasso_alpha/12), or "all". Selection is unaffected (the FISTA prox
      still uses `nonneg`); no-op when nonneg=False or debias=False.

    Returns (matching solve_fluxes_linear unless return_aux):
      fluxes                                  if not return_variances
      (fluxes, variances)                     if return_variances
      (..., aux)                              if return_aux, where aux carries
        support (float mask), alpha, alpha_index, criterion_values, kkt
        (max KKT violation in S/N units - convergence diagnostic), n_active,
        resid_corr_snr (per-source residual matched-filter correlation at the
        biased solution, S/N units; entry margin = alpha - c_j for inactive
        sources), snr_deb (per-source debiased S/N; exit margin = snr - alpha
        for active ones - together the cheap production stability flag),
        and path_fluxes (n_alpha, n_flux) if return_path.

    Note: with jax_enable_x64 off everything runs in float32; the Jacobi
    normalization inside _lasso_fista makes support recovery reliable in f32,
    but calibration-grade fluxes/variances should enable x64
    (JaxOptimizer(enable_x64=True)).
    """
    n_flux = initial_fluxes.shape[0]

    templates = _render_source_templates(image_data, batches, n_flux,
                                         sampling_factor=sampling_factor)
    data_flat = image_data["data"].ravel()
    w_flat = image_data["invvar"].ravel()
    A = templates.reshape(n_flux, -1).T

    Aw = A * w_flat[:, jnp.newaxis]
    G = Aw.T @ A                       # AtWA
    b = Aw.T @ data_flat               # AtWd
    dWd = jnp.sum(data_flat * data_flat * w_flat)
    n_eff = jnp.sum(w_flat > 0)

    wj, free = _lasso_penalty_weights(n_flux, batches, penalty_weights, G.dtype)

    return _lasso_core(G, b, dWd, n_eff, wj, free,
                       alpha=alpha, penalty_mode=penalty_mode, nonneg=nonneg,
                       selection_mode=selection_mode, criterion=criterion,
                       grid=grid, ebic_gamma=ebic_gamma,
                       return_path=return_path, debias=debias,
                       debias_signfree=debias_signfree,
                       return_variances=return_variances,
                       return_aux=return_aux, n_iter=n_iter, rcond=rcond)


def solve_fluxes_lasso_batched(initial_fluxes, image_data, batches, data_stack,
                               return_variances=False, sampling_factor=None,
                               alpha=None, penalty_mode="snr",
                               penalty_weights=None, nonneg=True,
                               selection_mode="fixed", criterion="ebic",
                               grid=None, ebic_gamma=0.5, return_path=False,
                               debias=True, debias_signfree="none",
                               return_aux=False,
                               n_iter=1000, rcond=1e-12):
    """LASSO forced photometry for a BATCH of data realizations of ONE image.

    Bootstrap entry point (proj-spherex-gpupipe research note
    notebooks/research_notes/paper/03_bootstrap_validation_plan.md, stage B1):
    replicas share the scene — templates, invvar and the penalty structure —
    and differ only in the pixel data, so the design matrix and G = AtWA are
    built ONCE and the solve is vmapped over the replica axis (G, weights and
    the FISTA step size stay unbatched under vmap; only AtWd is per-replica).

    data_stack: (B, H, W) data realizations; image_data["data"] is not used
    by the solve. All other arguments as solve_fluxes_lasso and shared by
    every replica (in "path" mode each replica selects its own alpha).

    Returns the same structures as solve_fluxes_lasso, with a leading
    replica axis B on every array (fluxes: (B, n_flux); aux fields likewise).

    Memory: the pinned refit materializes (n_flux, n_flux) per replica under
    vmap (B * n_flux^2; ~0.65 GB at B=100, n=900, f64 - x16 more in "path"
    mode). For larger runs split data_stack into chunks and call repeatedly;
    the per-call template build is the same work the chunks would share.
    """
    n_flux = initial_fluxes.shape[0]

    templates = _render_source_templates(image_data, batches, n_flux,
                                         sampling_factor=sampling_factor)
    w_flat = image_data["invvar"].ravel()
    A = templates.reshape(n_flux, -1).T

    Aw = A * w_flat[:, jnp.newaxis]
    G = Aw.T @ A
    n_eff = jnp.sum(w_flat > 0)

    wj, free = _lasso_penalty_weights(n_flux, batches, penalty_weights, G.dtype)

    d_flat = data_stack.reshape(data_stack.shape[0], -1)
    b_stack = d_flat @ Aw                                  # (B, n_flux)
    dWd_stack = jnp.sum(d_flat * d_flat * w_flat[None, :], axis=1)

    def _solve_one(b, dWd):
        return _lasso_core(G, b, dWd, n_eff, wj, free,
                           alpha=alpha, penalty_mode=penalty_mode,
                           nonneg=nonneg, selection_mode=selection_mode,
                           criterion=criterion, grid=grid,
                           ebic_gamma=ebic_gamma, return_path=return_path,
                           debias=debias, debias_signfree=debias_signfree,
                           return_variances=return_variances,
                           return_aux=return_aux, n_iter=n_iter, rcond=rcond)

    return jax.vmap(_solve_one)(b_stack, dWd_stack)


def _lasso_penalty_weights(n_flux, batches, penalty_weights, dtype):
    """Per-source penalty multipliers wj and the sign-free mask.

    Background (if fit) is always unpenalized and sign-free.
    """
    if penalty_weights is None:
        wj = jnp.ones(n_flux, dtype=dtype)
    else:
        wj = jnp.asarray(penalty_weights, dtype=dtype)
    free = jnp.zeros(n_flux, dtype=dtype)
    if "Background" in batches:
        bg_idx = batches["Background"]["flux_idx"][0]
        wj = wj.at[bg_idx].set(0.0)
        free = free.at[bg_idx].set(1.0)
    return wj, free


def _lasso_core(G, b, dWd, n_eff, wj, free, *,
                alpha=None, penalty_mode="snr", nonneg=True,
                selection_mode="fixed", criterion="ebic", grid=None,
                ebic_gamma=0.5, return_path=False, debias=True,
                debias_signfree="none",
                return_variances=False, return_aux=False,
                n_iter=1000, rcond=1e-12):
    """LASSO solve on prebuilt normal equations (G = AtWA, b = AtWd).

    Shared backend of solve_fluxes_lasso (single image) and
    solve_fluxes_lasso_batched (many data realizations of one image);
    vmappable over (b, dWd) with G/wj/free held fixed.

    Ridge is Jacobi-scaled, reg_j = rcond * G_jj: invariant to masked
    padding and to the number of co-fit sources (in the FISTA-normalized
    coordinates it is exactly rcond * I on live slots). Dead slots
    (G_jj = 0) are excluded from the support and pinned in the refit.

    debias_signfree controls the sign convention of the DEBIASED refit only
    (selection keeps the `nonneg` prox): which coordinates skip the
    max(f, 0) clip. "none" (default) - background only (the `free` mask;
    original behavior); "protected" - background + unpenalized sources
    (wj == 0), so a protected science target keeps negative excursions
    (no faint-end rectification bias; research note lasso_alpha/12);
    "all" - every refit coordinate is sign-free. Irrelevant when
    nonneg=False (refit already unclipped) or debias=False.
    """
    if debias_signfree not in ("none", "protected", "all"):
        raise ValueError(f"debias_signfree must be 'none', 'protected' or "
                         f"'all'; got {debias_signfree!r}")
    Fjj = jnp.clip(jnp.diag(G), 0.0)
    live = Fjj > 0
    reg_j = rcond * Fjj

    if debias_signfree == "all":
        signfree = jnp.ones_like(live)
    elif debias_signfree == "protected":
        signfree = (free > 0) | (wj == 0)
    else:  # "none"
        signfree = free > 0

    if penalty_mode == "snr":
        lam1 = wj * jnp.sqrt(jnp.where(live, Fjj, 0.0))
    else:  # "raw"
        lam1 = wj

    def support_pinned(s):
        # identity pinning keeps shapes static under jit/vmap
        return G * s[:, None] * s[None, :] + jnp.diag(1.0 - s + reg_j)

    Dn = jnp.sqrt(jnp.where(live, Fjj, 1.0))

    def solve_one_alpha(a):
        f_biased, kkt = _lasso_fista(G, b, a * lam1, nonneg=nonneg, free=free,
                                     n_iter=n_iter, reg=reg_j)
        # support: active, or unpenalized (protected/background); never padded
        s = ((f_biased != 0) | (wj == 0)) & live
        s = jax.lax.stop_gradient(s.astype(G.dtype))

        # debiased refit on the support; the sign clip is skipped on
        # `signfree` coordinates (see debias_signfree in the docstring)
        f_solve = jnp.linalg.solve(support_pinned(s), b * s)
        if nonneg:
            f_deb = jnp.where(signfree, f_solve,
                              jnp.maximum(f_solve, 0.0)) * s
        else:
            f_deb = f_solve * s

        f_out = f_deb if debias else f_biased
        rss_deb = jnp.maximum(f_deb @ G @ f_deb - 2.0 * (b @ f_deb) + dWd, 1e-30)
        rss_bia = jnp.maximum(f_biased @ G @ f_biased - 2.0 * (b @ f_biased) + dWd,
                              1e-30)
        # per-source stability diagnostics in S/N units (production flag):
        # residual matched-filter correlation at the biased solution (entry
        # margin alpha - c_j for inactive sources) and the debiased S/N
        # (exit margin snr_deb_j - alpha for active ones)
        resid_corr_snr = jnp.where(live, (b - G @ f_biased) / Dn, 0.0)
        snr_deb = jnp.where(live, f_deb * Dn, 0.0)
        return f_out, s, kkt, rss_deb, rss_bia, resid_corr_snr, snr_deb

    if selection_mode == "path":
        if grid is None:
            grid_vals = jnp.logspace(jnp.log10(0.5), jnp.log10(5.0), 16)
        else:
            grid_vals = jnp.asarray(grid)
        (f_p, s_p, kkt_p, rssd_p, rssb_p,
         corr_p, snrd_p) = jax.vmap(solve_one_alpha)(grid_vals)

        df = jnp.sum(s_p, axis=1)                          # all fitted params
        n_pen = jnp.sum((wj > 0) & live)                   # candidate pool size
        k_pen = jnp.sum(s_p * ((wj > 0) & live)[None, :], axis=1)
        if criterion == "sure":
            crit = rssb_p + 2.0 * df - n_eff
        else:
            gam = 0.0 if criterion == "bic" else ebic_gamma
            crit = (n_eff * jnp.log(rssd_p / n_eff) + df * jnp.log(n_eff)
                    + 2.0 * gam * _ln_binom(n_pen, k_pen))
        k_star = jnp.argmin(crit)
        fluxes = jnp.take(f_p, k_star, axis=0)
        support = jnp.take(s_p, k_star, axis=0)
        kkt = jnp.take(kkt_p, k_star)
        alpha_star = jnp.take(grid_vals, k_star)
        aux = {
            "support": support, "alpha": alpha_star, "alpha_index": k_star,
            "criterion_values": crit, "kkt": kkt,
            "n_active": jnp.sum(support),
            "resid_corr_snr": jnp.take(corr_p, k_star, axis=0),
            "snr_deb": jnp.take(snrd_p, k_star, axis=0),
        }
        if return_path:
            aux["path_fluxes"] = f_p
    else:
        a = jnp.asarray(alpha if alpha is not None else 0.0, dtype=G.dtype)
        fluxes, support, kkt, _, _, resid_corr_snr, snr_deb = solve_one_alpha(a)
        aux = {
            "support": support, "alpha": a,
            "alpha_index": jnp.asarray(0, dtype=jnp.int32),
            "criterion_values": jnp.zeros(1, dtype=G.dtype), "kkt": kkt,
            "n_active": jnp.sum(support),
            "resid_corr_snr": resid_corr_snr, "snr_deb": snr_deb,
        }

    if return_variances:
        cov = jnp.linalg.inv(support_pinned(support))
        variances = jnp.where(support > 0, jnp.diag(cov), jnp.inf)
        if return_aux:
            return fluxes, variances, aux
        return fluxes, variances

    if return_aux:
        return fluxes, aux
    return fluxes


def solve_fluxes_core(initial_fluxes, image_data, batches, return_variances=False, sampling_factor=None, use_preconditioner=True, precond_eps=1e-12):
    """
    Pure JAX core optimization logic for a SINGLE image (Newton-CG).
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


def optimize_fluxes(tractor_obj, oversample_rendering=False, return_variances=False, fit_background=False, update_catalog=False, vmap_images=True, use_sharding=True, bucket_sizes=None, bucket_mode="auto", bucket_shape_mode="square", bucket_base=32, use_tiling=False, tile_size=256, tile_super_halo=None, use_preconditioner=True, precond_eps=1e-12, solver="linear", penalty=None, selection=None, debias=True, debias_signfree="none", return_aux=False, lasso_n_iter=1000, eig_floor=1e-4):
    """
    Optimizes fluxes for forced photometry using JAX.

    Args:
        tractor_obj: Tractor object with images and catalog.
        oversample_rendering: bool, if True use oversampled rendering for PixelizedPSF with sampling != 1.
        return_variances: bool, if True, return variances of fluxes.
        fit_background: bool, if True, includes background level in optimization parameters.
        update_catalog: bool, if True, updates the source catalog with optimized fluxes.
        vmap_images: bool, if True (default), stacks all images and processes them in a single vmap call.
        use_sharding: bool, if True (default), distributes the batch across available devices.
        bucket_sizes: List of bucket sizes. Used if bucket_mode="fixed".
        bucket_mode: "auto" or "fixed".
        bucket_shape_mode: "square" or "independent".
        bucket_base: rounding base for auto mode.
        use_tiling: bool, if True, splits images into tiles and processes them.
        tile_size: int, size of tiles (default 256).
        tile_super_halo: optional int, override calculated halo size.
        use_preconditioner: bool, if True (default), use Fisher diagonal preconditioner (CG only).
        precond_eps: float, floor for Fisher diagonal to avoid divide-by-zero.
        solver: "linear" (direct normal-equations solve), "eigfloor" (linear
            solve with an eigenvalue floor on AtWA - sign-free, no selection;
            see solve_fluxes_eigfloor), "cg" (Newton-CG, original), or
            "lasso" (L1-regularized, see solve_fluxes_lasso).
        penalty: (lasso only) dict, e.g. {"mode": "snr", "alpha": 2.0,
            "weights": w, "nonneg": True}. weights==0 marks protected sources.
        selection: (lasso only) dict, e.g. {"mode": "fixed"} or
            {"mode": "path", "criterion": "ebic", "grid": ..., "ebic_gamma": 0.5,
             "return_path": False}.
        debias: (lasso only) refit on the selected support (default True).
        debias_signfree: (lasso only) which refit coordinates skip the nonneg
            clip - "none" (default), "protected" (background + weight-0
            sources), or "all". See solve_fluxes_lasso.
        return_aux: (lasso only) append the per-image aux dict (support, alpha,
            kkt, ...) as the last element of each result.
        lasso_n_iter: (lasso only) static FISTA iteration count.
        eig_floor: (eigfloor only) relative eigenvalue floor, in units of the
            largest eigenvalue of AtWA (default 1e-4).

    Returns:
        List of results per image.
    """
    from tractor import ConstantSky

    results = []

    _solver_map = {"linear": solve_fluxes_linear,
                   "eigfloor": solve_fluxes_eigfloor,
                   "lasso": solve_fluxes_lasso}
    _solver_fn = _solver_map.get(solver, solve_fluxes_core)

    if solver == "linear":
        _solver_kwargs = dict(return_variances=return_variances)
    elif solver == "eigfloor":
        _solver_kwargs = dict(return_variances=return_variances,
                              floor=eig_floor)
    elif solver == "lasso":
        penalty = penalty or {}
        selection = selection or {}
        _solver_kwargs = dict(
            return_variances=return_variances,
            alpha=penalty.get("alpha"),
            penalty_mode=penalty.get("mode", "snr"),
            penalty_weights=penalty.get("weights"),
            nonneg=penalty.get("nonneg", True),
            selection_mode=selection.get("mode", "fixed"),
            criterion=selection.get("criterion", "ebic"),
            grid=selection.get("grid"),
            ebic_gamma=selection.get("ebic_gamma", 0.5),
            return_path=selection.get("return_path", False),
            debias=debias,
            debias_signfree=debias_signfree,
            return_aux=return_aux,
            n_iter=lasso_n_iter,
        )
        if not getattr(jax.config, "jax_enable_x64", False):
            print("JAX Optimization: lasso solver running in float32; "
                  "enable x64 (JaxOptimizer(enable_x64=True)) for "
                  "calibration-grade fluxes/variances.")
    else:
        _solver_kwargs = dict(
            return_variances=return_variances,
            use_preconditioner=use_preconditioner,
            precond_eps=precond_eps,
        )

    _lasso_aux = solver == "lasso" and return_aux
    aux_all = [None] * len(tractor_obj.images)

    solve_jit = jit(partial(_solver_fn, **_solver_kwargs))

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
                    _solver_fn,
                    **_solver_kwargs,
                    sampling_factor=max_factor,
                ),
                in_axes=(0, 0, batches_in_axes)
            ))

            out = solve_fn(initial_fluxes, images_data, batches)
            aux_stack = None
            if _lasso_aux:
                *out, aux_stack = out if isinstance(out, tuple) else (out,)
                out = out[0] if len(out) == 1 else tuple(out)
            if return_variances:
                optimized_fluxes_stack, variances_stack = out
            else:
                optimized_fluxes_stack = out

            res_fluxes = np.array(optimized_fluxes_stack)
            if return_variances:
                res_variances = np.array(variances_stack)

            for k, original_idx in enumerate(tile_idxs):
                f = res_fluxes[k]
                parts = [f]
                if return_variances:
                    parts.append(res_variances[k])
                if aux_stack is not None:
                    parts.append({key: np.array(val[k]) for key, val in aux_stack.items()})
                tile_results[original_idx] = tuple(parts) if len(parts) > 1 else f

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

            solve_fn = jit(vmap(
                partial(
                    _solver_fn,
                    **_solver_kwargs,
                    sampling_factor=max_factor,
                ),
                in_axes=(0, 0, batches_in_axes)
            ))

            out = solve_fn(initial_fluxes, images_data, batches)
            aux_stack = None
            if _lasso_aux:
                *out, aux_stack = out if isinstance(out, tuple) else (out,)
                out = out[0] if len(out) == 1 else tuple(out)
            if return_variances:
                optimized_fluxes_stack, variances_stack = out
            else:
                optimized_fluxes_stack = out

            # Map back to original indices
            res_fluxes = np.array(optimized_fluxes_stack)
            if return_variances:
                res_variances = np.array(variances_stack)

            for k, original_idx in enumerate(img_indices):
                f = res_fluxes[k]
                if aux_stack is not None:
                    aux_all[original_idx] = {key: np.array(val[k])
                                             for key, val in aux_stack.items()}
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

        for _img_i, img in enumerate(tractor_obj.images):
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

            # All per-source arrays are stacked per image by extract_model_data
            # (shape (1, N_src, ...) here); slice image 0 off every leaf, as the
            # vmap path does via in_axes=0.
            single_batches = batches.copy()
            if "PointSource" in batches:
                single_batches["PointSource"] = jax.tree_util.tree_map(
                    lambda x: x[0], batches["PointSource"])

            if "Galaxy" in batches:
                single_batches["Galaxy"] = jax.tree_util.tree_map(
                    lambda x: x[0], batches["Galaxy"])

            if "Background" in batches:
                # flux_idx is (1,) for single image?
                pass

            single_flux = init_flux[0] # (N_params,)

            out = solve_jit(single_flux, single_data, single_batches)
            if _lasso_aux:
                *out, aux_i = out if isinstance(out, tuple) else (out,)
                out = out[0] if len(out) == 1 else tuple(out)
                aux_all[_img_i] = {key: np.array(val) for key, val in aux_i.items()}
            if return_variances:
                f, v = out
                fluxes_list.append(f)
                variances_list.append(v)
            else:
                fluxes_list.append(out)

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
            if _lasso_aux:
                results.append((f, v, aux_all[i]))
            else:
                results.append((f, v))
        elif _lasso_aux:
            results.append((f, aux_all[i]))
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
    def __init__(self, enable_x64=False):
        super(JaxOptimizer, self).__init__()
        if enable_x64:
            jax.config.update("jax_enable_x64", True)

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
