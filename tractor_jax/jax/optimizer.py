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


def psf_kind(psf):
    """Classify a PSF for the batched renderer.

    Returns ``("pixelized", psf)`` for a `PixelizedPSF` — or for a hybrid that
    wraps one in ``.pix``, whose pixelized model is the accurate one — and
    ``("mog", mixture)`` for anything exposing a mixture-of-Gaussians
    representation (`GaussianMixturePSF`, `NCircularGaussianPSF`,
    `GaussianMixtureEllipsePSF`, ...).

    Raises
    ------
    TypeError
        If the PSF is neither. This is deliberate: an unrecognised PSF used to
        fall through to an all-zero template, so the solve silently returned
        flux 0 with infinite variance for every source instead of failing.
    """
    if isinstance(psf, PixelizedPSF) or isinstance(getattr(psf, "pix", None),
                                                   PixelizedPSF):
        return "pixelized", psf
    mog = getattr(psf, "mog", None)
    if mog is None:
        getter = getattr(psf, "getMixtureOfGaussians", None)
        mog = getter() if getter is not None else None
    if mog is not None and hasattr(mog, "amp"):
        return "mog", mog
    raise TypeError(
        f"unsupported PSF for the batched JAX path: {type(psf).__name__}. "
        "Use a PixelizedPSF, a GaussianMixturePSF, or any PSF exposing "
        "getMixtureOfGaussians().")


def _even_hr_width_shrink(padded_w, max_factor, max_drop=8):
    """Largest ``padded_w' <= padded_w`` whose high-res width is even.

    The bucketed path cannot GROW the low-res width (that would overflow the
    bucket), so it shrinks instead. Returns ``padded_w`` unchanged if no small
    decrement works.
    """
    for drop in range(max_drop + 1):
        w = padded_w - drop
        if w > 0 and int(round(w * max_factor)) % 2 == 0:
            return w
    return padded_w


def _even_hr_width_pad(padded_w, max_factor, max_extra=8):
    """Grow a low-res padded width until the high-res width lands even.

    The HR grid width must be even for the rfft2 round-trip, and it must stay
    an exact integer multiple of the LR width so that ``downsample_image``
    takes its integer-factor path. Adding one HR column satisfies the first
    and breaks the second, which resamples every rendered template; adding
    LR padding satisfies both (padding is zero-weight, so it is free).

    Returns ``padded_w`` unchanged if no small bump works (non-integer
    ``max_factor``); the caller then keeps the previous behaviour.
    """
    for extra in range(max_extra + 1):
        if int(round((padded_w + extra) * max_factor)) % 2 == 0:
            return padded_w + extra
    return padded_w


def compute_image_shapes(images, stats):
    """
    Compute the required target-grid shape for each image.

    Parameters
    ----------
    images : list
        Image objects with a ``shape`` attribute giving (H, W).
    stats : dict
        Global statistics from `compute_target_stats`, with keys
        ``max_factor``, ``fft_pad_h_lr`` and ``fft_pad_w_lr``.

    Returns
    -------
    shapes : list of tuple of int
        Per-image ``(target_h, target_w)``: the FFT-padded input shape
        scaled by ``max_factor``.
    """
    max_factor = stats["max_factor"]
    fft_pad_h_lr = stats["fft_pad_h_lr"]
    fft_pad_w_lr = stats["fft_pad_w_lr"]

    shapes = []
    for img in images:
        h, w = img.shape
        padded_h = h + fft_pad_h_lr
        padded_w = _even_hr_width_pad(w + fft_pad_w_lr, max_factor)

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
    Assign images to shape buckets based on their required grid shapes.

    Parameters
    ----------
    required_shapes : list of tuple of int
        Required (H, W) target grid shape per image.
    bucket_sizes : list, optional
        Allowed bucket sizes when ``bucket_mode="fixed"``; entries may be
        ints (square buckets) or (H, W) tuples. Defaults to powers of two
        from 32 to 4096.
    bucket_mode : {"auto", "fixed"}, optional
        "auto" derives the buckets from the shape distribution; "fixed"
        uses ``bucket_sizes`` as the allowed grid.
    bucket_shape_mode : {"square", "independent"}, optional
        Whether auto-mode buckets are forced square or may have independent
        height and width.
    bucket_base : int, optional
        Quantization base: required shapes are rounded up to multiples of
        this value.
    max_buckets : int, optional
        Maximum number of distinct buckets in auto mode.

    Returns
    -------
    bucket_map : dict
        Mapping ``{bucket_shape: [img_indices]}``, where each bucket shape
        is an (H, W) tuple large enough for all its assigned images.
    """

    # 1. Determine available buckets
    allowed_sizes = []
    allowed_shapes = []

    if bucket_mode == "fixed":
        if bucket_sizes is None:
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

        allowed_shapes = norm_shapes

    else: # auto
        # Quantize required shapes up to multiples of bucket_base
        quantized_shapes = []
        for h, w in required_shapes:
            h_q = int(math.ceil(h / bucket_base) * bucket_base)
            w_q = int(math.ceil(w / bucket_base) * bucket_base)
            quantized_shapes.append((h_q, w_q))

        if bucket_shape_mode == "square":
            sq_sizes = []
            for h, w in quantized_shapes:
                s = max(h, w)
                sq_sizes.append(s)

            counts = Counter(sq_sizes)
            max_size = max(sq_sizes) if sq_sizes else bucket_base

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
    Compute global statistics for a set of images to size the rendering grids.

    Determines the common oversampling factor and the FFT padding (derived
    from the PSF support) shared by every image in the set.

    Parameters
    ----------
    images : list
        Image objects; each PSF (via ``getPsf``) is inspected for its
        sampling and effective radius / kernel support.
    oversample_rendering : bool, optional
        If True, undersampled ``PixelizedPSF`` instances (sampling < 1)
        raise the common oversampling factor so rendering happens at PSF
        resolution.

    Returns
    -------
    stats : dict
        Keys ``max_factor`` (common oversampling factor), ``max_psf_h`` and
        ``max_psf_w`` (maximum PSF support in target pixels), and
        ``fft_pad_h_lr`` and ``fft_pad_w_lr`` (padding in input-resolution
        pixels).
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
            # No r_eff: fall back to the raw kernel footprint. `s` is PSF
            # pixels per image pixel, so the kernel spans ph * max_factor / s
            # target pixels (dividing by s, not multiplying — the old *s
            # version exploded for oversampled PSFs).
            if oversample_rendering:
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
    img_source_indices=None,
    compact_fluxes=False,
    use_psf_fft_cache=True,
):
    """
    Extract all data needed for JAX optimization from a Tractor object.

    Groups sources into batches and stacks image data with padding for
    vectorized rendering.

    Parameters
    ----------
    tractor_obj : Tractor
        Tractor object providing ``images`` and ``catalog``.
    oversample_rendering : bool, optional
        If True, handles oversampled ``PixelizedPSF`` by rendering at high
        resolution.
    fit_background : bool, optional
        If True, includes a per-image constant background level in the
        optimization parameters.
    fixed_target_shape : tuple of int, optional
        (H, W) tuple. If provided, forces the target grid size to this
        shape. Useful for bucketing.
    fixed_max_factor : float, optional
        The oversampling factor assumed for the bucket. Required if
        ``fixed_target_shape`` is provided.
    img_source_indices : sequence or dict, optional
        Per-image collection of catalog indices to include; if None, all
        supported catalog sources are used for every image.
    compact_fluxes : bool, optional
        If True (requires ``img_source_indices``), each image row gets its
        own compact flux vector containing only that image's sources, padded
        to the largest per-image parameter count. The background slot (when
        fit) sits at the shared last column. Used by the tiling path, where
        every "image" is a tile seeing a small catalog subset; a shared
        full-catalog vector would grow quadratically in the solver.

    Returns
    -------
    images_data : dict
        Stacked image data (``data``, ``invvar``, ``psf``). Shapes are
        (N_img, max_H, max_W) or (N_img, ...).
    batches : dict
        Batched source data. Shapes are (N_img, N_src, ...).
    initial_fluxes : jax.numpy.ndarray
        Initial fluxes of shape (N_img, N_params). Source fluxes are
        broadcast/shared across images (or per-image compact slots when
        ``compact_fluxes``); the background parameter (if fit) is per-image.
    slot_maps : list of dict
        Only returned when ``compact_fluxes`` is True: per image, a dict
        mapping catalog index -> (slot offset, n_params) in that image's
        flux row.
    """
    from tractor_jax.sky import ConstantSky
    images = tractor_obj.images
    catalog = tractor_obj.catalog

    if fixed_target_shape is not None:
        if fixed_max_factor is None:
            raise ValueError("fixed_max_factor is required when fixed_target_shape is used.")

        target_H, target_W = fixed_target_shape
        max_factor = fixed_max_factor

        # The bucket is an UPPER BOUND on the HR grid, not the grid itself.
        # Taking padded = floor(bucket/max_factor) and then rendering on the
        # full bucket leaves target/padded non-integer (256/51 = 5.02 for a
        # power-of-two bucket at max_factor 5), which drops
        # `downsample_image` to its boxcar path and resamples every template.
        # Shrink to the largest exact multiple of the LR grid that fits inside
        # the bucket, with an even HR width for the rfft2 round-trip. The
        # result is still deterministic per bucket, so compile-shape sharing
        # -- the whole point of bucketing -- is preserved.
        padded_H = int(math.floor(target_H / max_factor))
        padded_W = int(math.floor(target_W / max_factor))
        padded_W = _even_hr_width_shrink(padded_W, max_factor)
        target_H = int(round(padded_H * max_factor))
        target_W = int(round(padded_W * max_factor))

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
            # The HR width must be EVEN: downstream the true width is
            # reconstructed from the rfft2 array as (shape[1]-1)*2, which is
            # wrong for odd widths (e.g. 487 -> 486) and silently evaluates
            # the phase gradient on the wrong frequency grid — a
            # position-dependent registration drift (proj research note
            # lasso_alpha/15 render-mismatch diagnosis).
            #
            # Buy that even width with EXTRA LOW-RES PADDING, never by adding
            # one HR column: `target_W += target_W % 2` leaves target_W no
            # longer an integer multiple of padded_W, so `downsample_image`
            # silently drops to the boxcar path and resamples every template.
            # On SPHEREx (max_factor 5) that shifted sum(t^2) by ~5% and
            # fitted fluxes by ~2%, differently for every stamp size — it is
            # what made the tiled and whole-image geometries disagree.
            padded_W = _even_hr_width_pad(padded_W, max_factor)
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
        kind, obj = psf_kind(img.getPsf())
        if kind == "mog":
            max_mog_K = max(max_mog_K, len(obj.amp))

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

        psf_kind_name, psf_mog = psf_kind(psf)

        if psf_kind_name == "pixelized":
            p_type = 0

            s = getattr(psf, "sampling", 1.0)
            local_factor = 1.0/s if s < 1.0 else 1.0

            # The padded oversampled rfft2 depends only on the PSF stamp and
            # the (target shape, sampling) geometry, so staple it on the PSF
            # object — repeated extracts over the same PSF (buckets, tiles,
            # re-fits) skip the resize+FFT. Mirrors the CPU-side
            # PixelizedPSF.fftcache convention (same stamp-immutability
            # assumption).
            fft_key = (target_H, target_W, round(float(p_sampling), 9))
            cached_fft = (getattr(psf, "_jax_fft_cache", {}).get(fft_key)
                          if use_psf_fft_cache else None)
            if cached_fft is not None:
                p_fft = cached_fft
            else:
                if abs(local_factor - p_sampling) > 1e-3:
                    # Resize PSF image to match target resolution
                    raw_img = jnp.array(psf.img)
                    ph, pw = raw_img.shape
                    ratio = p_sampling / local_factor
                    new_shape = (int(round(ph * ratio)), int(round(pw * ratio)))

                    resized_img = jax.image.resize(raw_img, new_shape, method='lanczos3')

                    # Normalize flux to preserve sum
                    orig_sum = jnp.sum(raw_img)
                    new_sum = jnp.sum(resized_img)
                    resized_img = resized_img * (orig_sum / new_sum)

                    raw_img = resized_img
                else:
                    raw_img = jnp.array(psf.img)

                ph, pw = raw_img.shape

                if ph % 2 == 0 or pw % 2 == 0:
                    raise ValueError(
                        f"PSF kernel {ph}x{pw} has an even axis: the centered "
                        "pad below anchors it by ph//2, which for an even size "
                        "is half a pixel off its true center (n-1)/2, so the "
                        "kernel lands 0.5 high-res px from the ifftshift "
                        "origin. Use an odd-sized kernel. See "
                        "batching.psf_to_fft's even_parity handling.")

                # Pad to (target_H, target_W), centered
                pad_img = jnp.zeros((target_H, target_W))
                cy, cx = target_H // 2, target_W // 2
                y0 = cy - ph // 2
                x0 = cx - pw // 2

                pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(raw_img)
                pad_img = jnp.fft.ifftshift(pad_img)
                p_fft = jfft.rfft2(pad_img)
                if use_psf_fft_cache:
                    if not hasattr(psf, "_jax_fft_cache"):
                        psf._jax_fft_cache = {}
                    psf._jax_fft_cache[fft_key] = p_fft

        else:
            p_type = 1
            K = len(psf_mog.amp)
            pad_len = max_mog_K - K

            amp = jnp.array(psf_mog.amp)
            mean = jnp.array(psf_mog.mean)
            var = jnp.array(psf_mog.var)

            if pad_len > 0:
                amp = jnp.pad(amp, (0, pad_len), constant_values=0)
                mean = jnp.pad(mean, ((0, pad_len), (0, 0)), constant_values=0)

                # Pad var with identity blocks (zero covariances would be singular)
                new_var = jnp.zeros((max_mog_K, 2, 2), dtype=var.dtype)
                new_var = new_var.at[:K].set(var)

                padding_eye = jnp.tile(jnp.eye(2), (pad_len, 1, 1))
                new_var = new_var.at[K:].set(padding_eye)
                var = new_var

            p_amp = amp
            p_mean = mean
            p_var = var

            # Dummy FFT (Zeros)
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

    if compact_fluxes and img_source_indices is None:
        raise ValueError("compact_fluxes=True requires img_source_indices")
    slot_maps = []

    N_img = len(images)
    for i_img in range(N_img):
        img = images[i_img]
        wcs = img.getWcs()

        if img_source_indices is not None:
            indices = img_source_indices[i_img]
        else:
            indices = sorted(cat_idx_to_flux_idx.keys())

        # Per-image compact slots: this image's sources packed contiguously
        # in catalog order, instead of the shared full-catalog offsets.
        slot_map = {}
        if compact_fluxes:
            off = 0
            for cat_idx in indices:
                cat_idx = int(cat_idx)
                if cat_idx not in cat_idx_to_flux_idx:
                    continue
                n_p = len(catalog[cat_idx].brightness.getParams())
                slot_map[cat_idx] = (off, n_p)
                off += n_p
        slot_maps.append(slot_map)

        ps_flux = []
        ps_pos = []

        gal_flux = []
        gal_pos = []
        gal_cd = []
        gal_shape = []
        gal_prof = [] # (amp, mean, var)

        for cat_idx in indices:
            cat_idx = int(cat_idx)
            if cat_idx not in cat_idx_to_flux_idx:
                continue

            src = catalog[cat_idx]
            if compact_fluxes:
                f_idx = slot_map[cat_idx][0]
            else:
                f_idx = cat_idx_to_flux_idx[cat_idx]

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

            f_arr = np.array(fl, dtype=np.int32)
            f_arr = np.pad(f_arr, (0, pad), constant_values=0)
            flux_idx_stack.append(f_arr)

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

            cd_arr = np.pad(cd_arr, ((0, pad), (0, 0), (0, 0)), constant_values=0) # zeros OK: padded slots are masked
            wcs_stack.append(cd_arr)

            sh_arr = np.pad(sh_arr, ((0, pad), (0, 0)), constant_values=0)
            shape_stack.append(sh_arr)

            m_arr = np.ones(n, dtype=np.float32)
            m_arr = np.pad(m_arr, (0, pad), constant_values=0)
            mask_stack.append(m_arr)

            # Profile padding (MoG): pad each source's mixture to
            # max_gal_mog_K components AND the source list to max_gal,
            # giving (max_gal, max_K, ...) arrays for this image.
            img_amp = np.zeros((max_gal, max_gal_mog_K), dtype=np.float32)
            img_mean = np.zeros((max_gal, max_gal_mog_K, 2), dtype=np.float32)
            img_var = np.zeros((max_gal, max_gal_mog_K, 2, 2), dtype=np.float32)
            # identity var keeps padded components non-singular
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

    if compact_fluxes:
        # Per-image compact rows padded to the widest image; padded slots
        # never appear in any flux_idx and stay dead in the solver.
        n_src_max = max((sum(n_p for (_, n_p) in m.values()) for m in slot_maps),
                        default=0)
        n_src_max = max(n_src_max, 1)
        initial_fluxes_matrix = np.zeros((N_img, n_src_max), dtype=np.float32)
        for i_img, slot_map in enumerate(slot_maps):
            for cat_idx, (off, n_p) in slot_map.items():
                params = catalog[cat_idx].brightness.getParams()
                initial_fluxes_matrix[i_img, off:off + n_p] = params
    else:
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

        initial_fluxes_matrix = np.hstack([initial_fluxes_matrix, bg_vals])

        # Each row carries its own bg param at the end, so the index is a
        # single row-relative scalar shared by all images.
        bg_idx = initial_fluxes_matrix.shape[1] - 1
        batches["Background"] = {
            "flux_idx": jnp.array([bg_idx], dtype=jnp.int32)
        }

    initial_fluxes_matrix = jnp.array(initial_fluxes_matrix, dtype=jnp.float32)
    if compact_fluxes:
        return images_data, batches, initial_fluxes_matrix, slot_maps
    return images_data, batches, initial_fluxes_matrix


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
    Build JAX arrays directly from raw frame data and a catalog table.

    Bypasses Tractor/Image/Source object construction.

    Parameters
    ----------
    frames : list of dict
        One dict per frame with keys ``'data'`` (2-D ndarray,
        background-subtracted image), ``'invvar'`` (2-D ndarray,
        inverse-variance), ``'psf'`` (2-D ndarray, pixelized PSF image) and
        ``'wcs'`` (`astropy.wcs.WCS` object).
    catalog_table : astropy.table.Table or similar
        Catalog with columns ``'ra'``, ``'dec'``, ``'shape_r'``,
        ``'shape_ab'``, ``'shape_phi'`` and ``'sersic'``
        (``shape_r == 0`` marks point sources).
    psf_sampling : float
        Pixel scale of the PSF relative to the science pixel (e.g. 0.2
        means the PSF is oversampled 5x).
    fit_background : bool, optional
        If True, includes a per-image constant background parameter.
    fixed_target_shape : tuple of int, optional
        (H, W) for the padded rendering grid.
    fixed_max_factor : float, optional
        Oversampling factor.
    profile_lookup_fn : callable, optional
        ``profile_lookup_fn(sersic_index)`` returning a MoG object with
        ``.amp``, ``.mean``, ``.var`` attributes. Required if the catalog
        has galaxies.

    Returns
    -------
    images_data : dict
        Stacked image data, as returned by `extract_model_data`.
    batches : dict
        Batched source data, as returned by `extract_model_data`.
    initial_fluxes : jax.numpy.ndarray
        Initial fluxes of shape (N_img, N_params), as returned by
        `extract_model_data`.
    """
    from astropy.coordinates import SkyCoord

    N_img = len(frames)
    max_factor = fixed_max_factor if fixed_max_factor is not None else (1.0 / psf_sampling if psf_sampling < 1.0 else 1.0)
    target_sampling = float(max_factor) if max_factor > 1.0 else 1.0

    if fixed_target_shape is not None:
        target_H, target_W = fixed_target_shape
        # See extract_model_data: shrink the bucket to the largest exact
        # integer multiple of the LR grid, with an even HR width. Bumping the
        # HR width instead would break the integer downsample factor.
        padded_H = int(math.floor(target_H / max_factor))
        padded_W = int(math.floor(target_W / max_factor))
        padded_W = _even_hr_width_shrink(padded_W, max_factor)
        target_H = int(round(padded_H * max_factor))
        target_W = int(round(padded_W * max_factor))
    else:
        max_H = max(f['data'].shape[0] for f in frames)
        max_W = max(f['data'].shape[1] for f in frames)
        max_psf_h = max(f['psf'].shape[0] for f in frames)
        max_psf_w = max(f['psf'].shape[1] for f in frames)
        fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
        fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))
        padded_H = max_H + fft_pad_h_lr
        padded_W = max_W + fft_pad_w_lr
        # even HR width via extra LOW-RES padding, so the HR->LR downsample
        # factor stays an exact integer (see extract_model_data)
        padded_W = _even_hr_width_pad(padded_W, max_factor)
        target_H = int(round(padded_H * max_factor))
        target_W = int(round(padded_W * max_factor))

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

        if ph % 2 == 0 or pw % 2 == 0:
            raise ValueError(
                f"PSF kernel {ph}x{pw} has an even axis: the centered pad "
                "below anchors it by ph//2, half a pixel off the true center "
                "(n-1)/2 for an even size, so it lands 0.5 high-res px from "
                "the ifftshift origin. Use an odd-sized kernel.")

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
    Render a batch of point sources onto a single image grid.

    Parameters
    ----------
    fluxes : jax.numpy.ndarray
        Per-source fluxes, shape (N_src,).
    pos_pix : jax.numpy.ndarray
        Pixel positions (x, y) per source, shape (N_src, 2).
    psf_data : dict
        Per-image PSF data with keys ``type_code``, ``sampling``, ``fft``,
        ``amp``, ``mean`` and ``var``.
    img_shape : tuple of int
        Output shape (H, W) at native image resolution.
    sampling_factor : float, optional
        High-resolution oversampling factor; if None, taken from
        ``psf_data['sampling']``.
    mask : jax.numpy.ndarray, optional
        Per-source validity mask (1 for real sources, 0 for padding);
        multiplied into the fluxes.

    Returns
    -------
    jax.numpy.ndarray
        Combined model image of shape ``img_shape``.

    Notes
    -----
    The PSF-type branch (pixelized/FFT vs. Gaussian mixture) is selected
    with ``jnp.where`` rather than ``lax.cond``: under ``vmap`` a
    batched-predicate ``cond`` lowers to a select over both branches, which
    XLA:GPU miscompiles for the FFT branch (stamps corrupted at the
    tens-of-percent level; jax 0.5.3). Computing both branches and
    selecting with ``where`` is what the batched cond executed anyway, and
    compiles correctly on CPU and GPU.
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
    Render a batch of galaxies onto a single image grid.

    Parameters
    ----------
    fluxes : jax.numpy.ndarray
        Per-source fluxes, shape (N_src,).
    pos_pix : jax.numpy.ndarray
        Pixel positions (x, y) per source, shape (N_src, 2).
    wcs_cd_inv : jax.numpy.ndarray
        Inverse CD (world-to-pixel) matrices per source, shape
        (N_src, 2, 2).
    shapes : jax.numpy.ndarray
        Galaxy shape parameters (re, ab, phi) per source, shape (N_src, 3).
    profiles : dict
        Galaxy mixture-of-Gaussians profiles with keys ``amp``, ``mean``
        and ``var``.
    psf_data : dict
        Per-image PSF data (see `render_batch_point_sources`).
    img_shape : tuple of int
        Output shape (H, W) at native image resolution.
    sampling_factor : float, optional
        High-resolution oversampling factor; if None, taken from
        ``psf_data['sampling']``.
    mask : jax.numpy.ndarray, optional
        Per-source validity mask; multiplied into the fluxes.

    Returns
    -------
    jax.numpy.ndarray
        Combined model image of shape ``img_shape``.

    Notes
    -----
    See `render_batch_point_sources` for why the PSF-type branch uses
    ``jnp.where`` instead of a batched-predicate ``lax.cond``.
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
    Distribute data across available devices using NamedSharding (GSPMD).

    Shards image-based arrays along axis 0 (the image batch axis) and
    replicates shared source parameters on all devices.

    Parameters
    ----------
    images_data : dict
        Stacked image data; every leaf has shape (N_img, ...).
    batches : dict
        Batched source data; per-image leaves (``pos_pix``,
        ``wcs_cd_inv``) are sharded, the others replicated.
    initial_fluxes : jax.numpy.ndarray
        Initial fluxes of shape (N_img, N_params).

    Returns
    -------
    images_data : dict
        Device-placed image data.
    batches : dict
        Device-placed batched source data.
    initial_fluxes : jax.numpy.ndarray
        Device-placed initial fluxes.

    Notes
    -----
    NamedSharding requires the batch axis to be divisible by the device
    count. When it is not, every per-image leaf (leading dim == N_img) is
    padded by repeating its last entry; the padded rows are solved
    redundantly and simply ignored by callers, which only read results
    for the original image indices.
    """
    devices = jax.devices()

    n_img = int(initial_fluxes.shape[0])
    pad = (-n_img) % len(devices)
    if pad:
        def _pad(x):
            x = jnp.asarray(x)
            if x.ndim and x.shape[0] == n_img:
                return jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)], axis=0)
            return x
        images_data = jax.tree_util.tree_map(_pad, images_data)
        batches = jax.tree_util.tree_map(_pad, batches)
        initial_fluxes = _pad(initial_fluxes)

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
    Render a single model image from sliced batch data.

    Sums point-source, galaxy and (optionally) constant-background
    contributions.

    Parameters
    ----------
    fluxes : jax.numpy.ndarray
        Flux parameter vector for this image, shape (N_flux,).
    image_data : dict
        Single-image data with keys ``data`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data (keys ``PointSource``,
        ``Galaxy``, ``Background``).
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the renderers.

    Returns
    -------
    jax.numpy.ndarray
        Model image of shape (H, W).
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
    Compute the diagonal of the Fisher information matrix for a single image.

    For each flux parameter s,
    ``F_ss = sum_pixels ( (dModel/dFlux_s)^2 * invvar )``.

    Parameters
    ----------
    image_data : dict
        Single-image data with keys ``data``, ``invvar`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    n_flux : int
        Total number of flux parameters.

    Returns
    -------
    jax.numpy.ndarray
        Fisher diagonal of shape (n_flux,).
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

        # render_batch_point_sources sums the stamps internally, but the Fisher
        # diagonal needs each per-source stamp squared, so the stamp rendering
        # is implemented inline here.
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
    Render unit-flux template images for every source (and background).

    Parameters
    ----------
    image_data : dict
        Single-image data with keys ``data`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    n_flux : int
        Total number of flux parameters.
    sampling_factor : float, optional
        High-resolution oversampling factor; if None, taken from
        ``psf_data['sampling']``.

    Returns
    -------
    jax.numpy.ndarray
        Design matrix A of shape (N_flux, H, W): one unit-flux template
        per flux parameter.
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

    Designed to be vmapped. Builds the design matrix A (one unit-flux
    template per source), then solves the normal equations
    ``(A^T W A) f = A^T W d`` via Cholesky/LU.

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Flux parameter vector, shape (n_flux,); only its length is used.
    image_data : dict
        Single-image data with keys ``data``, ``invvar`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    return_variances : bool, optional
        If True, also return flux variances (diagonal of the inverse
        regularized normal matrix).
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the template
        renderer.
    rcond : float, optional
        Jacobi-scaled ridge strength (see Notes).

    Returns
    -------
    optimized_fluxes : jax.numpy.ndarray
        Solved fluxes, shape (n_flux,).
    variances : jax.numpy.ndarray
        Flux variances, shape (n_flux,). Only returned if
        ``return_variances`` is True.

    Notes
    -----
    Regularization is a Jacobi-scaled ridge, ``reg_j = rcond * diag(AtWA)_j``:
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
    """
    Direct linear solve with an eigenvalue floor on the Jacobi-normalized AtWA.

    Operates on a SINGLE image; designed to be vmapped, like
    `solve_fluxes_linear`. Same normal equations as `solve_fluxes_linear`,
    but the solve happens in Jacobi-normalized coordinates
    ``beta_j = sqrt(AtWA_jj) * f_j``, where the Gram
    ``Ghat = D^{-1/2} AtWA D^{-1/2}`` has UNIT diagonal (a correlation
    matrix): its spectrum is clamped from below at
    ``floor * lambda_max(Ghat)`` (Tikhonov in the eigenbasis). In these
    coordinates only the genuinely correlation-degenerate directions - the
    anti-correlated flux splits of blended groups - sit near zero
    eigenvalue and get damped, while well-constrained sources
    (eigenvalue ~ 1) are solved exactly.

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Flux parameter vector, shape (n_flux,); only its length is used.
    image_data : dict
        Single-image data with keys ``data``, ``invvar`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    return_variances : bool, optional
        If True, also return flux variances.
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the template
        renderer.
    floor : float, optional
        Relative eigenvalue floor, in units of the largest eigenvalue of
        the normalized Gram matrix.

    Returns
    -------
    optimized_fluxes : jax.numpy.ndarray
        Solved fluxes, shape (n_flux,).
    variances : jax.numpy.ndarray
        Flux variances, shape (n_flux,). Only returned if
        ``return_variances`` is True.

    Notes
    -----
    The normalization is essential, not cosmetic: on the RAW AtWA the
    largest eigenvalue is dominated by whichever column has the largest
    norm - typically the constant BACKGROUND column (~n_pix * w) or a
    bright galaxy - so floor * lambda_max would exceed most source
    eigenvalues and shrink every flux toward zero (verified on the field
    sim: -50..-99% bias at high S/N before normalization; research note
    lasso_alpha/08 §5 item 8). After normalization lambda_max(Ghat) <= n
    regardless of units, background, or depth, and ``floor`` is a pure
    correlation-degeneracy threshold, invariant to flux units and image
    count - the same reasoning as the S/N-units lasso penalty.

    Symmetric and SIGN-FREE: no non-negativity clip and no per-band
    selection, so faint fluxes keep their negative excursions (no
    rectification bias, no selection-conditioning bias; research note
    lasso_alpha/12). Candidate default estimator for blind multi-target SED
    photometry.

    Dead slots (all-zero template, e.g. shape padding or a fully-masked
    source) are pinned to flux 0 with infinite variance.
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
    # Padded flux vectors can contain hundreds of exactly dead coordinates.
    # Leaving their rows/columns at zero is mathematically harmless, but
    # CUDA's symmetric eigensolver can fail to converge on the resulting
    # large null block (returning NaN for every eigenpair).  Replace that
    # discarded block by an identity block before ``eigh``.  The live block
    # is unchanged and has unit diagonal, hence lambda_max >= 1: dead
    # eigenvalues of 1 cannot change ``emax`` or the floor applied to live
    # modes.  Dead outputs are still forced to 0/inf below.
    live_outer = live[:, jnp.newaxis] & live[jnp.newaxis, :]
    Ghat = (jnp.where(live_outer, Ghat, 0.0)
            + jnp.diag((~live).astype(Ghat.dtype)))
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


def _eigfloor_prior_core(AtWA, AtWd, lambda_diag, f_prior, floor=1e-4,
                         return_variances=False):
    """
    Ridge-toward-prior eigfloor solve on prebuilt normal equations.

    Solves ``(AtWA + Lambda) f = AtWd + Lambda f_prior`` with
    ``Lambda = diag(lambda_diag)``, in the Jacobi-normalized coordinates of
    `solve_fluxes_eigfloor` (``D = sqrt(diag(AtWA))``, from the UNregularized
    Gram, so a ``lambda_diag = 0`` coordinate keeps its unit diagonal). The
    eigenvalue floor acts on the REGULARIZED normalized Gram
    ``Ghat + D^{-1} Lambda D^{-1}`` (floor after adding Lambda), and the
    variances are ``diag((AtWA + Lambda)^{-1})`` with the same floored
    spectrum — the prior tightens the reported uncertainty, as a Gaussian
    prior should.

    With ``lambda_diag = 0`` everywhere this reproduces the eigfloor solve
    on (AtWA, AtWd) exactly.

    Parameters
    ----------
    AtWA : jax.numpy.ndarray
        Normal matrix ``A^T W A``, shape (n, n).
    AtWd : jax.numpy.ndarray
        Right-hand side ``A^T W d``, shape (n,).
    lambda_diag : jax.numpy.ndarray
        Per-coordinate Gaussian-prior precision ``1/sigma_prior^2``,
        shape (n,); 0 = unregularized (protected). Must be finite —
        map ``sigma_prior = 0`` to a large finite value, not inf.
    f_prior : jax.numpy.ndarray
        Per-coordinate prior mean, shape (n,); ignored where
        ``lambda_diag = 0``.
    floor : float, optional
        Relative eigenvalue floor, in units of the largest eigenvalue of
        the regularized normalized Gram.
    return_variances : bool, optional
        If True, also return ``diag((AtWA + Lambda)^{-1})`` (floored).

    Returns
    -------
    fluxes : jax.numpy.ndarray
        Solution, shape (n,). Dead slots (``AtWA_jj = 0``) are pinned to 0.
    variances : jax.numpy.ndarray
        Shape (n,), only if ``return_variances``; inf on dead slots.
    """
    Fjj = jnp.clip(jnp.diag(AtWA), 0.0)
    live = Fjj > 0
    D = jnp.where(live, jnp.sqrt(jnp.where(live, Fjj, 1.0)), 1.0)

    # lam_hat can be enormous even with sane sigma_prior: Jacobi
    # normalization divides by D^2 = AtWA_jj, so a weak-overlap source
    # (template power ~ 0, e.g. at the image edge) has lam_hat = lam / D^2
    # -> 1e10+. Two consequences handled here:
    #   (1) the eigen-FLOOR must be relative to the UNregularized (data)
    #       Gram's largest eigenvalue — the floor exists to regularize
    #       data degeneracies; prior-dominated directions are already
    #       conditioned by Lambda. Flooring on the regularized emax would
    #       scale the floor with lam_hat_max and crush every direction
    #       (protected sources included).
    #   (2) cap lam_hat for eigh conditioning: 1e6 is far stiffer than any
    #       physical prior while keeping the matrix scale-mixed range sane.
    lam_hat = jnp.minimum(lambda_diag / (D * D), 1e6)
    Ghat_data = AtWA / (D[:, jnp.newaxis] * D[jnp.newaxis, :])
    live_outer = live[:, jnp.newaxis] & live[jnp.newaxis, :]
    Ghat_data = jnp.where(live_outer, Ghat_data, 0.0)
    # As in the blind eigfloor solver, pin the otherwise all-zero dead block
    # to identity before the GPU eigendecomposition.  This changes only
    # coordinates whose returned flux/variance are discarded as 0/inf.
    Ghat = Ghat_data + jnp.diag(
        lam_hat + (~live).astype(Ghat_data.dtype))
    bhat = AtWd / D + lam_hat * (D * f_prior)

    evals, evecs = jnp.linalg.eigh(Ghat)      # ascending eigenvalues
    emax_data = jnp.clip(_power_iter_lmax(Ghat_data), 1e-30)
    evals_f = jnp.maximum(evals, floor * emax_data)

    xhat = evecs @ ((evecs.T @ bhat) / evals_f)
    fluxes = jnp.where(live, xhat / D, 0.0)

    if return_variances:
        # diag of D^{-1} V diag(1/evals_f) V^T D^{-1}
        var_hat = jnp.sum(evecs * evecs / evals_f[jnp.newaxis, :], axis=1)
        variances = jnp.where(live, var_hat / (D * D), jnp.inf)
        return fluxes, variances

    return fluxes


def solve_fluxes_eigfloor_prior(initial_fluxes, image_data, batches,
                                lambda_diag=None, f_prior=None,
                                return_variances=False, sampling_factor=None,
                                floor=1e-4):
    """
    Eigfloor solve with per-source Gaussian flux priors (ridge-toward-prior).

    Operates on a SINGLE image; designed to be vmapped, like
    `solve_fluxes_eigfloor`. Solves::

        (AtWA + Lambda) f = AtWd + Lambda f_prior,
        Lambda = diag(lambda_diag) = diag(1 / sigma_prior^2)

    - ``lambda_diag_j = 0``: protected source, exactly the current eigfloor
      behavior (unbiased). With ``lambda_diag = 0`` everywhere the output is
      identical to `solve_fluxes_eigfloor`.
    - ``lambda_diag_j > 0``: nuisance source ridged toward its externally
      predicted flux ``f_prior_j`` (typical ``sigma_prior ~ 0.5-1 x
      f_prior``).

    The solve happens in the same Jacobi-normalized coordinates as
    `solve_fluxes_eigfloor` (``D`` from the UNregularized diagonal), and the
    eigenvalue floor is applied to the regularized normalized Gram — i.e.
    floor AFTER adding Lambda. Variances are ``diag((AtWA + Lambda)^{-1})``
    with the floored spectrum.

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Flux parameter vector, shape (n_flux,); only its length is used.
    image_data : dict
        Single-image data with keys ``data``, ``invvar`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    lambda_diag : jax.numpy.ndarray, optional
        Per-flux prior precisions ``1/sigma_prior^2``, shape (n_flux,);
        must be FINITE (map sigma -> 0 to a large finite value). ``None``
        means all zero (pure eigfloor). Build with
        :func:`tractor_jax.jax.batching.prior_arrays_from_slots`.
    f_prior : jax.numpy.ndarray, optional
        Per-flux prior means, shape (n_flux,); ignored where
        ``lambda_diag = 0``. ``None`` means all zero.
    return_variances : bool, optional
        If True, also return flux variances.
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the template
        renderer.
    floor : float, optional
        Relative eigenvalue floor, in units of the largest eigenvalue of
        the regularized normalized Gram matrix.

    Returns
    -------
    optimized_fluxes : jax.numpy.ndarray
        Solved fluxes, shape (n_flux,).
    variances : jax.numpy.ndarray
        Flux variances, shape (n_flux,). Only returned if
        ``return_variances`` is True.

    Notes
    -----
    Because the floor is relative to lambda_max of the REGULARIZED Gram, an
    extreme prior precision (``lambda_hat = lambda / AtWA_jj >> 1/floor``)
    inflates the floor felt by every other direction; keep ``lambda_diag``
    at physically motivated values (sigma_prior a fraction of the predicted
    flux) rather than using it to hard-pin sources.

    Dead slots (all-zero template, e.g. shape padding or a fully-masked
    source) are pinned to flux 0 with infinite variance regardless of any
    prior placed on them, so padding is prior-safe.
    """
    n_flux = initial_fluxes.shape[0]

    if lambda_diag is None:
        lambda_diag = jnp.zeros(n_flux, dtype=initial_fluxes.dtype)
    if f_prior is None:
        f_prior = jnp.zeros(n_flux, dtype=initial_fluxes.dtype)

    templates = _render_source_templates(image_data, batches, n_flux,
                                         sampling_factor=sampling_factor)

    data_flat = image_data["data"].ravel()
    w_flat = image_data["invvar"].ravel()
    A = templates.reshape(n_flux, -1).T

    Aw = A * w_flat[:, jnp.newaxis]
    AtWA = Aw.T @ A
    AtWd = Aw.T @ data_flat

    return _eigfloor_prior_core(AtWA, AtWd, lambda_diag, f_prior,
                                floor=floor,
                                return_variances=return_variances)


def _power_iter_lmax(G, n_steps=16):
    """
    Largest eigenvalue of a symmetric PSD matrix via power iteration.

    Deterministic init (ones vector) so results are reproducible under
    jit/vmap.

    Parameters
    ----------
    G : jax.numpy.ndarray
        Symmetric positive semi-definite matrix, shape (n, n).
    n_steps : int, optional
        Number of power-iteration steps.

    Returns
    -------
    jax.numpy.ndarray
        Rayleigh-quotient estimate of the largest eigenvalue (scalar).
    """
    n = G.shape[0]
    v0 = jnp.ones(n) / jnp.sqrt(n)

    def body(v, _):
        w = G @ v
        v_new = w / (jnp.linalg.norm(w) + 1e-30)
        return v_new, None

    v, _ = jax.lax.scan(body, v0, None, length=n_steps)
    return jnp.vdot(v, G @ v)


def lasso_fista(G, b, lam, *, nonneg=True, free=None, n_iter=1000, reg=0.0):
    """
    Per-coordinate L1-penalized quadratic solve on the normal equations.

    Solves::

        minimize  1/2 f^T (G + reg*I) f - b^T f + sum_j lam_j |f_j|
        subject to f_j >= 0 for penalized coordinates (if nonneg)

    Runs FISTA with gradient-scheme adaptive restart in Jacobi-normalized
    coordinates ``beta_j = sqrt(G_jj) * f_j`` (unit-diagonal system), which
    removes the dynamic-range part of the conditioning; with
    ``lam_j = alpha*sqrt(G_jj)`` the normalized threshold is uniform and
    dimensionless (S/N units).

    Parameters
    ----------
    G : jax.numpy.ndarray
        ``A^T W A`` of a whitened linear model, shape (n, n).
    b : jax.numpy.ndarray
        ``A^T W d`` of a whitened linear model, shape (n,).
    lam : jax.numpy.ndarray
        Absolute per-coordinate penalty ``lam_j``, shape (n,);
        ``lam_j = 0`` leaves coordinate j unpenalized.
    nonneg : bool, optional
        If True, penalized coordinates are constrained to ``f_j >= 0``.
    free : jax.numpy.ndarray, optional
        Coordinates with ``free[j] = 1`` are unpenalized AND sign-free
        (background). Defaults to all zeros.
    n_iter : int, optional
        Number of FISTA iterations.
    reg : float or jax.numpy.ndarray, optional
        Scalar or per-coordinate (n,) ridge vector (Jacobi-scaled
        ``reg_j = rcond * G_jj`` becomes exactly ``rcond`` on the
        normalized diagonal).

    Returns
    -------
    f : jax.numpy.ndarray
        The solution in original coordinates, shape (n,).
    kkt : jax.numpy.ndarray
        The maximum KKT violation in normalized (S/N) units - the
        convergence diagnostic.
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


# Back-compat alias (pre-public name).
_lasso_fista = lasso_fista

# Jitted entry: G/b/lam/free/reg are all traced arguments, so distinct
# problem values — including a Python-float ``reg``, which the RAW function
# bakes into its lax.scan traces as a fresh constant — share one compile per
# shape. Prefer this over the raw function for repeated solves with varying
# n or reg (pad n to a fixed bucket via tractor_jax.jax.batching.pad_normal_eq
# to make the shape fixed too).
lasso_fista_jit = jax.jit(lasso_fista, static_argnames=("nonneg", "n_iter"))


def _ln_binom(p, k):
    """
    Log binomial coefficient ``log C(p, k)`` via gammaln.

    Exact; valid for traced float ``k``.

    Parameters
    ----------
    p : float or jax.numpy.ndarray
        Total count.
    k : float or jax.numpy.ndarray
        Number chosen; may be a traced float.

    Returns
    -------
    jax.numpy.ndarray
        ``log C(p, k)``.
    """
    from jax.scipy.special import gammaln
    return gammaln(p + 1.0) - gammaln(k + 1.0) - gammaln(p - k + 1.0)


def solve_fluxes_lasso(initial_fluxes, image_data, batches,
                       return_variances=False, sampling_factor=None,
                       alpha=None, penalty_mode="snr", penalty_weights=None,
                       nonneg=True, selection_mode="fixed", criterion="ebic",
                       grid=None, ebic_gamma=0.5, return_path=False,
                       debias=True, debias_signfree="none", return_aux=False,
                       n_iter=1000, rcond=1e-12):
    """
    L1-regularized (LASSO) forced photometry on a SINGLE image.

    Designed to be vmapped, like `solve_fluxes_linear`. Builds the same
    design matrix A (unit-flux templates) and solves::

        min 1/2 || W^{1/2} (d - A f) ||^2 + sum_j lambda_j |f_j|,   f >= 0

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Flux parameter vector, shape (n_flux,); only its length is used.
    image_data : dict
        Single-image data with keys ``data``, ``invvar`` and ``psf``.
    batches : dict
        Per-image slices of the batched source data.
    return_variances : bool, optional
        If True, also return flux variances,
        ``diag(inv(AtWA_S + reg))`` of the debiased refit, inf off-support
        (conditional on the selected support - not post-selection
        corrected).
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the template
        renderer.
    alpha : float or "auto", optional
        Penalty strength; its meaning depends on ``penalty_mode``. Used
        directly when ``selection_mode="fixed"``. The string ``"auto"``
        applies the universal-threshold rule ``alpha = sqrt(2 ln p)`` with
        ``p`` the number of penalized live candidates in the solve —
        catalog-deterministic (never pixel-dependent) and validated
        in-basin across catalog depths (research note lasso_alpha/16 §6).
    penalty_mode : {"snr", "raw"}, optional
        Penalty parameterization (see proj-spherex-gpupipe research note
        notebooks/research_notes/lasso_alpha/01). ``"snr"``:
        ``lambda_j = alpha * w_j * sqrt(F_jj)``, with
        ``F_jj = diag(A^T W A)`` the squared matched-filter norm of
        template j, TAKEN FROM THE SAME TEMPLATES USED FOR THE SOLVE (not
        from `compute_fisher_diagonal`, which deviates in the
        oversampled-FFT path); ``alpha`` is then a dimensionless residual
        matched-filter S/N entry threshold, invariant across images, bands,
        depth and cutout size. ``"raw"``: ``lambda_j = alpha * w_j``
        (absolute units; sklearn conversion:
        ``lambda_raw = n_pix * alpha_sklearn``).
    penalty_weights : array_like, optional
        (n_flux,) per-source multiplier; 0 = PROTECTED source (never
        shrunk, never zeroed, always refit - use for the forced-photometry
        target list). None = ones. The background parameter (if fit) is
        always forced unpenalized and sign-free.
    nonneg : bool, optional
        If True, penalized coordinates are constrained to be non-negative.
    selection_mode : {"fixed", "path"}, optional
        ``"fixed"``: solve at the given ``alpha`` (production; inject the
        sim-calibrated value here). ``"path"``: solve on ``grid``, score
        with ``criterion`` on the DEBIASED refit, pick the argmin. No CV,
        deliberately: pixel folds violate independence and select overfit
        alphas (research note lasso_alpha/02 section 5).
    criterion : {"ebic", "bic", "sure"}, optional
        Path-selection criterion: ``"ebic"`` (default; exact
        ``2*gamma*ln C(p,k)`` multiplicity term), ``"bic"`` (gamma=0), or
        ``"sure"`` (biased RSS + 2*df - n_eff).
    grid : array_like, optional
        Alpha grid for ``selection_mode="path"`` (default
        ``logspace(0.5..5, 16)`` in S/N units).
    ebic_gamma : float, optional
        EBIC multiplicity weight gamma.
    return_path : bool, optional
        If True (path mode), include ``path_fluxes`` of shape
        (n_alpha, n_flux) in the aux dict.
    debias : bool, optional
        If True, exact re-solve on the selected support (identity pinning,
        static shapes), nonneg-clipped for penalized coordinates.
    debias_signfree : {"none", "protected", "all"}, optional
        Which refit coordinates skip the nonneg clip: ``"none"`` (default;
        background only, original behavior), ``"protected"`` (background +
        ``wj==0`` sources: protected SED/photo-z targets keep negative
        excursions - no faint-end rectification bias, research note
        lasso_alpha/12), or ``"all"``. Selection is unaffected (the FISTA
        prox still uses ``nonneg``); no-op when ``nonneg=False`` or
        ``debias=False``.
    return_aux : bool, optional
        If True, append the aux diagnostics dict to the returned tuple
        (see Returns).
    n_iter : int, optional
        Number of FISTA iterations.
    rcond : float, optional
        Jacobi-scaled ridge strength.

    Returns
    -------
    result
        Matching `solve_fluxes_linear` unless ``return_aux``:

        - ``fluxes`` if not ``return_variances``;
        - ``(fluxes, variances)`` if ``return_variances``;
        - ``(..., aux)`` if ``return_aux``, where ``aux`` carries
          ``support`` (float mask), ``alpha``, ``alpha_index``,
          ``criterion_values``, ``kkt`` (max KKT violation in S/N units -
          convergence diagnostic), ``n_active``, ``resid_corr_snr``
          (per-source residual matched-filter correlation at the biased
          solution, S/N units; entry margin = alpha - c_j for inactive
          sources), ``snr_deb`` (per-source debiased S/N; exit margin =
          snr - alpha for active ones - together the cheap production
          stability flag), and ``path_fluxes`` (n_alpha, n_flux) if
          ``return_path``.

    Notes
    -----
    With jax_enable_x64 off everything runs in float32; the Jacobi
    normalization inside `_lasso_fista` makes support recovery reliable in
    f32, but calibration-grade fluxes/variances should enable x64
    (``JaxOptimizer(enable_x64=True)``).

    **Not bit-reproducible run to run on GPU.** Unlike the linear/eigfloor
    solvers, which reproduce exactly, repeating a lasso solve on identical
    inputs and identical code can move a small number of fluxes: GPU reduction
    order is not fixed between runs, and near the sparsity threshold a
    coefficient that crosses zero flips the selected support discontinuously,
    so a rounding-level perturbation becomes a visible flux change.

    Measured on a full SPHEREx field (A2537, 85,415 measurements, two runs
    minutes apart): 0.27% of fluxes differ, and the largest difference is
    0.027 sigma — no measurement moves by even 0.1 sigma, so this is far below
    the noise and does not affect any downstream inference. It does mean a
    lasso product cannot be byte-compared against an earlier one; use a
    tolerance, and do not read a small lasso-vs-lasso difference as evidence of
    a code change. Forcing determinism (e.g. ``--xla_gpu_deterministic_ops``)
    serializes reductions and costs throughput, which is not worth paying for a
    0.03-sigma effect.
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
    """
    LASSO forced photometry for a BATCH of data realizations of ONE image.

    Bootstrap entry point (proj-spherex-gpupipe research note
    notebooks/research_notes/paper/03_bootstrap_validation_plan.md, stage
    B1): replicas share the scene — templates, invvar and the penalty
    structure — and differ only in the pixel data, so the design matrix and
    G = AtWA are built ONCE and the solve is vmapped over the replica axis
    (G, weights and the FISTA step size stay unbatched under vmap; only
    AtWd is per-replica).

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Flux parameter vector, shape (n_flux,); only its length is used.
    image_data : dict
        Single-image data; ``image_data["data"]`` is not used by the solve.
    batches : dict
        Per-image slices of the batched source data.
    data_stack : jax.numpy.ndarray
        (B, H, W) data realizations.
    return_variances, sampling_factor, alpha, penalty_mode, penalty_weights, nonneg, selection_mode, criterion, grid, ebic_gamma, return_path, debias, debias_signfree, return_aux, n_iter, rcond
        As `solve_fluxes_lasso`; shared by every replica (in "path" mode
        each replica selects its own alpha).

    Returns
    -------
    result
        Same structures as `solve_fluxes_lasso`, with a leading replica
        axis B on every array (fluxes: (B, n_flux); aux fields likewise).

    Notes
    -----
    Memory: the pinned refit materializes (n_flux, n_flux) per replica
    under vmap (B * n_flux^2; ~0.65 GB at B=100, n=900, f64 - x16 more in
    "path" mode). For larger runs split ``data_stack`` into chunks and call
    repeatedly; the per-call template build is the same work the chunks
    would share.
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
    """
    Per-source penalty multipliers ``wj`` and the sign-free mask.

    Background (if fit) is always unpenalized and sign-free.

    Parameters
    ----------
    n_flux : int
        Total number of flux parameters.
    batches : dict
        Batched source data; only the ``Background`` entry is inspected.
    penalty_weights : array_like or None
        Per-source penalty multipliers of shape (n_flux,); None means ones.
    dtype : dtype
        Dtype of the returned arrays.

    Returns
    -------
    wj : jax.numpy.ndarray
        Per-source penalty multipliers, shape (n_flux,).
    free : jax.numpy.ndarray
        Sign-free mask (1 for the background coordinate, if fit), shape
        (n_flux,).
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
    """
    LASSO solve on prebuilt normal equations (G = AtWA, b = AtWd).

    Shared backend of `solve_fluxes_lasso` (single image) and
    `solve_fluxes_lasso_batched` (many data realizations of one image);
    vmappable over ``(b, dWd)`` with ``G``/``wj``/``free`` held fixed.

    Parameters
    ----------
    G : jax.numpy.ndarray
        Normal matrix ``A^T W A``, shape (n_flux, n_flux).
    b : jax.numpy.ndarray
        Right-hand side ``A^T W d``, shape (n_flux,).
    dWd : jax.numpy.ndarray
        Scalar data quadratic term ``d^T W d`` (for the RSS).
    n_eff : jax.numpy.ndarray
        Effective number of pixels (count of positive weights).
    wj : jax.numpy.ndarray
        Per-source penalty multipliers, shape (n_flux,).
    free : jax.numpy.ndarray
        Sign-free/unpenalized mask (background), shape (n_flux,).
    alpha, penalty_mode, nonneg, selection_mode, criterion, grid, ebic_gamma, return_path, debias, debias_signfree, return_variances, return_aux, n_iter, rcond
        As `solve_fluxes_lasso`.

    Returns
    -------
    result
        Same structures as `solve_fluxes_lasso`.

    Notes
    -----
    Ridge is Jacobi-scaled, ``reg_j = rcond * G_jj``: invariant to masked
    padding and to the number of co-fit sources (in the FISTA-normalized
    coordinates it is exactly ``rcond * I`` on live slots). Dead slots
    (``G_jj = 0``) are excluded from the support and pinned in the refit.

    ``debias_signfree`` controls the sign convention of the DEBIASED refit
    only (selection keeps the ``nonneg`` prox): which coordinates skip the
    ``max(f, 0)`` clip. ``"none"`` (default) - background only (the
    ``free`` mask; original behavior); ``"protected"`` - background +
    unpenalized sources (``wj == 0``), so a protected science target keeps
    negative excursions (no faint-end rectification bias; research note
    lasso_alpha/12); ``"all"`` - every refit coordinate is sign-free.
    Irrelevant when ``nonneg=False`` (refit already unclipped) or
    ``debias=False``.
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
        if isinstance(alpha, str):
            if alpha != "auto":
                raise ValueError(f"alpha must be a number, None, or 'auto'; "
                                 f"got {alpha!r}")
            # universal-threshold rule alpha = sqrt(2 ln p), p = number of
            # PENALIZED live candidates in this solve. Deterministic in the
            # prior catalog (never in the pixel data), so it cannot couple
            # the selection to the noise, and it keeps the noise-only
            # false-entry rate uniform across problems of different catalog
            # depth (validated in-basin on 9 SPHEREx sim fields; proj
            # research note lasso_alpha/16 §6).
            p = jnp.sum((wj > 0) & live)
            a = jnp.sqrt(2.0 * jnp.log(jnp.maximum(p, 2).astype(G.dtype)))
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

    Parameters
    ----------
    initial_fluxes : jax.numpy.ndarray
        Initial flux parameter vector, shape (N_flux,).
    image_data : dict
        Single image data (slices).
    batches : dict
        Batched source data (slices).
    return_variances : bool, optional
        If True, also return flux variances (inverse Fisher diagonal).
    sampling_factor : float, optional
        High-resolution oversampling factor forwarded to the renderer.
    use_preconditioner : bool, optional
        If True, precondition CG with the inverse Fisher diagonal.
    precond_eps : float, optional
        Floor for the Fisher diagonal to avoid division by zero.

    Returns
    -------
    optimized_fluxes : jax.numpy.ndarray
        Optimized fluxes, shape (N_flux,).
    variances : jax.numpy.ndarray
        Flux variances (inverse Fisher diagonal), shape (N_flux,). Only
        returned if ``return_variances`` is True.
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
    Optimize fluxes for forced photometry using JAX.

    Parameters
    ----------
    tractor_obj : Tractor
        Tractor object with images and catalog.
    oversample_rendering : bool, optional
        If True, use oversampled rendering for ``PixelizedPSF`` with
        sampling != 1.
    return_variances : bool, optional
        If True, return variances of fluxes.
    fit_background : bool, optional
        If True, includes a background level in the optimization
        parameters.
    update_catalog : bool, optional
        If True, updates the source catalog with optimized fluxes.
    vmap_images : bool, optional
        If True (default), stacks all images and processes them in a
        single vmap call.
    use_sharding : bool, optional
        If True (default), distributes the batch across available devices.
    bucket_sizes : list, optional
        List of bucket sizes. Used if ``bucket_mode="fixed"``.
    bucket_mode : {"auto", "fixed"}, optional
        Bucket determination mode.
    bucket_shape_mode : {"square", "independent"}, optional
        Bucket shape mode.
    bucket_base : int, optional
        Rounding base for auto mode.
    use_tiling : bool, optional
        If True, splits each image into core tiles padded by a PSF-sized
        halo, solves every tile independently with a compact per-tile flux
        vector, and merges the results back into one catalog-layout vector
        per image: each source is read from the tile whose core box
        contains it (halo overlaps never double-count), and the background
        (if fit) is reported as the core-area weighted mean of the per-tile
        backgrounds. The return format matches the non-tiled path;
        ``return_aux`` is not supported.
    tile_size : int, optional
        Size of tiles (default 256).
    tile_super_halo : int, optional
        Override the calculated halo size.
    use_preconditioner : bool, optional
        If True (default), use the Fisher diagonal preconditioner (CG
        only).
    precond_eps : float, optional
        Floor for the Fisher diagonal to avoid divide-by-zero.
    solver : {"linear", "eigfloor", "cg", "lasso"}, optional
        "linear" (direct normal-equations solve), "eigfloor" (linear solve
        with an eigenvalue floor on AtWA - sign-free, no selection; see
        `solve_fluxes_eigfloor`), "cg" (Newton-CG, original), or "lasso"
        (L1-regularized, see `solve_fluxes_lasso`).
    penalty : dict, optional
        (lasso only) E.g. ``{"mode": "snr", "alpha": 2.0, "weights": w,
        "nonneg": True}``. ``weights==0`` marks protected sources.
    selection : dict, optional
        (lasso only) E.g. ``{"mode": "fixed"}`` or ``{"mode": "path",
        "criterion": "ebic", "grid": ..., "ebic_gamma": 0.5,
        "return_path": False}``.
    debias : bool, optional
        (lasso only) Refit on the selected support (default True).
    debias_signfree : {"none", "protected", "all"}, optional
        (lasso only) Which refit coordinates skip the nonneg clip -
        "none" (default), "protected" (background + weight-0 sources), or
        "all". See `solve_fluxes_lasso`.
    return_aux : bool, optional
        (lasso only) Append the per-image aux dict (support, alpha, kkt,
        ...) as the last element of each result.
    lasso_n_iter : int, optional
        (lasso only) Static FISTA iteration count.
    eig_floor : float, optional
        (eigfloor only) Relative eigenvalue floor, in units of the largest
        eigenvalue of AtWA (default 1e-4).

    Returns
    -------
    list
        List of results per image.
    """
    from tractor_jax.sky import ConstantSky

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
        all_meta = []
        pos_cat_per_img = []

        if _lasso_aux:
            raise NotImplementedError(
                "return_aux is not supported in tiling mode")

        for i_img, img in enumerate(tractor_obj.images):
            # Project catalog to this image's pixel coords
            pos_cat = project_catalog(tractor_obj.catalog, img.getWcs())
            pos_cat_per_img.append(pos_cat)

            tiles_with_meta = tile_image(img, tile_size, halo)

            for (tile_img, meta) in tiles_with_meta:
                indices = filter_sources_by_box(
                    pos_cat,
                    meta['x_start'], meta['x_end'],
                    meta['y_start'], meta['y_end'],
                    margin=0 # Halo already included in start/end
                )

                # Keep tiles with no sources: the background can still be fit.
                all_tiles.append(tile_img)
                all_meta.append(meta)
                all_indices.append(indices)
                original_img_indices.append(i_img)

        # 3. Bucket Tiles
        stats = compute_target_stats(all_tiles, oversample_rendering)
        max_factor = stats["max_factor"]
        req_shapes = compute_image_shapes(all_tiles, stats)

        bucket_map = assign_buckets(req_shapes, bucket_sizes, bucket_mode, bucket_shape_mode, bucket_base)

        print(f"JAX Optimization: {len(all_tiles)} tiles -> {len(bucket_map)} buckets")

        # Per-tile solved rows and slot maps, in 'all_tiles' order
        tile_fluxes = [None] * len(all_tiles)
        tile_vars = [None] * len(all_tiles)
        tile_slots = [None] * len(all_tiles)

        for shape, tile_idxs in bucket_map.items():
            if not tile_idxs:
                continue

            sub_tiles = [all_tiles[i] for i in tile_idxs]
            sub_source_indices = {k: all_indices[original_idx] for k, original_idx in enumerate(tile_idxs)}

            # sub_tractor needs same catalog
            sub_tractor = Tractor(sub_tiles, tractor_obj.catalog)

            # Compact per-tile flux rows: each tile solves only its own
            # sources, so the solver's normal matrix stays
            # (n_tile_src)^2 instead of (n_catalog)^2.
            images_data, batches, initial_fluxes, bucket_slots = extract_model_data(
                sub_tractor,
                oversample_rendering=oversample_rendering,
                fit_background=fit_background,
                fixed_target_shape=shape,
                fixed_max_factor=max_factor,
                img_source_indices=sub_source_indices,
                compact_fluxes=True,
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
                # Background flux_idx is a row-relative scalar identical for
                # every image (see extract_model_data), so it is not vmapped.
                batches_in_axes["Background"] = {"flux_idx": None}

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
            if return_variances:
                optimized_fluxes_stack, variances_stack = out
            else:
                optimized_fluxes_stack = out

            res_fluxes = np.array(optimized_fluxes_stack)
            if return_variances:
                res_variances = np.array(variances_stack)

            for k, original_idx in enumerate(tile_idxs):
                tile_fluxes[original_idx] = res_fluxes[k]
                if return_variances:
                    tile_vars[original_idx] = res_variances[k]
                tile_slots[original_idx] = bucket_slots[k]

        # 4. Merge the per-tile solutions into one catalog-layout flux
        # vector per image. Each source is read from the tile whose CORE box
        # contains its position, so halo overlaps never double-count a
        # source; sources projecting outside every core fall back to the
        # nearest core centre.
        catalog = tractor_obj.catalog
        global_offsets = {}
        g_off = 0
        for ci, src in enumerate(catalog):
            if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
                continue
            if hasattr(src, "brightness"):
                n_p = len(src.brightness.getParams())
                global_offsets[ci] = (g_off, n_p)
                g_off += n_p
        n_src_params = g_off

        tiles_of_img = defaultdict(list)
        for t_i, i_img in enumerate(original_img_indices):
            tiles_of_img[i_img].append(t_i)

        results = []
        for i_img in range(len(tractor_obj.images)):
            t_list = tiles_of_img[i_img]
            metas = [all_meta[t] for t in t_list]
            core_cx = np.array([m['x0'] + 0.5 * m['core_w'] for m in metas])
            core_cy = np.array([m['y0'] + 0.5 * m['core_h'] for m in metas])

            n_flux = n_src_params + (1 if fit_background else 0)
            merged_f = np.zeros(n_flux, dtype=np.float32)
            merged_v = np.zeros(n_flux, dtype=np.float32) if return_variances else None

            pos_cat = pos_cat_per_img[i_img]
            for ci, (f_off, n_p) in global_offsets.items():
                x, y = pos_cat[ci]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue  # unprojectable source: left at 0
                owner = None
                for k, m in enumerate(metas):
                    if (m['x0'] <= x < m['x0'] + m['core_w']
                            and m['y0'] <= y < m['y0'] + m['core_h']):
                        owner = k
                        break
                if owner is None:
                    owner = int(np.argmin((core_cx - x) ** 2
                                          + (core_cy - y) ** 2))
                t_i = t_list[owner]
                slot = tile_slots[t_i].get(ci)
                if slot is None:
                    continue  # outside the owner tile's padded box: left at 0
                s_off = slot[0]
                merged_f[f_off:f_off + n_p] = tile_fluxes[t_i][s_off:s_off + n_p]
                if return_variances:
                    merged_v[f_off:f_off + n_p] = tile_vars[t_i][s_off:s_off + n_p]

            if fit_background:
                # Tile backgrounds are independent fits; report the
                # core-area weighted mean (variance likewise, approximate).
                w = np.array([m['core_w'] * m['core_h'] for m in metas],
                             dtype=np.float64)
                w = w / w.sum()
                bgs = np.array([tile_fluxes[t][-1] for t in t_list])
                merged_f[-1] = float((w * bgs).sum())
                if return_variances:
                    bgv = np.array([tile_vars[t][-1] for t in t_list])
                    merged_v[-1] = float((w * bgv).sum())

            results.append((merged_f, merged_v) if return_variances else merged_f)

        if update_catalog:
            if len(tractor_obj.images) == 1:
                f_vec = results[0][0] if return_variances else results[0]
                ptr = 0
                for src in tractor_obj.catalog:
                    if isinstance(src, (CompositeGalaxy, FixedCompositeGalaxy)):
                        continue
                    if hasattr(src, "brightness"):
                        n = src.brightness.numberOfParams()
                        src.brightness.setParams(f_vec[ptr:ptr + n])
                        ptr += n
            else:
                print("Warning: update_catalog=True but N_img > 1. Catalog not updated to avoid ambiguity.")

        return results

    elif vmap_images:
        # Determine buckets
        stats = compute_target_stats(tractor_obj.images, oversample_rendering)
        max_factor = stats["max_factor"]
        req_shapes = compute_image_shapes(tractor_obj.images, stats)
        bucket_map = assign_buckets(req_shapes, bucket_sizes, bucket_mode, bucket_shape_mode, bucket_base)

        print(f"JAX Optimization: {len(tractor_obj.images)} images -> {len(bucket_map)} buckets")
        for shape, idxs in bucket_map.items():
            print(f"  Bucket {shape}: {len(idxs)} images")

        all_results = [None] * len(tractor_obj.images)

        # (N_img, N_params); rebuilt from all_results after the bucket loop
        # for the update_catalog logic below.
        optimized_fluxes_np = None

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
            # flux_idx, shapes and profiles are per-image arrays
            # (N_img, N_src, ...), so they are mapped with in_axes=0.
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
             optimized_fluxes_np = np.array([])
             if return_variances:
                 variances_np = np.array([])

    else:
        # Sequential Processing: images one by one to save memory,
        # collecting results in the same format as the vmap path.
        fluxes_list = []
        variances_list = []

        batches = {} # Initialize in case loop doesn't run, to avoid UnboundLocalError for bg check

        for _img_i, img in enumerate(tractor_obj.images):
            # extract_model_data works on Tractor objects; wrap the single image
            sub_tractor = Tractor([img], tractor_obj.catalog)

            img_data, batches, init_flux = extract_model_data(
                sub_tractor,
                oversample_rendering=oversample_rendering,
                fit_background=fit_background
            )

            # img_data is stacked with shape (1, ...). We unbatch.
            single_data = jax.tree_util.tree_map(lambda x: x[0], img_data)

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
                # Background flux_idx is row-relative, not image-batched;
                # nothing to slice.
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
            f_vec = optimized_fluxes_np[0] # Single image

            if "PointSource" in batches:
                # Fluxes are packed [src1_params, src2_params, ...] in catalog
                # iteration order with the same filtering as extract_model_data
                # (composites skipped), so re-iterating the catalog in order
                # recovers the source <-> flux mapping.
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
        Perform one optimization step using JAX.

        Runs `optimize_fluxes` with ``update_catalog=True``,
        ``fit_background=True`` and ``oversample_rendering=True``, then
        reports the change in log-probability and in the parameter vector.

        Parameters
        ----------
        tractor : Tractor
            Tractor object; its catalog is updated in place.
        alphas, damp, priors, scale_columns, shared_params, just_variance
            Accepted for API compatibility with the base ``Optimizer``;
            not used by the JAX path.
        variance : bool, optional
            If True, also return a variance vector for the parameters.
        vmap_images : bool, optional
            Forwarded to `optimize_fluxes`.
        use_sharding : bool, optional
            Forwarded to `optimize_fluxes`.
        **kwargs
            Forwarded to `optimize_fluxes`.

        Returns
        -------
        dlnp : float
            Change in log-probability.
        X : numpy.ndarray
            Parameter update vector.
        alpha : float
            Step scale (always 1.0).
        var : numpy.ndarray or None
            Parameter variances (flux parameters only; may not match the
            full parameter vector if positions are thawed). Only returned
            if ``variance`` is True.
        """
        lnp0 = tractor.getLogProb()
        p0 = tractor.getParams()

        # oversample_rendering=True is a safe default: it only takes effect
        # for undersampled PixelizedPSFs (sampling < 1).
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

        p1 = tractor.getParams()
        lnp1 = tractor.getLogProb()
        dlnp = lnp1 - lnp0
        X = np.array(p1) - np.array(p0)

        alpha = 1.0

        if variance:
            if len(res) == 1:
                fluxes, vars = res[0]
                # vars covers FLUX parameters only, while X spans every thawed
                # tractor parameter (tractor orders thawed image params, then
                # catalog params); a robust flux -> full-parameter mapping
                # would need extract_model_data to export one. In the simple
                # case (sky and positions fixed, fluxes thawed) the lengths
                # match and vars is the full vector; otherwise vars is passed
                # through unchanged (length mismatch if positions are thawed).
                full_var = vars
            else:
                full_var = None

            return dlnp, X, alpha, full_var

        return dlnp, X, alpha

    def optimize_loop(self, tractor, dchisq=0., steps=50, **kwargs):
        # Run single step as JAX CG solves it
        return self.optimize(tractor, **kwargs)
