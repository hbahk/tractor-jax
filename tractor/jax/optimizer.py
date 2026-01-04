import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import jit, value_and_grad
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


# Helper to extract data from Tractor objects
def extract_model_data(tractor_obj):
    """
    Extracts all necessary data from a Tractor object for JAX optimization.

    Returns:
        images_data: list of dicts (data, invvar, wcs_cd, psf_data)
        sources_data: list of dicts (type, params, profile_mix)
        param_mapping: list of (source_idx, param_name, slice)
    """
    images = tractor_obj.images
    catalog = tractor_obj.catalog

    # Extract Image Data
    images_data = []
    for img in images:
        h, w = img.shape
        data = jnp.array(img.getImage())
        invvar = jnp.array(img.getInvError()) ** 2

        # WCS (assuming simple CD matrix for now, usually at center or per source)
        # For small regions, one CD at center is enough. For large, we need more.
        # Here we will assume we evaluate CD at source position during rendering
        # but since we want to vectorize, maybe we precompute CD per source per image.
        # But `extract_model_data` prepares static data.
        # Let's store the WCS object wrapper or sufficient info.
        # For this implementation, let's assume constant WCS across the stamp/image
        # or that we can get CD at any pixel.
        # But JAX needs arrays.
        # Let's assume we optimize on small stamps (ROIs).
        # Or if full image, we might need a map of CD.
        # For simplicity, we assume we extract ROI for each source or group of sources.
        # BUT the user said "forced photometry... fixed positions".
        # So CD at source position is constant.

        wcs = img.getWcs()

        psf = img.getPsf()
        psf_data = {}
        if isinstance(psf, PixelizedPSF):
            psf_data["type"] = "pixelized"
            psf_data["img"] = jnp.array(psf.img)  # The kernel
            # PixelizedPSF usually has `sampling` etc. but `img` is the kernel.
            # We need the FFT of the PSF for Galaxy rendering.
            # We can precompute it if image size is fixed.
            # But image size depends on source/ROI.
            # Let's store raw image and compute FFT on fly or cache.
        elif isinstance(psf, GaussianMixturePSF):
            psf_data["type"] = "mog"
            # (amp, mean, var)
            # Tractor MoG: amp (K), mean (K,2), var (K,2,2)
            mog = psf.mog
            psf_data["amp"] = jnp.array(mog.amp)
            psf_data["mean"] = jnp.array(mog.mean)
            psf_data["var"] = jnp.array(mog.var)
            # print("EXTRACTED MOG PSF")

        images_data.append(
            {
                "data": data,
                "invvar": invvar,
                "wcs": wcs,  # Keep python object to query CD
                "psf": psf_data,
                "shape": (h, w),
            }
        )

    # Extract Source Data
    sources_data = []
    initial_fluxes = []

    for i, src in enumerate(catalog):
        src_info = {}
        src_type = src.getSourceType()
        src_info["type"] = src_type

        # Position
        pos = src.pos.getParams()
        src_info["pos"] = jnp.array(pos)

        # Brightness (Flux)
        # Assuming single band or handled by brightness object.
        # For forced photometry, we want to optimize this.
        # src.brightness.getParams() returns array of fluxes (one per band usually?)
        # Or just one value for PointSource/SimpleGalaxy.
        br = src.brightness.getParams()
        src_info["flux_idx"] = len(initial_fluxes)
        initial_fluxes.extend(br)
        src_info["n_flux"] = len(br)

        # Shapes & Profiles
        if isinstance(src, Galaxy):
            # Exp, Dev, Composite
            shape = src.shape.getParams()  # re, ab, phi
            src_info["shape"] = jnp.array(shape)
            # print(f"DEBUG: Source {i} type {src_type} shape {shape}")

            if isinstance(src, (ExpGalaxy, DevGalaxy)):
                # Get mixture
                # src.getProfile() returns MoG
                profile = src.getProfile()
                src_info["profile"] = {
                    "amp": jnp.array(profile.amp),
                    "mean": jnp.array(profile.mean),
                    "var": jnp.array(profile.var),
                }
            # Handle other subclasses that are effectively SingleProfileSource
            elif hasattr(src, "getProfile") and src.getProfile() is not None:
                profile = src.getProfile()
                src_info["profile"] = {
                    "amp": jnp.array(profile.amp),
                    "mean": jnp.array(profile.mean),
                    "var": jnp.array(profile.var),
                }
                # print(f"Added profile for {src_type}")
                # Treat as ProfileGalaxy/ExpGalaxy equivalent for rendering
                if src_type not in ["ExpGalaxy", "DevGalaxy"]:
                    # Map to generic handling in render loop
                    # Or rely on src_type check.
                    pass
            elif isinstance(src, CompositeGalaxy):
                # Handle composite...
                pass

        sources_data.append(src_info)

    return images_data, sources_data, jnp.array(initial_fluxes)


def render_scene(fluxes, images_data, sources_data):
    """
    Renders the scene for all images.

    Args:
        fluxes: JAX array of all source fluxes.
        images_data: Static data for images.
        sources_data: Static data for sources (positions, shapes, profiles).

    Returns:
        List of model images (one per image_data).
    """
    model_images = []

    for img_idx, img_dat in enumerate(images_data):
        H, W = img_dat["shape"]
        model = jnp.zeros((H, W))

        # Loop over sources
        # (This loop is python unrolled, which is fine for moderate N sources)
        for src in sources_data:
            # Get flux for this source
            f_idx = src["flux_idx"]
            # Assuming 1 flux per source for simplicity here,
            # or we need to map image band to flux index.
            # In simple Tractor usage, brightness params align with bands if MultiParams.
            # But let's assume single band for this demo or just take first flux.
            flux = fluxes[f_idx]

            # WCS at source position
            # We need to call wcs object. This breaks JIT if WCS is complex.
            # For this task, we can assume we extracted CD matrix beforehand if we want JIT.
            # BUT: We are inside JAX function now. We cannot call python `img_dat['wcs'].cdAtPixel`.
            # We must pre-compute CD for all sources in `extract_model_data` or pass it in `sources_data`.

            # Let's fix this by computing CD in `extract_model_data` and storing in `sources_data` per image.
            # See implementation below.

            wcs_cd_inv = src["wcs_cd_inv"][img_idx]
            pos_pix = src["pos_pix"][img_idx]  # (x, y) on this image

            # Subpixel offset
            # pos_pix is float.
            # We render centered at integer pixel?
            # The rendering functions take full offset or subpixel?
            # `render_pixelized_psf` takes (dx, dy) and shifts.
            # `render_galaxy` centers at 0 and adds `pos`.

            # Wait, `render_pixelized_psf` shifts an image.
            # The image should be placed at the integer pixel location.
            # But JAX arrays are fixed size. We cannot "place" a small stamp into a large image efficiently
            # without `jax.lax.dynamic_update_slice`.
            # And `render_galaxy` returns full image? No, that would be wasteful.

            # Optimization strategy:
            # 1. We render a stamp centered on the source.
            # 2. We add this stamp to the full model image.

            # But `render_galaxy_fft` does IFFT on full image size?
            # "psf_fft: (H, W) Complex Fourier Transform of the PSF (padded to image_shape)."
            # Yes, standard FFT convolution requires padding to image size (or at least stamp size).

            # If we do full image FFT, it's slow for many sources.
            # Usually we do stamps.

            # For this JAX implementation, let's assume we render into full image for simplicity,
            # or assume the "image" passed in is actually a cutout (ROI) around the source.

            # The user provided `jax_optimize.py` rendered into `image_data_shape`.

            # Let's implement full image rendering for now (simple correctness).

            src_type = src["type"]

            if src_type == "PointSource":
                if img_dat["psf"]["type"] == "pixelized":
                    # We assume psf_img is centered.
                    # We need to shift it to pos_pix.
                    # pos_pix = (x, y).
                    # Center of PSF image is at (pH//2, pW//2).
                    # We want center at (x, y).
                    # Shift = (x - pH//2, y - pW//2)?

                    # No, `render_pixelized_psf` takes `dx, dy` (subpixel) and returns shifted PSF kernel.
                    # Then we need to place it.

                    # Actually `render_pixelized_psf` shifts the array elements.
                    # If we pass the FULL size PSF image (padded), we can shift it to position.
                    # But usually we have a small PSF stamp.

                    # For JAX, maybe it's better to implement `add_at` (scatter_add).

                    psf_img = img_dat["psf"]["img"]
                    pH, pW = psf_img.shape

                    # To render into full model image (H, W):
                    # 1. Pad PSF kernel to (H, W).
                    # 2. Shift to pos_pix.

                    # Create padded PSF image centered at (pH//2, pW//2) inside (H, W) or at (0,0)?
                    # render_pixelized_psf shifts an input image by (dx, dy).
                    # If we input an image with PSF at some location (sx, sy), output has PSF at (sx+dx, sy+dy).

                    # We want final PSF at pos_pix = (x, y).
                    # Let's put PSF at (0, 0) (top-left) or center of image?
                    # Using fftshift convention might be confusing with lanczos.

                    # Let's start with PSF at center of kernel: (pH//2, pW//2).
                    # We place this kernel into a large zero image at a reference position.
                    # Say reference position (cx, cy) = (pH//2, pW//2).
                    # We want to shift to (x, y).
                    # Shift = (x - cx, y - cy).

                    # Use FFT based rendering with phase shift for exact positioning globally
                    psf_fft = img_dat["psf_fft_full"]
                    stamp = render_point_source_fft(flux, pos_pix, psf_fft, (H, W))
                    model = model + stamp
                elif img_dat["psf"]["type"] == "mog":
                    # MoG rendering
                    stamp = render_point_source_mog(
                        flux,
                        pos_pix,
                        (
                            img_dat["psf"]["amp"],
                            img_dat["psf"]["mean"],
                            img_dat["psf"]["var"],
                        ),
                        (H, W),
                    )
                    model = model + stamp

            elif (
                src_type
                in ["Galaxy", "ExpGalaxy", "DevGalaxy", "HoggGalaxy", "ProfileGalaxy"]
                or "profile" in src
            ):
                shape = src["shape"]
                # if 'profile' not in src, skip?
                if "profile" not in src:
                    continue
                profile = src["profile"]  # (amp, mean, var)

                # Check for ProfileGalaxy subclass?
                # extract_model_data puts 'profile' in src info only if instance is ExpGalaxy/DevGalaxy
                # In test we use ExpGalaxy.

                if img_dat["psf"]["type"] == "pixelized":
                    # FFT convolution
                    psf_fft = img_dat["psf_fft_full"]

                    gal_mix = (profile["amp"], profile["mean"], profile["var"])

                    stamp = render_galaxy_fft(
                        gal_mix, psf_fft, shape, wcs_cd_inv, pos_pix, (H, W)
                    )
                    model = model + flux * stamp

                elif img_dat["psf"]["type"] == "mog":
                    gal_mix = (profile["amp"], profile["mean"], profile["var"])
                    psf_mix = (
                        img_dat["psf"]["amp"],
                        img_dat["psf"]["mean"],
                        img_dat["psf"]["var"],
                    )

                    # print(f"Rendering Galaxy MoG: Flux {flux} Pos {pos_pix}")
                    stamp = render_galaxy_mog(
                        gal_mix, psf_mix, shape, wcs_cd_inv, pos_pix, (H, W)
                    )
                    # print(f"Stamp Sum: {jnp.sum(stamp)}")
                    model = model + flux * stamp

        model_images.append(model)

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
    images_data, sources_data, initial_fluxes = extract_model_data(tractor_obj)

    # 2. Add Precomputations (WCS CD, PSF FFT)
    for i, img_dat in enumerate(images_data):
        wcs = img_dat["wcs"]
        H, W = img_dat["shape"]

        # Precompute PSF FFT if needed
        if img_dat["psf"]["type"] == "pixelized":
            psf_img = img_dat["psf"]["img"]
            # Pad PSF to image size
            # Assuming PSF is small odd kernel.
            # We place it at (0,0) (top-left) for FFT, handling wrap-around?
            # Or centered?
            # Standard FFT convolution: Pad kernels to size (H, W).
            # If PSF is centered at its own center, we should shift it to (0,0) before FFT to avoid phase shift?
            # Or just handle phase shift later.

            # Simple approach: Pad to (H, W), keeping center at center.
            # Then fftshift before fft?
            # `np.fft.rfft2` expects (0,0) at index (0,0).
            # So we should `ifftshift` the centered PSF.

            ph, pw = psf_img.shape
            pad_img = jnp.zeros((H, W))
            # Place at center
            cy, cx = H // 2, W // 2
            y0 = cy - ph // 2
            x0 = cx - pw // 2
            pad_img = pad_img.at[y0 : y0 + ph, x0 : x0 + pw].set(psf_img)

            # Shift so center is at (0,0)
            pad_img = jnp.fft.ifftshift(pad_img)

            psf_fft = jfft.rfft2(pad_img)
            img_dat["psf_fft_full"] = psf_fft

        # Precompute per-source data on this image
        for src in sources_data:
            # Position on this image
            if "pos_pix" not in src:
                src["pos_pix"] = []
                src["wcs_cd_inv"] = []

            # We need to call WCS methods which might not be JAX-traceable if inside JIT.
            # But we are in python here.
            # src['pos'] is JAX array, but we can convert to numpy for WCS call.
            ra, dec = src["pos"]  # Tractor pos is usually RaDecPos
            # Tractor `pos.getParams()` returns `(ra, dec)`.

            # Map (ra, dec) to (x, y)
            # We need the `src` object from catalog to pass to positionToPixel?
            # Wait, `extract_model_data` lost the objects.
            # But `tractor_obj.catalog` is available.
            # `sources_data` aligns with `catalog`.

            # We can use `wcs.positionToPixel(src_obj.getPosition())`
            # But we already extracted params.

            # Let's assume RaDecPos.
            # `img.getWcs().positionToPixel(pos, src)`
            # We need to perform this mapping.
            # Since `optimize_fluxes` is the entry point (Python), we can do this loop.

            # But wait, `src['pos']` is a JAX array from `extract_model_data`.
            # We should probably do this IN `extract_model_data` where we have the objects.
            pass

    # Re-do extraction with more detail
    images = tractor_obj.images
    catalog = tractor_obj.catalog

    # Update sources_data with WCS info
    for i, src_info in enumerate(sources_data):
        src_obj = catalog[i]
        src_info["pos_pix"] = []
        src_info["wcs_cd_inv"] = []

        for img in images:
            wcs = img.getWcs()
            # Pos
            x, y = wcs.positionToPixel(src_obj.getPosition(), src_obj)
            src_info["pos_pix"].append(jnp.array([x, y]))

            # CD Inv
            cdinv = wcs.cdInverseAtPixel(x, y)
            src_info["wcs_cd_inv"].append(jnp.array(cdinv))

        src_info["pos_pix"] = jnp.stack(src_info["pos_pix"])  # (N_img, 2)
        src_info["wcs_cd_inv"] = jnp.stack(src_info["wcs_cd_inv"])  # (N_img, 2, 2)

    # 3. Define Loss Function
    def loss_fn(fluxes):
        model_images = render_scene(fluxes, images_data, sources_data)
        chi2 = 0.0
        for i, model in enumerate(model_images):
            data = images_data[i]["data"]
            invvar = images_data[i]["invvar"]
            diff = data - model
            chi2 += jnp.sum(diff**2 * invvar)
        return chi2

    # 4. Optimize
    # Use Matrix-Free Newton-CG for linear least squares.
    # Since the problem is linear in fluxes, the loss is quadratic.
    # We solve H * delta_x = -grad using CG.

    # 1. Compute Gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(initial_fluxes)

    # 2. Define Hessian-Vector Product (HVP)
    # The Hessian of f(x) is the Jacobian of grad(f)(x).
    # HVP(v) = J(grad_f)(x) * v
    # In JAX, we can use jvp to compute forward-mode derivatives.
    # jvp(fun, primals, tangents) -> (primals_out, tangents_out)
    # where tangents_out is Jacobian * tangents.

    def matvec(v):
        # We compute H*v at the initial point (any point works for quadratic/linear problem)
        # return jax.jvp(grad_fn, (initial_fluxes,), (v,))[1]

        # Alternatively, for linear least squares specifically:
        # Loss = ||Ax - b||^2
        # Grad = 2 A^T (Ax - b)
        # Hessian = 2 A^T A
        # H*v = 2 A^T (A * v)
        # We can implement this using jvp on the residual function if we exposed it.
        # But jvp on grad is generic and works.
        return jax.jvp(grad_fn, (initial_fluxes,), (v,))[1]

    # 3. Solve H * step = -grads using CG
    # jax.scipy.sparse.linalg.cg(A, b, x0=None, tol=1e-05, atol=0.0, maxiter=None, M=None)
    # A can be a function.

    step, info = jax.scipy.sparse.linalg.cg(
        matvec, -grads, maxiter=50
    )  # 50 steps usually enough for convergence

    # 4. Apply Update
    # For a purely quadratic function, one Newton step is exact.
    optimized_fluxes = initial_fluxes + step

    # 5. Update Tractor object
    # Copy tractor object?
    # Or update in place.

    # We must ensure optimized_fluxes is a numpy array (cpu) before setting params
    # because tractor objects expect numpy usually.
    optimized_fluxes_np = np.array(optimized_fluxes)

    flux_idx = 0
    for src in catalog:
        n_flux = len(src.brightness.getParams())
        new_flux = optimized_fluxes_np[flux_idx : flux_idx + n_flux]
        src.brightness.setParams(new_flux)
        flux_idx += n_flux

    return tractor_obj
