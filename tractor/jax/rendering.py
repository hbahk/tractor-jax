import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax import vmap

from tractor.miscutils import lanczos_filter, batch_correlate1d


def get_galaxy_shape_matrix(re, ab, phi):
    """
    Computes the transformation matrix G that takes unit vectors (in re)
    to degrees (intermediate world coords).

    Args:
        re: Effective radius in arcsec (scalar or array)
        ab: Axis ratio (scalar or array)
        phi: Position angle in degrees (scalar or array)

    Returns:
        G: Matrix of shape (..., 2, 2)
    """
    # Phi is E of N.
    # 0 = N (Dec increasing)
    # 90 = E (RA increasing)
    phi_rad = jnp.deg2rad(90.0 - phi)
    # HACK -- bring up to a minimum size to prevent singular matrix inversions
    # Matching tractor/galaxy.py logic
    re_deg = jnp.maximum(1.0 / 30.0, re) / 3600.0

    c = jnp.cos(phi_rad)
    s = jnp.sin(phi_rad)

    # G = re_deg * [[cp, sp * ab], [-sp, cp * ab]]
    # Shape construction
    # Note: re_deg might be scalar.

    # In tractor/galaxy.py:
    # return re_deg * np.array([[cp, sp * self.ab], [-sp, cp * self.ab]])

    # Correct stacking.
    # If re, ab, phi are scalars, we want (2, 2).
    # If arrays (N,), we want (N, 2, 2).

    row1 = jnp.stack([c, s * ab], axis=-1)
    row2 = jnp.stack([-s, c * ab], axis=-1)
    mat = jnp.stack([row1, row2], axis=-2)  # (..., 2, 2)

    G = re_deg[..., jnp.newaxis, jnp.newaxis] * mat
    return G


def get_shear_matrix(cd_inv, G):
    """
    Computes Tinv = cd_inv . G

    Args:
        cd_inv: Inverse CD matrix (..., 2, 2) (degrees -> pixels)
        G: Galaxy shape matrix (..., 2, 2) (unit re -> degrees)

    Returns:
        Tinv: Transformation matrix (..., 2, 2) (unit re -> pixels)
    """
    # Matrix multiplication
    return jnp.matmul(cd_inv, G)


def apply_shear_to_cov(cov, Tinv):
    """
    Applies shear Tinv to covariance matrix cov.
    New cov = Tinv * cov * Tinv^T

    Args:
        cov: Covariance matrices (..., K, 2, 2)
        Tinv: Shear matrices (..., 2, 2)

    Returns:
        New covariance matrices (..., K, 2, 2)
    """
    # cov is (..., K, 2, 2)
    # Tinv is (..., 2, 2) -> expand to (..., 1, 2, 2)
    Tinv_expanded = Tinv[..., jnp.newaxis, :, :]

    # matmul: (..., 1, 2, 2) @ (..., K, 2, 2) -> (..., K, 2, 2)
    # But Tinv is broadcasted over K

    # Tinv @ cov
    res = jnp.matmul(Tinv_expanded, cov)
    # res @ Tinv^T
    # Tinv^T is (..., 1, 2, 2) (transpose last two dims)
    Tinv_T = jnp.swapaxes(Tinv_expanded, -1, -2)

    new_cov = jnp.matmul(res, Tinv_T)
    return new_cov


def gaussian_fourier_transform(amp, var, mu, v, w):
    """
    Computes Fourier Transform of Mixture of Gaussians.

    Args:
        amp: Amplitudes (..., K)
        var: Variances (..., K, 2, 2)
        mu: Means (..., K, 2) (Can be None or zeros if centered)
        v: Frequencies x (..., W) or (..., W)
        w: Frequencies y (..., H)

    Returns:
        Fsum: (..., H, W) Complex array
    """
    # v, w can be 1D arrays of frequencies.
    # Let's assume v corresponds to last dim (width), w to second last (height).

    # var components
    a = var[..., 0, 0]
    b = var[..., 0, 1]
    d = var[..., 1, 1]

    # Expand dims for broadcasting
    # v: (W,) -> (1, ..., 1, 1, W)
    # w: (H,) -> (1, ..., 1, H, 1)
    # amp: (..., K) -> (..., K, 1, 1)
    # var elements: (..., K) -> (..., K, 1, 1)

    v_grid = v
    w_grid = w

    # We assume v and w are passed such that they broadcast correctly or we reshape them.
    # Usually v is (W,), w is (H,).
    # We want output (..., H, W).
    # Inputs have shape (..., K).

    # Let's add dimensions
    a = a[..., jnp.newaxis, jnp.newaxis]
    b = b[..., jnp.newaxis, jnp.newaxis]
    d = d[..., jnp.newaxis, jnp.newaxis]
    amp = amp[..., jnp.newaxis, jnp.newaxis]

    vv = v_grid**2
    ww = w_grid**2
    vw = v_grid * w_grid

    # Exponential argument (real part)
    # -2 * pi^2 * (a*v^2 + d*w^2 + 2*b*v*w)
    arg_real = -2.0 * (jnp.pi**2) * (a * vv + d * ww + 2 * b * vw)

    F = jnp.exp(arg_real)

    if mu is not None:
        mx = mu[..., 0][..., jnp.newaxis, jnp.newaxis]
        my = mu[..., 1][..., jnp.newaxis, jnp.newaxis]

        # Exponential argument (imaginary part)
        # -2 * pi * i * (mx*v + my*w)
        arg_imag = -2.0 * jnp.pi * 1j * (mx * v_grid + my * w_grid)
        F = F * jnp.exp(arg_imag)

    # Sum over K components
    Fsum = jnp.sum(amp * F, axis=-3)  # Sum over K axis (which is -3 now: ..., K, H, W)

    return Fsum


def render_pixelized_psf(psf_img, dx, dy):
    """
    Shifts a pixelized PSF image by (dx, dy).

    Args:
        psf_img: (H, W)
        dx, dy: scalars

    Returns:
        Shifted image (H, W)
    """
    # Replicate tractor logic:
    # 1. Lanczos filter x-shift (correlate rows)
    # 2. Lanczos filter y-shift (correlate cols of result)

    # We use batch_correlate1d from miscutils which expects (Batch, H, W).
    # So we add batch dim.
    img_b = psf_img[jnp.newaxis, :, :]

    dx_b = jnp.array([dx])
    dy_b = jnp.array([dy])

    # lanczos_shift_image_batch_gpu in psf.py
    # But we can just write it here using miscutils imports

    L = 3
    # kernels
    # We need a grid of shifts.
    # miscutils.lanczos_filter(order, x)

    # Construct kernels
    k_range = jnp.arange(-L, L + 1)
    Lx = lanczos_filter(L, k_range + dx)
    Ly = lanczos_filter(L, k_range + dy)

    # Normalize
    Lx = Lx / jnp.sum(Lx)
    Ly = Ly / jnp.sum(Ly)

    # Lx shape: (7,)
    # correlate1d expects b to be (Batch, Len).
    Lx = Lx[jnp.newaxis, :]
    Ly = Ly[jnp.newaxis, :]

    # Shift X (axis 2)
    sx = batch_correlate1d(img_b, Lx, axis=2, mode="constant")
    # Shift Y (axis 1)
    outimg = batch_correlate1d(sx, Ly, axis=1, mode="constant")

    return outimg[0]


def render_galaxy_fft(
    galaxy_mix, psf_fft, shape_params, wcs_cd_inv, subpixel_offset, image_shape
):
    """
    Renders a galaxy using FFT convolution.

    Args:
        galaxy_mix: (amp, mean, var) of the galaxy profile (normalized, unsheared).
        psf_fft: (H, W) Complex Fourier Transform of the PSF (padded to image_shape).
        shape_params: (re, ab, phi)
        wcs_cd_inv: (2, 2) Inverse CD matrix
        subpixel_offset: (dx, dy)
        image_shape: (H, W)

    Returns:
        Rendered image (H, W)
    """
    amp, mean, var = galaxy_mix
    re, ab, phi = shape_params
    # subpixel_offset is actually full position (x,y) if we assume galaxy_fft is centered at 0
    # and we want to shift it to position.
    # Or is it (dx, dy)?
    # The signature in optimizer calls it `pos_pix`.
    # So it is (x, y).

    pos_x, pos_y = subpixel_offset
    H, W = image_shape

    # 1. Compute shear matrix
    G = get_galaxy_shape_matrix(re, ab, phi)
    Tinv = get_shear_matrix(wcs_cd_inv, G)

    # 2. Shear the galaxy profile
    # Only variance changes for centered profile
    sheared_var = apply_shear_to_cov(var, Tinv)
    # Means are 0 for centered profile.
    sheared_mean = jnp.zeros_like(mean)

    # 3. Compute FFT of galaxy profile
    freq_x = jfft.rfftfreq(W)
    freq_y = jfft.fftfreq(H)

    # Meshgrid frequencies
    # v (x freq) varies along last axis
    # w (y freq) varies along second last axis
    v_grid, w_grid = jnp.meshgrid(freq_x, freq_y)

    # Galaxy is centered at (0,0).
    # We want to shift it to (pos_x, pos_y).
    # We can do this by adding phase shift in Fourier domain.
    # Shift theorem: F(f(x-x0)) = F(f(x)) * exp(-2pi i k x0)
    # Here k is frequency.

    # But wait. `gaussian_fourier_transform` handles mean (position).
    # We can just set the mean of the Gaussian to (pos_x, pos_y).
    # BUT: (0,0) in FFT corresponds to index (0,0).
    # If we want to place it at (pos_x, pos_y) in image coordinates,
    # we should check coordinate system convention.
    # Usually index (0,0) is top-left.
    # The gaussian_fourier_transform formula uses exp(-2pi i (mx*v + my*w)).
    # This corresponds to shifting the function centered at 0 to mean (mx, my).
    # So if we set mean = (pos_x, pos_y), the inverse FFT will have the gaussian at (pos_x, pos_y).

    # Note: sheared_mean is (K, 2). It is 0.
    # We add pos to it.
    shifted_mean = sheared_mean + jnp.array([pos_x, pos_y])

    gal_fft = gaussian_fourier_transform(amp, sheared_var, shifted_mean, v_grid, w_grid)

    # 4. Multiply with PSF FFT
    # psf_fft should be rfft2 format
    convolved_fft = gal_fft * psf_fft

    # 5. Inverse FFT
    # irfft2
    img = jfft.irfft2(convolved_fft, s=(H, W))

    # 6. Lanczos shift by subpixel offset
    # No need if we used phase shift for full position!
    # Unless we want to split integer and fractional part.
    # Using phase shift for integer part is exact.
    # Using phase shift for fractional part is also exact (sinc interpolation equivalent).
    # But Lanczos is local and avoids ringing?
    # Actually phase shift is perfect reconstruction for bandlimited signals.
    # Gaussians are not bandlimited but quickly decaying.
    # FFT based shift is fine.

    # So we don't need render_pixelized_psf here if we use phase shift.

    # However, if we wanted to replicate Tractor exactly:
    # Tractor computes galaxy on grid, then convolves.
    # Or computes galaxy FFT, multiplies PSF FFT.
    # Tractor centers the patch.

    # Our `optimizer.py` implementation pads PSF to full image and centers at 0 (via ifftshift).
    # This implies the PSF kernel is centered at (0,0).
    # So convolution does not introduce shift.
    # The galaxy shift determines final position.

    # Important: `irfft2` result has origin at (0,0).
    # But standard image display assumes (0,0) is center of top-left pixel?
    # Or corner?
    # We want result to be on grid.

    # Also, we need to take Real part because of numerical noise.
    # irfft2 returns real.

    # FFT normalization?
    # jax.numpy.fft.rfft2 is unnormalized.
    # jax.numpy.fft.irfft2 has 1/N normalization?
    # Docs say: "The inverse of rfft2 is irfft2."
    # Standard: forward sum, backward 1/N sum.
    # If we do F_gal * F_psf, and invert.
    # F_gal is from gaussian_fourier_transform which is analytic FT formula.
    # Analytic FT is integral.
    # Discrete FT (DFT) approximates integral but with scaling.
    # DFT(x) ~ Sum x.
    # FT(x) ~ Integral x.
    # If dx=1 (pixels), then DFT(x) ~ FT(x).
    # So scaling should be fine.

    return img


def render_point_source_pixelized(flux, subpixel_offset, psf_image):
    """
    Renders a point source with pixelized PSF.

    Args:
        flux: Scalar
        subpixel_offset: (dx, dy)
        psf_image: (H, W) PSF stamp

    Returns:
        (H, W) Image
    """
    dx, dy = subpixel_offset
    shifted_psf = render_pixelized_psf(psf_image, dx, dy)
    return flux * shifted_psf


def render_point_source_fft(flux, pos, psf_fft, image_shape):
    """
    Renders a point source using FFT convolution (phase shift).

    Args:
        flux: Scalar flux.
        pos: (x, y) Position.
        psf_fft: (H, W) FFT of PSF (centered at 0 frequency).
        image_shape: (H, W).

    Returns:
        Rendered image (H, W).
    """
    H, W = image_shape

    # Frequencies
    freq_x = jfft.rfftfreq(W)
    freq_y = jfft.fftfreq(H)

    v, w = jnp.meshgrid(freq_x, freq_y)

    # Phase shift for position pos=(x, y)
    # exp(-2pi * i * (x*v + y*w))
    # Note: v corresponds to x (last axis), w to y (second last axis).

    phase = -2.0 * jnp.pi * 1j * (pos[0] * v + pos[1] * w)
    shift_fft = jnp.exp(phase)

    # Convolve: Multiply FFTs
    # Point source FFT is flux * shift_fft
    model_fft = flux * shift_fft * psf_fft

    # Inverse FFT
    img = jfft.irfft2(model_fft, s=(H, W))

    return img


def convolve_gaussians(amp1, mean1, var1, amp2, mean2, var2):
    """
    Convolves two MoGs.
    MoG1: (K1) components
    MoG2: (K2) components

    Args:
        amp1: (K1,)
        mean1: (K1, 2)
        var1: (K1, 2, 2)
        amp2: (K2,)
        mean2: (K2, 2)
        var2: (K2, 2, 2)

    Returns:
        (amp, mean, var) of size K1*K2
    """
    # Reshape for broadcasting
    # (K1, 1) * (1, K2) -> (K1, K2)
    new_amp = (amp1[:, jnp.newaxis] * amp2[jnp.newaxis, :]).reshape(-1)

    # (K1, 1, 2) + (1, K2, 2) -> (K1, K2, 2)
    new_mean = (mean1[:, jnp.newaxis, :] + mean2[jnp.newaxis, :, :]).reshape(-1, 2)

    # (K1, 1, 2, 2) + (1, K2, 2, 2) -> (K1, K2, 2, 2)
    new_var = (var1[:, jnp.newaxis, :, :] + var2[jnp.newaxis, :, :, :]).reshape(
        -1, 2, 2
    )

    return new_amp, new_mean, new_var


def evaluate_mog_grid(amp, mean, var, X, Y):
    """
    Evaluates MoG on a grid (X, Y).

    Args:
        amp: (K,)
        mean: (K, 2)
        var: (K, 2, 2)
        X, Y: (H, W) coordinate arrays

    Returns:
        Image (H, W)
    """
    # Stack coords: (H, W, 2)
    pos = jnp.stack([X, Y], axis=-1)

    # Expand dims for K
    # pos: (H, W, 1, 2)
    pos = pos[..., jnp.newaxis, :]

    # mean: (1, 1, K, 2)
    mu = mean[jnp.newaxis, jnp.newaxis, :, :]

    # diff: (H, W, K, 2)
    diff = pos - mu

    # var: (1, 1, K, 2, 2)
    cov = var[jnp.newaxis, jnp.newaxis, :, :, :]

    # Inverse covariance and determinant
    # We can use jnp.linalg.inv and det.
    # But for 2x2, explicit formula is faster/simpler?
    # Let's use jax.numpy.linalg for generality.

    inv_cov = jnp.linalg.inv(cov)  # (1, 1, K, 2, 2)
    det_cov = jnp.linalg.det(cov)  # (1, 1, K)

    # Mahalanobis distance
    # diff^T * inv_cov * diff
    # (H, W, K, 1, 2) @ (H, W, K, 2, 2) @ (H, W, K, 2, 1)

    # diff is (..., 2). Expand to column vector (..., 2, 1)
    diff_col = diff[..., jnp.newaxis]
    diff_row = diff[..., jnp.newaxis, :]  # (..., 1, 2)

    # inv_cov @ diff
    # (..., 2, 2) @ (..., 2, 1) -> (..., 2, 1)
    temp = jnp.matmul(inv_cov, diff_col)

    # diff^T @ temp
    # (..., 1, 2) @ (..., 2, 1) -> (..., 1, 1)
    exponent = -0.5 * jnp.matmul(diff_row, temp).squeeze((-1, -2))

    # Prefactor
    # 1 / (2*pi * sqrt(det))
    # Be careful with det sign? Cov should be positive definite.
    # Clip det for stability?
    det_cov = jnp.maximum(det_cov, 1e-12)

    norm = 1.0 / (2.0 * jnp.pi * jnp.sqrt(det_cov))

    # Gaussian values
    gauss = norm * jnp.exp(exponent)  # (H, W, K)

    # Replace nans with 0
    gauss = jnp.nan_to_num(gauss)

    # Weighted sum

    # amp is (K,).
    # gauss is (H, W, K).

    weighted_gauss = amp[jnp.newaxis, jnp.newaxis, :] * gauss

    return jnp.sum(weighted_gauss, axis=-1)


def render_galaxy_mog(galaxy_mix, psf_mix, shape_params, wcs_cd_inv, pos, image_shape):
    """
    Renders a galaxy using MoG convolution (Analytic).

    Args:
        galaxy_mix: (amp, mean, var) (normalized, unsheared)
        psf_mix: (amp, mean, var)
        shape_params: (re, ab, phi)
        wcs_cd_inv: (2, 2)
        pos: (x, y) Center position in pixels
        image_shape: (H, W)

    Returns:
        Image (H, W)
    """
    gal_amp, gal_mean, gal_var = galaxy_mix
    psf_amp, psf_mean, psf_var = psf_mix
    re, ab, phi = shape_params

    # 1. Shear Galaxy Profile
    G = get_galaxy_shape_matrix(re, ab, phi)
    Tinv = get_shear_matrix(wcs_cd_inv, G)

    # Tinv takes unit_re -> pixels.
    # We want to apply affine transform Tinv to the covariance.
    # The galaxy profile is in unit_re coords.
    # Covariance transforms as T C T^T.

    sheared_gal_var = apply_shear_to_cov(gal_var, Tinv)
    # Centered galaxy mean is 0
    sheared_gal_mean = jnp.zeros_like(gal_mean)

    # 2. Convolve with PSF
    # PSF is already in pixels.
    conv_amp, conv_mean, conv_var = convolve_gaussians(
        gal_amp, sheared_gal_mean, sheared_gal_var, psf_amp, psf_mean, psf_var
    )

    # Debug info
    # print(f"Gal Var: {sheared_gal_var}")
    # print(f"Conv Var: {conv_var}")

    # 3. Add position offset
    # conv_mean is relative to (0,0). Add pos.
    # But pos is (x, y) = (col, row).
    # Tractor means are (x, y).
    # evaluate_mog_grid expects mean as (x, y).
    # pos is from wcs.positionToPixel, so (x, y).

    final_mean = conv_mean + jnp.array(pos)

    # 4. Evaluate on grid
    H, W = image_shape
    xx, yy = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    img = evaluate_mog_grid(conv_amp, final_mean, conv_var, xx, yy)

    return img


def render_point_source_mog(flux, pos, psf_mix, image_shape):
    """
    Renders a point source with MoG PSF.

    Args:
        flux: Scalar
        pos: (x, y)
        psf_mix: (amp, mean, var)
        image_shape: (H, W)

    Returns:
        Image (H, W)
    """
    amp, mean, var = psf_mix

    # Shift mean
    final_mean = mean + jnp.array(pos)

    # Evaluate
    H, W = image_shape
    xx, yy = jnp.meshgrid(jnp.arange(W), jnp.arange(H))

    # Normalized PSF image
    img = evaluate_mog_grid(amp, final_mean, var, xx, yy)

    return flux * img
