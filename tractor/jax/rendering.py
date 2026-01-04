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


def downsample_image(img, factor):
    """
    Downsamples image by integer factor using sum (flux conservation).
    img: (H*factor, W*factor)
    Returns: (H, W)
    """
    H_hr, W_hr = img.shape
    H = H_hr // factor
    W = W_hr // factor
    # Reshape and sum
    # (H, factor, W, factor) -> sum axis 1, 3
    return img.reshape(H, factor, W, factor).sum(axis=(1, 3))


def render_galaxy_fft(
    galaxy_mix,
    psf_fft,
    shape_params,
    wcs_cd_inv,
    subpixel_offset,
    image_shape,
    psf_sampling=1.0,
):
    """
    Renders a galaxy using FFT convolution.

    Args:
        galaxy_mix: (amp, mean, var) of the galaxy profile (normalized, unsheared).
        psf_fft: (H, W) Complex Fourier Transform of the PSF.
                 If psf_sampling != 1, this should be the FFT of the oversampled PSF.
        shape_params: (re, ab, phi)
        wcs_cd_inv: (2, 2) Inverse CD matrix
        subpixel_offset: (x, y)
        image_shape: (H, W) target image shape (data pixels)
        psf_sampling: float. If < 1, indicates oversampling.

    Returns:
        Rendered image (H, W)
    """
    amp, mean, var = galaxy_mix
    re, ab, phi = shape_params
    pos_x, pos_y = subpixel_offset
    H, W = image_shape

    # Handle Oversampling
    if psf_sampling < 1.0:
        factor = int(round(1.0 / psf_sampling))
        # Ensure factor is valid (assuming integer oversampling for now)
        H_fft = H * factor
        W_fft = W * factor

        # Scale inputs to high-res
        # CD inv: degrees -> pixels. High res has 'factor' times more pixels per degree.
        wcs_cd_inv = wcs_cd_inv / psf_sampling

        # Position: data pixels -> high res pixels
        # Add offset to align pixel centers.
        # Low res pixel 0 center maps to High res pixel center of the first block.
        # Center of first block (0..F-1) is (F-1)/2.0.
        # pos_hr = pos_lr * factor + (factor - 1) / 2.0
        pos_x = pos_x / psf_sampling + (factor - 1) / 2.0
        pos_y = pos_y / psf_sampling + (factor - 1) / 2.0
    else:
        H_fft = H
        W_fft = W

    # 1. Compute shear matrix
    G = get_galaxy_shape_matrix(re, ab, phi)
    Tinv = get_shear_matrix(wcs_cd_inv, G)

    # 2. Shear the galaxy profile
    # Only variance changes for centered profile
    sheared_var = apply_shear_to_cov(var, Tinv)
    # Means are 0 for centered profile.
    sheared_mean = jnp.zeros_like(mean)

    # 3. Compute FFT of galaxy profile
    freq_x = jfft.rfftfreq(W_fft)
    freq_y = jfft.fftfreq(H_fft)

    # Meshgrid frequencies
    v_grid, w_grid = jnp.meshgrid(freq_x, freq_y)

    # Galaxy is centered at (0,0).
    # We want to shift it to (pos_x, pos_y).
    shifted_mean = sheared_mean + jnp.array([pos_x, pos_y])

    gal_fft = gaussian_fourier_transform(amp, sheared_var, shifted_mean, v_grid, w_grid)

    # 4. Multiply with PSF FFT
    # psf_fft should be rfft2 format matching (H_fft, W_fft)
    convolved_fft = gal_fft * psf_fft

    # 5. Inverse FFT
    img = jfft.irfft2(convolved_fft, s=(H_fft, W_fft))

    # 6. Downsample if needed
    if psf_sampling < 1.0:
        factor = int(round(1.0 / psf_sampling))
        img = downsample_image(img, factor)

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


def render_point_source_fft(flux, pos, psf_fft, image_shape, psf_sampling=1.0):
    """
    Renders a point source using FFT convolution (phase shift).

    Args:
        flux: Scalar flux.
        pos: (x, y) Position.
        psf_fft: (H, W) FFT of PSF (centered at 0 frequency).
        image_shape: (H, W).
        psf_sampling: float.

    Returns:
        Rendered image (H, W).
    """
    H, W = image_shape

    if psf_sampling < 1.0:
        factor = int(round(1.0 / psf_sampling))
        H_fft = H * factor
        W_fft = W * factor
        # Shift position for alignment
        pos_fft = pos / psf_sampling + (factor - 1) / 2.0
    else:
        H_fft = H
        W_fft = W
        pos_fft = pos

    # Frequencies
    freq_x = jfft.rfftfreq(W_fft)
    freq_y = jfft.fftfreq(H_fft)

    v, w = jnp.meshgrid(freq_x, freq_y)

    # Phase shift for position
    # exp(-2pi * i * (x*v + y*w))
    phase = -2.0 * jnp.pi * 1j * (pos_fft[0] * v + pos_fft[1] * w)
    shift_fft = jnp.exp(phase)

    # Convolve: Multiply FFTs
    # Point source FFT is flux * shift_fft
    model_fft = flux * shift_fft * psf_fft

    # Inverse FFT
    img = jfft.irfft2(model_fft, s=(H_fft, W_fft))

    # Downsample
    if psf_sampling < 1.0:
        factor = int(round(1.0 / psf_sampling))
        img = downsample_image(img, factor)

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
