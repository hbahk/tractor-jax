import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import jax.image
from jax import vmap

from tractor_jax.miscutils import lanczos_filter, batch_correlate1d


def rebin_downsample_int_flux(img: jnp.ndarray, k_y: int, k_x: int) -> jnp.ndarray:
    """Flux-conserving integer-factor downsample.

    Block-sums the input image over non-overlapping ``k_y x k_x`` blocks.
    If the image dimensions are not divisible by the factors, the image is
    cropped to the largest divisible extent before rebinning.

    Parameters
    ----------
    img : jnp.ndarray
        Input image of shape ``(H, W)``; flux per pixel (integrated over
        the pixel).
    k_y : int
        Integer downsampling factor along the first (row) axis.
    k_x : int
        Integer downsampling factor along the second (column) axis.

    Returns
    -------
    jnp.ndarray
        Downsampled image of shape ``(H // k_y, W // k_x)``. The total sum
        is preserved when ``H`` and ``W`` are divisible by the factors.
    """
    H, W = img.shape
    H2 = (H // k_y) * k_y
    W2 = (W // k_x) * k_x
    img = img[:H2, :W2]  # crop; or pad if you prefer
    img = img.reshape(H2 // k_y, k_y, W2 // k_x, k_x)
    return img.sum(axis=(1, 3))


def get_galaxy_shape_matrix(re, ab, phi):
    """Compute the galaxy shape transformation matrix.

    Computes the transformation matrix ``G`` that takes unit vectors (in
    units of the effective radius) to degrees (intermediate world
    coordinates), matching the logic of ``tractor/galaxy.py``.

    Parameters
    ----------
    re : jnp.ndarray or float
        Effective radius in arcsec (scalar or array). A minimum size of
        1/30 arcsec is enforced to prevent singular matrix inversions.
    ab : jnp.ndarray or float
        Axis ratio (scalar or array).
    phi : jnp.ndarray or float
        Position angle in degrees, East of North (scalar or array).

    Returns
    -------
    G : jnp.ndarray
        Transformation matrix of shape ``(..., 2, 2)``.
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
    """Compute the shear matrix ``Tinv = cd_inv @ G``.

    Parameters
    ----------
    cd_inv : jnp.ndarray
        Inverse CD matrix of shape ``(..., 2, 2)``, mapping degrees to
        pixels.
    G : jnp.ndarray
        Galaxy shape matrix of shape ``(..., 2, 2)``, mapping unit
        effective-radius vectors to degrees.

    Returns
    -------
    Tinv : jnp.ndarray
        Transformation matrix of shape ``(..., 2, 2)``, mapping unit
        effective-radius vectors to pixels.
    """
    # Matrix multiplication
    return jnp.matmul(cd_inv, G)


def apply_shear_to_cov(cov, Tinv):
    """Apply a shear transformation to covariance matrices.

    Transforms each covariance matrix as
    ``new_cov = Tinv @ cov @ Tinv.T``, broadcasting ``Tinv`` over the
    mixture-component axis ``K``.

    Parameters
    ----------
    cov : jnp.ndarray
        Covariance matrices of shape ``(..., K, 2, 2)``.
    Tinv : jnp.ndarray
        Shear matrices of shape ``(..., 2, 2)``.

    Returns
    -------
    jnp.ndarray
        Transformed covariance matrices of shape ``(..., K, 2, 2)``.
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
    """Compute the Fourier transform of a mixture of Gaussians.

    Evaluates the analytic Fourier transform of a 2-D Gaussian mixture on
    a grid of frequencies and sums over the mixture components.

    Parameters
    ----------
    amp : jnp.ndarray
        Amplitudes of shape ``(..., K)``.
    var : jnp.ndarray
        Covariance matrices of shape ``(..., K, 2, 2)``.
    mu : jnp.ndarray or None
        Means of shape ``(..., K, 2)``. May be ``None`` (or zeros) for a
        centered mixture, in which case the phase term is omitted.
    v : jnp.ndarray
        Frequency grid along x, broadcastable against ``w`` to shape
        ``(H, W)`` (e.g. a meshgrid of ``rfftfreq(W)``).
    w : jnp.ndarray
        Frequency grid along y, broadcastable against ``v`` to shape
        ``(H, W)`` (e.g. a meshgrid of ``fftfreq(H)``).

    Returns
    -------
    Fsum : jnp.ndarray
        Complex array of shape ``(..., H, W)``: the mixture's Fourier
        transform summed over the ``K`` components.
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
    """Shift a pixelized PSF image by a subpixel offset.

    Replicates the Tractor logic: a Lanczos-3 filter shift in x
    (correlating rows), followed by a Lanczos-3 filter shift in y
    (correlating columns of the result). The Lanczos kernels are
    normalized to unit sum.

    Parameters
    ----------
    psf_img : jnp.ndarray
        PSF image of shape ``(H, W)``.
    dx : float
        Subpixel shift along x (scalar).
    dy : float
        Subpixel shift along y (scalar).

    Returns
    -------
    jnp.ndarray
        Shifted image of shape ``(H, W)``.
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


def _boxcar_downsample_flux(img, out_h, out_w):
    """Flux-conserving, window-applying downsample for non-integer factors.

    Integrates the high-resolution image over each output pixel's footprint
    (a boxcar of width ``factor = H_hr / out_h``). This is the native-pixel
    integration window that the detector applies and that
    :func:`rebin_downsample_int_flux` applies for integer factors.

    Parameters
    ----------
    img : jnp.ndarray
        High-resolution input image of shape ``(H, W)``.
    out_h : int
        Output height.
    out_w : int
        Output width.

    Returns
    -------
    jnp.ndarray
        Downsampled image of shape ``(out_h, out_w)``.

    Notes
    -----
    The previous ``jax.image.resize(lanczos3)`` fallback omitted the pixel
    window (it interpolates point values instead of integrating), which
    made templates too sharp on undersampled PSFs (research note
    proj-spherex-gpupipe lasso_alpha/11).

    The method is exact for a piecewise-constant high-resolution image: the
    integral up to a fractional position ``x`` equals the linear
    interpolant of the cumulative sum at ``x``, so each output pixel is
    ``cumsum(edge_{i+1}) - cumsum(edge_i)``. It reduces exactly to the
    integer block-sum when the factor is an integer, and conserves total
    flux by construction (telescoping sum).
    """
    H, W = img.shape
    dt = img.dtype
    # integrate over rows (axis 0): H -> out_h
    z = jnp.zeros((1, W), dt)
    Cr = jnp.concatenate([z, jnp.cumsum(img, axis=0)], axis=0)      # (H+1, W)
    er = (H / out_h) * jnp.arange(out_h + 1, dtype=dt)
    xr = jnp.arange(H + 1, dtype=dt)
    Cri = jax.vmap(lambda c: jnp.interp(er, xr, c), in_axes=1,
                   out_axes=1)(Cr)                                  # (out_h+1, W)
    rows = Cri[1:] - Cri[:-1]                                       # (out_h, W)
    # integrate over cols (axis 1): W -> out_w
    z2 = jnp.zeros((out_h, 1), dt)
    Cc = jnp.concatenate([z2, jnp.cumsum(rows, axis=1)], axis=1)    # (out_h, W+1)
    ec = (W / out_w) * jnp.arange(out_w + 1, dtype=dt)
    xc = jnp.arange(W + 1, dtype=dt)
    Cci = jax.vmap(lambda r: jnp.interp(ec, xc, r), in_axes=0,
                   out_axes=0)(Cc)                                  # (out_h, out_w+1)
    return Cci[:, 1:] - Cci[:, :-1]                                 # (out_h, out_w)


def downsample_image(img, target_shape):
    """Downsample an image flux-conservingly to a target shape.

    Applies the native output-pixel integration window (so a rendered PSF
    is integrated over each detector pixel, not point-sampled). Integer
    factors use fast block-sum rebinning; non-integer factors use the
    exact boxcar integral (:func:`_boxcar_downsample_flux`), which agrees
    with the block-sum at integer factors.

    Parameters
    ----------
    img : jnp.ndarray
        High-resolution input image of shape ``(H_hr, W_hr)``.
    target_shape : tuple of int
        Target shape ``(H, W)``. Must be static or concrete at trace time
        so that integer downsampling can be detected.

    Returns
    -------
    jnp.ndarray
        Downsampled image of shape ``(H, W)``.

    Notes
    -----
    The previous non-integer path used ``jax.image.resize(lanczos3)``,
    which band-limits but does NOT apply the pixel window — biasing
    forced-photometry templates on undersampled PSFs
    (proj-spherex-gpupipe lasso_alpha/11). Integer-factor rendering
    (e.g. production SPHEREx cutouts at OVERSAMP 10/5) is unchanged.
    """
    H_hr, W_hr = img.shape
    H, W = target_shape

    # Try to detect integer downsampling
    # This assumes shapes are static or concrete integers at trace time
    is_int_y = (H_hr % H == 0)
    is_int_x = (W_hr % W == 0)

    if is_int_y and is_int_x:
        k_y = int(H_hr // H)
        k_x = int(W_hr // W)
        return rebin_downsample_int_flux(img, k_y, k_x)

    return _boxcar_downsample_flux(img, H, W)


def render_galaxy_fft(
    galaxy_mix,
    psf_fft,
    shape_params,
    wcs_cd_inv,
    subpixel_offset,
    image_shape,
):
    """Render a galaxy using FFT convolution.

    Shears the (normalized, unsheared) galaxy mixture-of-Gaussians profile
    into pixel coordinates, evaluates its analytic Fourier transform at the
    subpixel-shifted position, multiplies by the PSF Fourier transform, and
    inverse-transforms to the image plane.

    Parameters
    ----------
    galaxy_mix : tuple of jnp.ndarray
        ``(amp, mean, var)`` of the galaxy profile (normalized,
        unsheared), with shapes ``(K,)``, ``(K, 2)`` and ``(K, 2, 2)``.
    psf_fft : jnp.ndarray
        Complex Fourier transform of the PSF, in ``rfft2`` layout matching
        ``image_shape``.
    shape_params : tuple
        ``(re, ab, phi)``: effective radius in arcsec, axis ratio, and
        position angle in degrees.
    wcs_cd_inv : jnp.ndarray
        Inverse CD matrix of shape ``(2, 2)``.
    subpixel_offset : tuple
        ``(x, y)`` subpixel offset of the galaxy center.
    image_shape : tuple of int
        Target image shape ``(H, W)`` in data pixels.

    Returns
    -------
    jnp.ndarray
        Rendered image of shape ``(H, W)``.
    """
    amp, mean, var = galaxy_mix
    re, ab, phi = shape_params
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
    v_grid, w_grid = jnp.meshgrid(freq_x, freq_y)

    # Galaxy is centered at (0,0).
    # We want to shift it to (pos_x, pos_y).
    shifted_mean = sheared_mean + jnp.array([pos_x, pos_y])

    gal_fft = gaussian_fourier_transform(amp, sheared_var, shifted_mean, v_grid, w_grid)

    # 4. Multiply with PSF FFT
    # psf_fft should be rfft2 format matching (H, W)
    convolved_fft = gal_fft * psf_fft

    # 5. Inverse FFT
    img = jfft.irfft2(convolved_fft, s=(H, W))

    return img


def render_point_source_pixelized(flux, subpixel_offset, psf_image):
    """Render a point source with a pixelized PSF.

    Shifts the PSF stamp by the subpixel offset (Lanczos interpolation via
    :func:`render_pixelized_psf`) and scales it by the flux.

    Parameters
    ----------
    flux : float or jnp.ndarray
        Scalar source flux (in the image's calibrated units).
    subpixel_offset : tuple
        ``(dx, dy)`` subpixel shift.
    psf_image : jnp.ndarray
        PSF stamp of shape ``(H, W)``.

    Returns
    -------
    jnp.ndarray
        Rendered image of shape ``(H, W)``.
    """
    dx, dy = subpixel_offset
    shifted_psf = render_pixelized_psf(psf_image, dx, dy)
    return flux * shifted_psf


def render_point_source_fft(flux, pos, psf_fft, image_shape):
    """Render a point source using FFT convolution (phase shift).

    Multiplies the PSF Fourier transform by a phase ramp encoding the
    source position and by the flux, then inverse-transforms to the image
    plane.

    Parameters
    ----------
    flux : float or jnp.ndarray
        Scalar source flux (in the image's calibrated units).
    pos : jnp.ndarray or tuple
        ``(x, y)`` position in pixels.
    psf_fft : jnp.ndarray
        Fourier transform of the PSF (centered at zero frequency), in
        ``rfft2`` layout matching ``image_shape``.
    image_shape : tuple of int
        Target image shape ``(H, W)``.

    Returns
    -------
    jnp.ndarray
        Rendered image of shape ``(H, W)``.
    """
    H, W = image_shape

    # Frequencies
    freq_x = jfft.rfftfreq(W)
    freq_y = jfft.fftfreq(H)

    v, w = jnp.meshgrid(freq_x, freq_y)

    # Phase shift for position
    # exp(-2pi * i * (x*v + y*w))
    phase = -2.0 * jnp.pi * 1j * (pos[0] * v + pos[1] * w)
    shift_fft = jnp.exp(phase)

    # Convolve: Multiply FFTs
    # Point source FFT is flux * shift_fft
    model_fft = flux * shift_fft * psf_fft

    # Inverse FFT
    img = jfft.irfft2(model_fft, s=(H, W))

    return img


def convolve_gaussians(amp1, mean1, var1, amp2, mean2, var2):
    """Convolve two mixtures of Gaussians.

    The convolution of a ``K1``-component mixture with a ``K2``-component
    mixture is a ``K1 * K2``-component mixture whose amplitudes multiply
    and whose means and covariances add, pairwise.

    Parameters
    ----------
    amp1 : jnp.ndarray
        Amplitudes of the first mixture, shape ``(K1,)``.
    mean1 : jnp.ndarray
        Means of the first mixture, shape ``(K1, 2)``.
    var1 : jnp.ndarray
        Covariances of the first mixture, shape ``(K1, 2, 2)``.
    amp2 : jnp.ndarray
        Amplitudes of the second mixture, shape ``(K2,)``.
    mean2 : jnp.ndarray
        Means of the second mixture, shape ``(K2, 2)``.
    var2 : jnp.ndarray
        Covariances of the second mixture, shape ``(K2, 2, 2)``.

    Returns
    -------
    amp : jnp.ndarray
        Amplitudes of the convolved mixture, shape ``(K1 * K2,)``.
    mean : jnp.ndarray
        Means of the convolved mixture, shape ``(K1 * K2, 2)``.
    var : jnp.ndarray
        Covariances of the convolved mixture, shape ``(K1 * K2, 2, 2)``.
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
    """Evaluate a mixture of Gaussians on a coordinate grid.

    Evaluates the amplitude-weighted sum of normalized 2-D Gaussian
    densities at each grid point. Covariance determinants are clipped
    at ``1e-12`` for numerical stability, and NaNs are replaced by zero.

    Parameters
    ----------
    amp : jnp.ndarray
        Amplitudes of shape ``(K,)``.
    mean : jnp.ndarray
        Means of shape ``(K, 2)``, ordered as ``(x, y)``.
    var : jnp.ndarray
        Covariance matrices of shape ``(K, 2, 2)``.
    X : jnp.ndarray
        x-coordinate array of shape ``(H, W)``.
    Y : jnp.ndarray
        y-coordinate array of shape ``(H, W)``.

    Returns
    -------
    jnp.ndarray
        Image of shape ``(H, W)``.
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
    """Render a galaxy using analytic mixture-of-Gaussians convolution.

    Shears the (normalized, unsheared) galaxy profile into pixel
    coordinates, convolves it analytically with the PSF mixture, shifts
    the result to the source position, and evaluates it on the pixel
    grid.

    Parameters
    ----------
    galaxy_mix : tuple of jnp.ndarray
        ``(amp, mean, var)`` of the galaxy profile (normalized,
        unsheared), with shapes ``(K,)``, ``(K, 2)`` and ``(K, 2, 2)``.
    psf_mix : tuple of jnp.ndarray
        ``(amp, mean, var)`` of the PSF mixture, in pixel coordinates.
    shape_params : tuple
        ``(re, ab, phi)``: effective radius in arcsec, axis ratio, and
        position angle in degrees.
    wcs_cd_inv : jnp.ndarray
        Inverse CD matrix of shape ``(2, 2)``.
    pos : jnp.ndarray or tuple
        ``(x, y)`` center position in pixels.
    image_shape : tuple of int
        Target image shape ``(H, W)``.

    Returns
    -------
    jnp.ndarray
        Rendered image of shape ``(H, W)``.
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
    """Render a point source with a mixture-of-Gaussians PSF.

    Shifts the PSF mixture means to the source position, evaluates the
    normalized mixture on the pixel grid, and scales by the flux.

    Parameters
    ----------
    flux : float or jnp.ndarray
        Scalar source flux (in the image's calibrated units).
    pos : jnp.ndarray or tuple
        ``(x, y)`` position in pixels.
    psf_mix : tuple of jnp.ndarray
        ``(amp, mean, var)`` of the PSF mixture, with shapes ``(K,)``,
        ``(K, 2)`` and ``(K, 2, 2)``.
    image_shape : tuple of int
        Target image shape ``(H, W)``.

    Returns
    -------
    jnp.ndarray
        Rendered image of shape ``(H, W)``.
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
