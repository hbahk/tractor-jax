import jax.numpy as jnp
from jax import jit
import numpy as np

def get_overlapping_region(xlo, xhi, xmin, xmax):
    if xlo > xmax or xhi < xmin or xlo > xhi or xmin > xmax:
        return ([], [])
    assert(xlo <= xhi)
    assert(xmin <= xmax)
    xloclamp = max(xlo, xmin)
    Xlo = xloclamp - xlo
    xhiclamp = min(xhi, xmax)
    Xhi = Xlo + (xhiclamp - xloclamp)
    return (slice(xloclamp, xhiclamp+1), slice(Xlo, Xhi+1))

@jit
def lanczos_filter(order, x):
    x = jnp.atleast_1d(x)
    pinz = jnp.pi * x
    # Avoid division by zero
    pinz_safe = jnp.where(x == 0, 1.0, pinz)

    val = order * jnp.sin(pinz) * jnp.sin(pinz / order) / (pinz_safe**2)
    val = jnp.where(x == 0, 1.0, val)
    val = jnp.where(jnp.abs(x) >= order, 0.0, val)
    return val

from functools import partial

@partial(jit, static_argnames=['axis', 'mode'])
def batch_correlate1d(a, b, axis=1, mode='constant'):
    # a: (z, m, n)
    # b: (y, x)
    # axis: 1 or 2
    # mode: 'constant' (default) or 'full'

    z, m, n = a.shape
    y, x = b.shape

    # In the original code, z == y is enforced.
    # a is [N_images, H, W] or similar.
    # b is [N_images, FilterLen] or similar.

    # We use FFT convolution.
    # Determine padding size.

    if axis == 1:
        dim_len = m
    else:
        dim_len = n

    npad = dim_len - x
    r = dim_len + x - 1
    nclip = r - dim_len

    # Padding b
    # In original code:
    # if npad > 0:
    #    if npad % 2 == 0:
    #        padded_b = cp.pad(b, [0, npad//2])
    #    else:
    #        padded_b = cp.pad(b, [(0,0), (npad//2+1, npad//2)])
    # The padding logic in original code seems a bit specific/odd for general convolution but matches 'same' or 'valid' or specific alignment.
    # Let's try to match the logic.

    pad_width = []
    if npad > 0:
        if npad % 2 == 0:
            pad_width = ((0, 0), (0, npad//2))
        else:
            pad_width = ((0, 0), (npad//2+1, npad//2))
        padded_b = jnp.pad(b, pad_width)
    else:
        padded_b = b

    f_a = jnp.fft.fft(a, r, axis=axis)
    # b is 2D (y, x), we need to fft along axis 1 (the x dimension)
    f_b = jnp.fft.fft(padded_b, r, axis=1)

    # Broadcasting for einsum
    # a: (z, m, n) -> f_a: (z, m, n) (fft along axis)
    # b: (y, x) -> f_b: (y, r)
    # if axis=1: a is (z, m, n), fft is along m. f_a is (z, r, n).
    #            b matches z. f_b is (z, r).
    #            We want to multiply f_a[i, :, k] * conj(f_b[i, :])

    if axis == 1:
        # f_a: (z, r, n)
        # f_b: (z, r)
        # result: (z, r, n)
        f_p = jnp.einsum("ijk,ij->ijk", f_a, jnp.conj(f_b))
    else:
        # axis == 2
        # f_a: (z, m, r)
        # f_b: (z, r)
        # result: (z, m, r)
        f_p = jnp.einsum("ijk,ik->ijk", f_a, jnp.conj(f_b))

    c = jnp.real(jnp.fft.fftshift(jnp.fft.ifft(f_p, axis=axis), axes=(axis)))

    if mode == 'full':
        return c

    # Clipping
    start = nclip // 2
    end = -nclip // 2
    if end == 0:
        end = None

    if axis == 1:
        return c[:, start:end, :]
    else:
        return c[:, :, start:end]

def lanczos3_interpolate_grid(xstart, xstep, ystart, ystep, out_img, in_img):
    """
    Numpy implementation of Lanczos-3 grid interpolation.
    """
    H_out, W_out = out_img.shape
    H_in, W_in = in_img.shape
    L = 3

    x_out = xstart + np.arange(W_out) * xstep
    y_out = ystart + np.arange(H_out) * ystep

    def lanczos_kernel_np(x, a=3):
        x = np.atleast_1d(x)
        res = np.zeros_like(x)
        mask = np.abs(x) < a
        mask0 = mask & (x == 0)
        mask1 = mask & (x != 0)
        res[mask0] = 1.0
        xp = x[mask1] * np.pi
        res[mask1] = a * np.sin(xp) * np.sin(xp/a) / (xp**2)
        return res

    temp_img = np.zeros((H_in, W_out), dtype=in_img.dtype)

    # Vectorize inner loop?
    # For small images, loops are fine. For large, maybe slow.
    # But this is primarily for verification/tests/legacy CPU path.

    # Precompute X weights
    # For each output column i, we need weights for relevant k

    for i in range(W_out):
        x = x_out[i]
        k_min = int(np.ceil(x - L))
        k_max = int(np.floor(x + L))

        # We can construct index array
        ks = np.arange(k_min, k_max + 1)
        # Filter valid
        valid = (ks >= 0) & (ks < W_in)
        ks = ks[valid]

        if len(ks) > 0:
            w = lanczos_kernel_np(x - ks, L)
            # temp_img[:, i] = sum(in_img[:, k] * w)
            # broadcasting: in_img[:, ks] is (H_in, len(ks))
            # w is (len(ks),)
            temp_img[:, i] = np.dot(in_img[:, ks], w)

    for j in range(H_out):
        y = y_out[j]
        k_min = int(np.ceil(y - L))
        k_max = int(np.floor(y + L))

        ks = np.arange(k_min, k_max + 1)
        valid = (ks >= 0) & (ks < H_in)
        ks = ks[valid]

        if len(ks) > 0:
            w = lanczos_kernel_np(y - ks, L)
            out_img[j, :] = np.dot(w, temp_img[ks, :])

    return out_img
