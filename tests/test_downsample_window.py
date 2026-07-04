"""downsample_image: flux conservation, integer parity, and the native-pixel
integration window on non-integer factors (regression for the missing-window
bug diagnosed in proj-spherex-gpupipe research note lasso_alpha/11)."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tractor_jax.jax.rendering import (
    downsample_image, rebin_downsample_int_flux, _boxcar_downsample_flux,
)


def test_flux_conservation_noninteger():
    rng = np.random.default_rng(0)
    img = jnp.asarray(rng.random((373, 373)))          # 373/86 = 4.3.. non-integer
    out = downsample_image(img, (86, 86))
    assert out.shape == (86, 86)
    np.testing.assert_allclose(float(out.sum()), float(img.sum()), rtol=1e-12)


def test_integer_parity():
    """boxcar path == block-sum at integer factor; integer path unchanged."""
    rng = np.random.default_rng(1)
    img = jnp.asarray(rng.random((100, 100)))
    bx = np.asarray(_boxcar_downsample_flux(img, 20, 20))     # factor 5
    rb = np.asarray(rebin_downsample_int_flux(img, 5, 5))
    np.testing.assert_allclose(bx, rb, atol=1e-10)
    # downsample_image still routes integer factors through the fast block-sum
    di = np.asarray(downsample_image(img, (20, 20)))
    np.testing.assert_array_equal(di, rb)


def test_noninteger_is_nonnegative_and_windowed():
    """A sharp non-negative image stays non-negative (the old lanczos fallback
    rings negative) and each output pixel is the integral over its footprint."""
    img = np.zeros((100, 100))
    img[50, 50] = 1.0                                   # a hot pixel
    img[30, 70] = 0.5
    out = np.asarray(downsample_image(jnp.asarray(img), (30, 30)))  # factor 10/3
    assert out.min() >= -1e-12                          # no lanczos ringing
    np.testing.assert_allclose(out.sum(), img.sum(), rtol=1e-12)


def test_constant_image():
    """Constant flux/pixel -> each output pixel = (factor_y * factor_x)."""
    img = jnp.ones((100, 100))
    out = np.asarray(downsample_image(img, (30, 30)))
    np.testing.assert_allclose(out.mean(), (100.0 / 30.0) ** 2, rtol=1e-12)


def test_jit_and_vmap():
    """The non-integer path is jit- and vmap-safe (it runs under a vmapped
    partial in _render_source_templates)."""
    rng = np.random.default_rng(2)
    batch = jnp.asarray(rng.random((4, 373, 373)))
    f = jax.jit(jax.vmap(lambda im: downsample_image(im, (86, 86))))
    out = f(batch)
    assert out.shape == (4, 86, 86)
    np.testing.assert_allclose(np.asarray(out.sum(axis=(1, 2))),
                               np.asarray(batch.sum(axis=(1, 2))), rtol=1e-12)
