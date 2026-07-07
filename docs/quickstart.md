# Quickstart

A minimal example: build a synthetic image containing one point source, then
render its model with a `Tractor`.

```python
import numpy as np
from tractor_jax import (
    Tractor, Image, PointSource, PixPos, Flux, NCircularGaussianPSF,
)

# --- a fake 50x50 image with a Gaussian PSF -------------------------------
H = W = 50
psf = NCircularGaussianPSF([1.5], [1.0])   # sigma = 1.5 pix

image = Image(
    data=np.zeros((H, W), dtype=np.float32),
    inverr=np.ones((H, W), dtype=np.float32),
    psf=psf,
)

# --- one point source at pixel (24, 27) with flux 100 ----------------------
src = PointSource(PixPos(24.0, 27.0), Flux(100.0))

# --- render the model image -------------------------------------------------
tractor = Tractor(images=[image], catalog=[src])
model = tractor.getModelImage(0)
```

Without a `wcs` the image is treated as pixel space, without `sky` as zero
sky, and without `photocal` as count units — see
{class}`tractor_jax.engine.Image`.

## GPU-batched rendering and optimization

The JAX-accelerated batch machinery — rendering many sources into many
images at once, tiling large mosaics, and gradient-based fitting — lives in
the {mod}`tractor_jax.jax` subpackage:

- {mod}`tractor_jax.jax.rendering` — FFT and Gaussian-mixture rendering
  kernels (`render_pixelized_psf`, `render_galaxy_fft`, ...)
- {mod}`tractor_jax.jax.optimizer` — batched `jax.grad`-based optimization
- {mod}`tractor_jax.jax.tiling` — image tiling and catalog projection

See the [API reference](api.md) for the full listing.
