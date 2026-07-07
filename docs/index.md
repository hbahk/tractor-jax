# tractor-jax

**tractor-jax** is a JAX-accelerated reimplementation of
[The Tractor](https://github.com/dstndstn/tractor) for astronomical image
modeling and forced photometry. It renders parametric source models (point
sources, exponential / de Vaucouleurs / composite galaxies) through pixelized
or Gaussian-mixture PSFs, and fits source parameters against multi-band,
multi-epoch imaging on GPUs via `jax.grad`-based optimization.

Developed for SPHEREx-scale joint deblending and forced photometry.

## Highlights

- GPU-batched rendering kernels (FFT and Gaussian-mixture pixel-space paths)
- Exact gradients through the full image model via JAX autodiff
- Drop-in model classes mirroring The Tractor's API (`Tractor`, `Image`,
  `PointSource`, `ExpGalaxy`, ...)
- Image tiling and bucketed batching for large mosaics

## Conventions

- Flux models carry whatever unit the input images are calibrated in (see
  `PhotoCal`); magnitudes are AB.
- WCS follows the **FITS convention** (1-based pixel origin, RA increasing to
  the left).
- Code follows JAX functional style: no in-place mutation, shape-determining
  arguments are static.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api
```
