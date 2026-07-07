# Tractor-JAX

JAX-accelerated astronomical image modeling and forced photometry.

**Tractor-JAX** is a GPU-oriented reimplementation of
[The Tractor](https://github.com/dstndstn/tractor) (Lang & Hogg). It renders
parametric source models — point sources and exponential / de Vaucouleurs /
composite / Sérsic galaxies — through pixelized or Gaussian-mixture PSFs, and
fits source parameters against multi-band, multi-epoch imaging with exact
gradients from JAX autodiff. Developed for SPHEREx-scale joint deblending and
forced photometry.

## Features

- GPU-batched rendering kernels (FFT and Gaussian-mixture pixel-space paths)
- Exact gradients through the full image model via `jax.grad`
- Model classes mirroring The Tractor's API (`Tractor`, `Image`,
  `PointSource`, `ExpGalaxy`, ...)
- Image tiling and bucketed batching for large mosaics

## Installation

```bash
git clone https://github.com/hbahk/tractor-jax.git
pip install -e tractor-jax
```

The default `pip install jax` dependency is CPU-only; for GPU acceleration
install the CUDA wheel, e.g. `pip install "jax[cuda12]"`.

## Documentation

Documentation lives in `docs/` and is hosted on Read the Docs
(<https://tractor-jax.readthedocs.io>). To build locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## History and provenance

This repository was started by extracting the parts of The Tractor needed for
GPU-based SPHEREx photometry, and was then rewritten around JAX. Commits made
before this repository was created can be found in the earlier fork
[hbahk/tractor-jax-forked](https://github.com/hbahk/tractor-jax-forked) and in
the upstream [dstndstn/tractor](https://github.com/dstndstn/tractor).

## License

See [LICENSE](LICENSE). Portions derive from The Tractor.
