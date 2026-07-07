# Tractor-JAX

[![CI](https://github.com/hbahk/tractor-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/hbahk/tractor-jax/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/tractor-jax/badge/?version=latest)](https://tractor-jax.readthedocs.io/en/latest/)

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

## Quickstart

```python
import numpy as np
from tractor_jax import (
    Tractor, Image, PointSource, PixPos, Flux, NCircularGaussianPSF,
)

image = Image(
    data=np.zeros((50, 50), dtype=np.float32),
    inverr=np.ones((50, 50), dtype=np.float32),
    psf=NCircularGaussianPSF([1.5], [1.0]),
)
src = PointSource(PixPos(24.0, 27.0), Flux(100.0))

tractor = Tractor(images=[image], catalog=[src])
model = tractor.getModelImage(0)
```

See the [documentation](https://tractor-jax.readthedocs.io) for the full
guide and API reference.

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

GPL-3.0 — see [LICENSE](LICENSE).

Portions of this code derive from
[The Tractor](https://github.com/dstndstn/tractor), copyright
Dustin Lang and David W. Hogg, distributed under the GNU General Public
License version 2 or (at your option) any later version; this project
exercises that option and distributes the derived work under GPL-3.0.
