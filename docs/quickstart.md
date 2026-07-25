# Quickstart

Build a scene, render it, then fit it. This runs on CPU in a couple of seconds —
no GPU needed to try the API.

## 1. A scene

An `Image` carries the pixels, the inverse *error* map, and the calibration
objects; a `Catalog` carries the sources. Without a `wcs` the image is treated as
pixel space, without `sky` as zero sky, and without `photocal` as count units —
see {class}`tractor_jax.engine.Image`.

```python
import numpy as np
from tractor_jax import (Tractor, Image, Catalog, PointSource, PixPos, Flux,
                         NCircularGaussianPSF)

H = W = 50
NOISE = 0.5
psf = NCircularGaussianPSF([1.5], [1.0])          # sigma = 1.5 px

catalog = Catalog(PointSource(PixPos(24.0, 27.0), Flux(100.0)),
                  PointSource(PixPos(31.0, 18.0), Flux(40.0)))

image = Image(data=np.zeros((H, W)), inverr=np.full((H, W), 1.0 / NOISE),
              psf=psf)
tractor = Tractor([image], catalog)
```

:::{note}
`inverr` is the inverse **error** ($1/\sigma$), not the inverse variance.
Passing an inverse variance silently squares your weights — see
{doc}`migration`.
:::

## 2. Render

`getModelImage` evaluates the current catalog into pixels. Here we use it to
manufacture data: the noiseless model plus noise.

```python
clean = np.asarray(tractor.getModelImage(0))
rng = np.random.default_rng(0)
tractor.images[0].data = clean + rng.normal(0.0, NOISE, (H, W))
```

## 3. Fit

{func}`~tractor_jax.jax.optimizer.optimize_fluxes` solves every source's flux
jointly, with positions and shapes held fixed. It returns one entry per image.

```python
from tractor_jax.jax.optimizer import optimize_fluxes

fluxes, variances = optimize_fluxes(
    tractor, solver="eigfloor", return_variances=True,
    update_catalog=True)[0]

print(fluxes)                 # ~[100, 40]
print(np.sqrt(variances))     # 1-sigma per source
```

`update_catalog=True` writes the solved fluxes back, so a subsequent
`tractor.getModelImage(0)` renders the *fitted* model — handy for residuals:

```python
residual = tractor.images[0].data - np.asarray(tractor.getModelImage(0))
```

`solver="eigfloor"` is a good default in crowded fields; `"linear"` reproduces
the classic forced-photometry solve. See {doc}`solvers` for the full menu, and
{doc}`worked_example` for a scene where the choice actually changes the answer.

## Where to go next

- {doc}`worked_example` — blends, faint sources, and how the estimators differ.
- {doc}`solvers` — the math and the trade-offs.
- {doc}`architecture` — batching, tiling, sharding, and the render chain.
- {doc}`performance` — measured throughput and hardware sizing.
- {doc}`migration` — porting from The Tractor.

For everything else, the [API reference](api.md) lists the full public surface,
including the functional layer ({mod}`tractor_jax.jax.rendering`,
{mod}`tractor_jax.jax.batching`, {mod}`tractor_jax.jax.pipeline`).
