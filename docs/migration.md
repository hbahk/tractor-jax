# Coming from The Tractor

Tractor-JAX mirrors [The Tractor](https://github.com/dstndstn/tractor)'s model
API deliberately: `Image`, `Catalog`, `PointSource`, `PixPos`, `Flux`,
`GalaxyShape`, `PixelizedPSF` and friends keep their names and constructor
signatures, so scene-building code usually ports with an import change. What
differs is how you *fit*, and a few behavioural details worth knowing before you
compare numbers.

## What ports unchanged

```python
# The Tractor                              # Tractor-JAX
from tractor import (Image, Tractor,       from tractor_jax import (Image, Tractor,
    PointSource, PixPos, Flux,                 PointSource, PixPos, Flux,
    ConstantSky, NullWCS)                      ConstantSky, NullWCS)
from tractor.galaxy import GalaxyShape     from tractor_jax.galaxy import GalaxyShape
from tractor.sersic import SersicGalaxy    from tractor_jax.sersic import SersicGalaxy
```

Scene construction, `tractor.getModelImage(0)`, parameter freezing/thawing, and
the `Params` machinery all behave as you expect.

## What changes

### Fitting: `optimize_fluxes` instead of `optimize_forced_photometry`

The Tractor's `optimize_forced_photometry` is replaced by
{func}`~tractor_jax.jax.optimizer.optimize_fluxes`, which solves **all images in
one batched, compiled call** and lets you choose the estimator:

```python
from tractor_jax.jax.optimizer import optimize_fluxes

results = optimize_fluxes(tractor, solver="eigfloor", eig_floor=1e-2,
                          return_variances=True, update_catalog=True)
fluxes, variances = results[0]        # one entry per image
```

Both assume the same thing — positions and shapes frozen, linear in flux — but
`optimize_fluxes` returns a plain list of arrays rather than an "OptResult" duck,
and gives you variances directly instead of `.IV` (inverse variance). See
{doc}`solvers` for the estimator menu, which has no counterpart in the legacy
package.

For the iterative, non-forced case (thawing positions or shapes) use
{class}`~tractor_jax.jax.optimizer.JaxOptimizer` as the `Tractor`'s optimizer and
call `optimize()` / `optimize_loop()` as before.

### Images take `inverr`, not `invvar`

```python
Image(data=data, inverr=1.0 / sigma, psf=psf, wcs=wcs, sky=sky)
```

Passing an inverse *variance* where an inverse *error* is expected silently
squares your weights and shrinks every error bar — a quiet, plausible-looking
failure. Check this first when ported results disagree.

### An oversampled `PixelizedPSF` is normalized differently

For a PSF stamp with `sampling != 1`, the model must carry the $1/\text{sampling}^2$
pixel-area factor. The legacy `PixelizedPSF` omits it on the oversampled path, so
a unit-flux source renders to $\sim\text{sampling}^2$ of its flux (1/25 at
`sampling=0.2`) and forced fluxes come out ~25× **too high**. Tractor-JAX applies
the factor on both the point-source and the Fourier (galaxy) paths.

If you are comparing against a legacy run that used an oversampled stamp, that
factor — not the engine — is very likely the discrepancy. The safe check on
either side: render one unit-flux source and confirm the model sums to 1.

### Precision is a global, up-front decision

JAX defaults to float32. Enable float64 **before creating any array**:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Legacy Tractor is float64 throughout, so an apples-to-apples comparison needs x64
here. For production throughput, float32 fluxes are fine; variances used for
calibration are not (see {doc}`performance`).

### Rendering happens on a grid, in batch

Legacy Tractor renders each source into a `Patch` sized by a `ModelMask` or a
radius cut. Tractor-JAX renders every source in a batch into a common padded
grid so the work `vmap`s — there is no per-source patch clipping, and no
`minval`/`modelMask` plumbing. The practical consequence is that very extended
sources cost the same as compact ones, and that the batch shape (not the source
count) sets the memory.

## What is deliberately not here

- **Position/shape fitting is not the focus.** The engine is built for forced
  photometry: fluxes free, everything else frozen. `JaxOptimizer` can thaw other
  parameters, but the batched fast path is flux-only.
- **No `astrometry.net` dependency**, and no C extensions to build.
- **CPU is not a supported production target** — the design is GPU + batching.
  See {doc}`performance`.

## Porting checklist

1. Swap the imports (`tractor` → `tractor_jax`).
2. Change `invvar=` to `inverr=`.
3. Replace `optimize_forced_photometry(...)` with `optimize_fluxes(...)`;
   read variances from the second return value instead of `.IV`.
4. Set `jax_enable_x64` if you are validating against legacy numbers.
5. If you use an oversampled `PixelizedPSF`, verify unit-flux normalization on
   both sides before attributing any difference to the engine.
6. Pick a solver ({doc}`solvers`) — `linear` reproduces the legacy estimator.
