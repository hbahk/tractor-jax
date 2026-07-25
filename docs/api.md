# API reference

The package has two layers: **model classes** that mirror The Tractor's API, and
the **functional JAX layer** (`tractor_jax.jax`) that does the batched numerical
work. See {doc}`architecture` for how they fit together.

## Entry points

Most users need only these:

| what | where |
|---|---|
| build a scene | {class}`tractor_jax.engine.Image`, {class}`tractor_jax.engine.Catalog`, {class}`tractor_jax.engine.Tractor` |
| sources | {class}`tractor_jax.pointsource.PointSource`, {mod}`tractor_jax.galaxy`, {mod}`tractor_jax.sersic` |
| PSFs | {mod}`tractor_jax.psf` |
| **fit fluxes** | {func}`tractor_jax.jax.optimizer.optimize_fluxes` |
| solvers | {func}`~tractor_jax.jax.optimizer.solve_fluxes_linear`, {func}`~tractor_jax.jax.optimizer.solve_fluxes_eigfloor`, {func}`~tractor_jax.jax.optimizer.solve_fluxes_eigfloor_prior`, {func}`~tractor_jax.jax.optimizer.solve_fluxes_lasso` |
| batching many views | {func}`~tractor_jax.jax.batching.build_padded_batches`, {func}`~tractor_jax.jax.batching.make_batched_solver` |
| overlap host and device | {func}`~tractor_jax.jax.pipeline.prefetch_pipeline` |
| iterative (non-forced) fits | {class}`tractor_jax.jax.optimizer.JaxOptimizer` |

## Modules

### Model layer

```{eval-rst}
.. autosummary::
   :toctree: api
   :recursive:

   tractor_jax.engine
   tractor_jax.image
   tractor_jax.pointsource
   tractor_jax.galaxy
   tractor_jax.sersic
   tractor_jax.psf
   tractor_jax.brightness
   tractor_jax.ellipses
   tractor_jax.wcs
   tractor_jax.sky
   tractor_jax.basics
   tractor_jax.ducks
   tractor_jax.patch
   tractor_jax.optimize
   tractor_jax.imageutils
   tractor_jax.shifted
   tractor_jax.mixture_profiles
   tractor_jax.cache
   tractor_jax.miscutils
   tractor_jax.utils
```

### JAX layer

```{eval-rst}
.. autosummary::
   :toctree: api
   :recursive:

   tractor_jax.jax.optimizer
   tractor_jax.jax.rendering
   tractor_jax.jax.batching
   tractor_jax.jax.pipeline
   tractor_jax.jax.tiling
   tractor_jax.jax.tree
```
