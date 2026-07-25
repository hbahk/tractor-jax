# Tractor-JAX

```{image} _static/tractorjax-logo.svg
:alt: Tractor-JAX
:width: 460px
:class: only-light
```

```{image} _static/tractorjax-logo-dark.svg
:alt: Tractor-JAX
:width: 460px
:class: only-dark
```

**GPU-accelerated astronomical image modeling and forced photometry.**

Tractor-JAX reimplements [The Tractor](https://github.com/dstndstn/tractor)'s
probabilistic image model on JAX. It keeps the model classes you already know —
`Image`, `Catalog`, `PointSource`, `SersicGalaxy`, `PixelizedPSF` — and replaces
the fitting engine with batched, `jit`-compiled kernels that solve thousands of
blended sources per image on a GPU, with exact autodiff gradients and a menu of
regularized estimators.

It was built for SPHEREx-scale forced photometry: many epochs, many bands, deeply
crowded fields, fluxes free and everything else frozen.

```python
from tractor_jax import Tractor, Image, Catalog, PointSource, PixPos, Flux
from tractor_jax.jax.optimizer import optimize_fluxes

tractor = Tractor([image], catalog)
fluxes, variances = optimize_fluxes(
    tractor, solver="eigfloor", return_variances=True)[0]
```

```{image} _static/example_solvers.png
:alt: recovered flux per source for three solvers on a blended scene
:width: 100%
```

That figure comes from the {doc}`worked_example`, which runs on a laptop CPU in
seconds: the estimators agree exactly on well-measured sources and diverge only
where the data are genuinely ambiguous.

## Highlights

- **Batched GPU rendering and solves.** All images (or tiles) of a problem are
  padded to a common shape and solved in one `vmap`, with the compiled solver
  memoized across calls.
- **A menu of estimators.** Plain least squares, an eigenvalue floor for
  degenerate blends, Gaussian flux priors, and L1 with selection and debiasing —
  see {doc}`solvers`.
- **Undersampling handled correctly.** Sources are rendered on an oversampled
  grid and integrated back to native pixels; flux conservation is a tested
  contract.
- **Scales across devices.** Data-parallel sharding over the image batch (~99%
  efficiency on two GPUs) and a prefetch pipeline that hides host work behind the
  accelerator.

## Conventions

- Fluxes carry whatever units your `PhotoCal` defines; magnitudes are AB.
- WCS follows the FITS convention (1-based pixel origin, RA increasing to the
  left).
- The JAX layer is functional: no in-place mutation, shape-determining arguments
  are static.
- Precision is a global, up-front choice. float32 is the validated production
  default; call `jax.config.update("jax_enable_x64", True)` before any array is
  created when you need exactness under padding and batching.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
worked_example
```

```{toctree}
:maxdepth: 1
:caption: Guides

solvers
architecture
performance
migration
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```
