# Worked example

A complete fit you can run in a few seconds on a laptop CPU, showing where the
solvers agree and where they don't. The full script is
`examples/fit_blended_sources.py`:

```bash
JAX_PLATFORMS=cpu python examples/fit_blended_sources.py
```

## The scene

Five point sources of known flux on a 64×64 image with a Gaussian PSF
($\sigma = 1.6$ px). Two cases are deliberately hard:

- a **degenerate pair** 0.4 px apart — a quarter of a PSF sigma, so the data
  constrain their *sum* well and their *split* barely at all;
- a **faint source** at S/N ≈ 0.4, far below any detection threshold.

```python
import jax
jax.config.update("jax_enable_x64", True)     # before any array is created

from tractor_jax import (Catalog, ConstantSky, Flux, GaussianMixturePSF,
                         Image, NullWCS, PixPos, PointSource, Tractor)
from tractor_jax.jax.optimizer import optimize_fluxes

psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                         np.array([[[1.6**2, 0.0], [0.0, 1.6**2]]]))
catalog = Catalog(*[PointSource(PixPos(x, y), Flux(f)) for x, y, f in truth])
image = Image(data=data, inverr=np.full(data.shape, 1 / 0.02),
              psf=psf, wcs=NullWCS(), sky=ConstantSky(0.0))
tractor = Tractor([image], catalog)
```

## Fitting

One call solves every flux jointly. Positions and shapes stay frozen — this is
forced photometry:

```python
fluxes, variances = optimize_fluxes(
    tractor, solver="eigfloor", eig_floor=1e-2,
    return_variances=True, update_catalog=True)[0]
```

`update_catalog=True` writes the solved fluxes back into the catalog, so
`tractor.getModelImage(0)` then renders the *fitted* model:

```{image} _static/example_scene.png
:alt: data, fitted model, and residual for the synthetic scene
:width: 100%
```

The residual is structureless noise — the model reproduces the image. Note the
blended pair appears as a single blob: no amount of fitting can make the *image*
show two sources there.

## What the solvers do differently

Running the same scene through `linear`, `eigfloor`, and `lasso`:

```{image} _static/example_solvers.png
:alt: recovered over true flux per source for three solvers
:width: 100%
```

| source | truth | `linear` | `eigfloor` | `lasso` |
|---|---|---|---|---|
| isolated, S/N 44 | 5.00 | 4.916 ± 0.113 | 4.916 ± 0.113 | 4.916 ± 0.113 |
| isolated, S/N 18 | 2.00 | 1.841 ± 0.113 | 1.841 ± 0.113 | 1.841 ± 0.113 |
| faint, S/N 0.4 | 0.05 | 0.192 ± 0.113 | 0.192 ± 0.113 | **0.000** |
| blend A | 3.00 | 4.268 ± 0.647 | 3.828 ± 0.572 | 4.268 ± 0.647 |
| blend B | 1.50 | 0.244 ± 0.647 | 0.685 ± 0.572 | 0.244 ± 0.647 |
| **blend A + B** | **4.50** | **4.512** | **4.512** | **4.512** |

Three things to take away:

**Regularizers only act where the data are ambiguous.** On the two isolated
sources all three solvers return bit-identical fluxes. If your field is sparse,
the choice of solver is irrelevant — pick `linear` and move on.

**The blend's split is not measured; its sum is.** All three solvers recover the
pair's total to 0.3%, but the individual fluxes are off by up to 40% with error
bars (±0.647) that say exactly that — 5.7× larger than the isolated sources'.
`eigfloor` damps the split toward the middle, trading a little bias for
substantially lower variance (±0.572); `linear` and `lasso` let it wander. None
of them is "wrong": they report different points along the bias–variance
trade-off, and their error bars are honest about it.

**`lasso` zeroes the faint source, the others don't.** At S/N ≈ 0.4 the faint
source is not detected. `linear` and `eigfloor` return a signed noise value
(here $+0.192$; on other noise realizations it comes out negative) with an error
bar that covers zero — the statistically correct answer, and the one you must
keep if you plan to average or stack. `lasso` applies selection and returns
exactly 0. That is the right behaviour for a targeted measurement of bright
sources and the wrong behaviour for a blind survey, which is why
{doc}`solvers` recommends `eigfloor` as the blind default.

## Next steps

- {doc}`solvers` — the math behind each estimator and when to use it.
- {doc}`architecture` — how a fit is actually executed: rendering, batching,
  tiling, and multi-GPU.
- {doc}`performance` — measured throughput and how to size hardware.
