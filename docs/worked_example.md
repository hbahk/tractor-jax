# Worked example

A complete fit you can run in a couple of minutes on a laptop CPU, on a scene
built to look like the problem the engine was written for: a **critically
undersampled** detector, a catalog mixing stars and galaxies, and a blend the
data cannot take apart. The full script is `examples/fit_blended_sources.py`:

```bash
JAX_PLATFORMS=cpu python examples/fit_blended_sources.py
```

Every number on this page is printed by that script.

## The scene

A 24×24 pixel cutout at **6.15″/px** with a **6.2″ FWHM** PSF. The pixel is
wider than the PSF ($\sigma = 0.43$ px), so each source lands on two or three
pixels and no amount of centroiding recovers what happened inside one. The PSF
is handed to the engine as a 5× **oversampled** `PixelizedPSF`
(`sampling=0.2`): sources are rendered on the fine grid and integrated back to
native pixels, which is the production path for undersampled data.

Five sources, stars and Sérsic galaxies:

| source | truth flux | S/N | note |
|---|---|---|---|
| bright star | 5.00 | 48.7 | isolated |
| isolated galaxy | 4.00 | 25.9 | $n=1$, $r_e = 6''$, $b/a = 0.6$ |
| blend A (galaxy) | 2.50 | 25.2 | $n=2$, $r_e = 1.6''$ |
| blend B (star) | 1.00 | 10.6 | 0.92″ from A |
| faint star | 0.04 | 0.4 | below any detection threshold |

The S/N values are $f_\text{true}\sqrt{G_{jj}}$, computed from the templates
themselves rather than assumed.

The sky is simulated on an 8× finer grid (0.77″/px, well sampled, Gaussian
mixture PSF) and summed into detector pixels, so the fit is never scored against
its own renderer.

```{image} _static/example_scene.png
:alt: truth, data, fitted model and chi residual, wide and zoomed on the blend
:width: 100%
```

The top row is the whole cutout; the bottom row zooms on the blend. The point of
the left two panels: the sky (0.77″/px) contains a resolved galaxy and point-like
stars, and the detector turns each of them into a handful of fat pixels. The
blend is a galaxy and a star 0.92″ apart — 0.15 native pixels, 0.15 FWHM. That
pair is one blob in the truth panel too: at a 6.2″ FWHM, *no* imaging of this
field separates them. The residual is structureless noise.

## Fitting

One call solves every flux jointly. Positions and shapes stay frozen — this is
forced photometry:

```python
import jax
jax.config.update("jax_enable_x64", True)     # before any array is created

from tractor_jax import (Catalog, ConstantSky, Flux, GalaxyShape, Image,
                         NullWCS, PixelizedPSF, PixPos, PointSource, Tractor)
from tractor_jax.sersic import SersicGalaxy, SersicIndex
from tractor_jax.jax.optimizer import optimize_fluxes

psf = PixelizedPSF(oversampled_stamp, sampling=0.2)     # 5x oversampled
image = Image(data=data, inverr=np.full(data.shape, 1 / 0.05),
              psf=psf, wcs=NullWCS(pixscale=6.15), sky=ConstantSky(0.0))
catalog = Catalog(PointSource(PixPos(5.5, 18.4), Flux(5.0)),
                  SersicGalaxy(PixPos(11.2, 7.15), Flux(2.5),
                               GalaxyShape(1.6, 0.7, 60.0), SersicIndex(2.0)),
                  ...)

fluxes, variances = optimize_fluxes(
    Tractor([image], catalog), solver="eigfloor", eig_floor=1e-2,
    oversample_rendering=True, return_variances=True, update_catalog=True)[0]
```

`oversample_rendering=True` activates the fine-grid path for the undersampled
PSF; `update_catalog=True` writes the solved fluxes back, so
`tractor.getModelImage(0)` then renders the *fitted* model — the third panel
above.

## What the solvers do differently

```{image} _static/example_solvers.png
:alt: recovered over true flux per source for three solvers
:width: 100%
```

| source | truth | `linear` | `eigfloor` | `lasso` |
|---|---|---|---|---|
| bright star, S/N 49 | 5.00 | 5.107 ± 0.102 | 5.107 ± 0.102 | 5.107 ± 0.102 |
| isolated galaxy, S/N 26 | 4.00 | 4.190 ± 0.154 | 4.190 ± 0.154 | 4.190 ± 0.154 |
| blend A (galaxy), S/N 25 | 2.50 | 3.417 ± 0.681 | 2.648 ± 0.498 | 3.417 ± 0.681 |
| blend B (star), S/N 11 | 1.00 | 0.112 ± 0.645 | 0.841 ± 0.472 | 0.112 ± 0.645 |
| faint star, S/N 0.4 | 0.04 | 0.066 ± 0.092 | 0.066 ± 0.092 | **0.000** |
| **blend A + B** | **3.50** | **3.529** | **3.488** | **3.529** |

Three things to take away.

**Regularizers only act where the data are ambiguous.** On the bright star and
the isolated galaxy all three solvers return bit-identical fluxes and identical
error bars. If your field is sparse, the choice of solver is irrelevant — pick
`linear` and move on.

**The blend's split is not measured; its sum is.** All three solvers recover the
pair's total to better than 1%, while the individual fluxes are off by up to
$-89\%$ — on this noise realization `linear` puts essentially the whole blend
into the galaxy and leaves the star at 0.112 against a truth of 1.00. The error
bars say so: ±0.645 is 6.9× the ±0.094 the same star would get if it were
isolated. `eigfloor` damps the split and lands at 0.841, trading bias for
variance; `linear` and `lasso` let it wander. None is "wrong" — they are
different points on the bias–variance trade-off, and the next two sections
quantify both ends.

**`lasso` zeroes the faint source, the others don't.** At S/N ≈ 0.4 the faint
star is not detected. `linear` and `eigfloor` return a signed noise value (here
$+0.066$; on other realizations it comes out negative) with an error bar covering
zero — the statistically correct answer, and the one you must keep if you plan
to stack or average. `lasso` applies selection and returns exactly 0 with an
infinite reported variance. That is right for a targeted measurement of bright
sources and wrong for a blind survey, which is why {doc}`solvers` recommends
`eigfloor` as the blind default.

## Reading the fit through the eigenmodes

The script also prints the eigen-decomposition of the Jacobi-normalized Gram
$\hat G$, whose coordinates are S/N (see {doc}`eigfloor`):

```
eigenvalues of Ghat:   0.0107  1.0000  1.0000  1.0000  1.9893
blend template overlap  rho   = 0.9893   (engagement threshold 0.9802)
sum mode   (1, 1)/sqrt2 lam_+ = 1.9893
split mode (1,-1)/sqrt2 lam_- = 0.0107
floor 1e-2 * lam_max    lam_f = 0.0199
filter factors          phi_+ = 1.0000,  phi_- = 0.5366
```

Read it left to right. Three eigenvalues sit at exactly 1: those are the three
isolated sources, each carrying exactly the information of one unblended
measurement. The remaining two belong to the pair, and they are the 2×2 blend
theory verbatim: $\lambda_\pm = 1 \pm \rho$. The pair's template overlap is
$\rho = 0.9893$, above the engagement threshold
$(1-\texttt{floor})/(1+\texttt{floor}) = 0.980$, so the floor bites — but only
on the split mode, whose filter factor is $\varphi_- = 0.537$; the sum mode's is
$\varphi_+ = 1$ exactly.

Projecting the actual solver outputs onto those two modes confirms it:

| mode | `linear` | `eigfloor` | ratio | predicted $\varphi$ |
|---|---|---|---|---|
| sum $(1,1)/\sqrt2$ | 25.181 | 25.183 | 1.0001 | 1.0000 |
| split $(1,-1)/\sqrt2$ | 23.498 | 12.532 | 0.5333 | 0.5366 |

The total flux passes through untouched and the split is scaled by exactly the
filter factor. Nothing else happens.

## Stability under repeated observations

One realization cannot show the difference between bias and variance. Refit the
*same* scene on 120 independent noise realizations — the synthetic twin of
observing one field in ~100 spectral channels:

```{image} _static/example_stability.png
:alt: blend split and blend sum over 120 noise realizations, linear vs eigfloor
:width: 100%
```

| quantity | truth | `linear` | `eigfloor` |
|---|---|---|---|
| blend B (star) | 1.00 | 0.990 ± 0.633 | 1.312 ± 0.342 |
| blend A + B (sum) | 3.50 | 3.494 ± 0.101 | 3.476 ± 0.099 |
| bright star (isolated) | 5.00 | 4.992 ± 0.096 | 4.992 ± 0.096 |

The top row of the figure is the unmeasured direction. `linear` is unbiased
(0.990 against a truth of 1.00) and wild: it swings from $-0.30$ to $3.01$,
crossing zero and passing three times the true flux. `eigfloor` is biased toward
the middle of the blend (1.312) and has 1.85× smaller scatter. That is the whole
bias–variance trade, measured.

The bottom row is the measured direction, and it is the more important panel:
the two solvers are *on top of each other*, both unbiased, both at a scatter of
±0.10 — the same ±0.10 an isolated source of that brightness would have. The
explosion lives entirely in the split; the sum never notices that a regularizer
was applied.

This is the synthetic version of what blended sources do on real survey data: a
per-channel least-squares SED of a blended galaxy oscillates by a large fraction
of its own flux with correspondingly huge error bars, while the same catalog run
through `eigfloor` produces a smooth spectrum.

### Are the error bars honest?

The same 120 fits, comparing the reported $\sigma$ with the actual scatter:

| source | solver | reported | actual | reported/actual | bias |
|---|---|---|---|---|---|
| blend B (star) | `linear` | 0.645 | 0.633 | 1.02 | $-0.02\sigma$ |
| blend B (star) | `eigfloor` | 0.472 | 0.342 | 1.38 | $+0.66\sigma$ |
| bright star | `linear` | 0.102 | 0.096 | 1.06 | $-0.07\sigma$ |
| bright star | `eigfloor` | 0.102 | 0.096 | 1.06 | $-0.07\sigma$ |

Two things to notice, both predicted exactly in {doc}`eigfloor`:

- on a floored mode the reported $\sigma$ is **conservative** by
  $1/\sqrt{\varphi_-} = 1/\sqrt{0.537} = 1.37$; measured, 1.38;
- the **shrinkage bias is not in the error bar**. `eigfloor` sits $+0.66\sigma$
  from the truth on the blend's fainter member while reporting an error bar that
  is, if anything, too large. This is why a survey pipeline that uses `eigfloor`
  still needs a per-S/N error calibration measured on simulations.

## Next steps

- {doc}`eigfloor` — where $\rho$, $\lambda_\pm$ and $\varphi$ come from, and what
  the floor is doing as a prior.
- {doc}`solvers` — the math behind each estimator and when to use it.
- {doc}`architecture` — how a fit is actually executed: rendering, batching,
  tiling, and multi-GPU.
- {doc}`performance` — measured throughput and how to size hardware.
