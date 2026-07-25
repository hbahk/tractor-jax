"""Worked example: forced photometry on a blended, degenerate scene.

Fits five point sources of known flux with three solvers and shows where they
agree and where they don't. The scene is built so that both interesting cases
are present:

* a **degenerate pair** 0.4 px apart (0.25 PSF sigma) — the data constrain
  their *sum* but barely their *split*;
* a **faint source** at S/N ~ 0.4 — below any sensible detection threshold.

Writes the two figures used by the docs:

    docs/_static/example_scene.png      data / fitted model / residual
    docs/_static/example_solvers.png    recovered/true flux per solver

Runs on CPU in a few seconds:

    JAX_PLATFORMS=cpu python examples/fit_blended_sources.py
"""

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Calibration-grade variances want float64; set it before any array is made.
jax.config.update("jax_enable_x64", True)

from tractor_jax import (Catalog, ConstantSky, Flux, GaussianMixturePSF, Image,
                         NullWCS, PixPos, PointSource, Tractor)
from tractor_jax.jax.optimizer import optimize_fluxes

OUT = Path(__file__).resolve().parents[1] / "docs" / "_static"
H = W = 64
PSF_SIGMA = 1.6
NOISE = 0.02
SEED = 0

# (x, y, flux, label)
TRUTH = [
    (12.0, 14.0, 5.00, "isolated\nS/N 44"),
    (48.0, 20.0, 2.00, "isolated\nS/N 18"),
    (20.0, 46.0, 0.05, "faint\nS/N 0.4"),
    (36.0, 40.0, 3.00, "blend A\n0.4 px apart"),
    (36.4, 40.0, 1.50, "blend B\n0.4 px apart"),
]
BLEND = (3, 4)

SOLVERS = [
    ("linear", {}, "C3", "o"),
    ("eigfloor", dict(eig_floor=1e-2), "C2", "^"),
    ("lasso", dict(penalty={"alpha": "auto"}), "C0", "s"),
]


def make_scene(seed=SEED):
    """A noisy realization of the scene; returns (tractor, true fluxes)."""
    psf = GaussianMixturePSF(
        np.array([1.0]), np.zeros((1, 2)),
        np.array([[[PSF_SIGMA ** 2, 0.0], [0.0, PSF_SIGMA ** 2]]]),
    )
    catalog = Catalog(*[PointSource(PixPos(x, y), Flux(f))
                        for x, y, f, _ in TRUTH])
    image = Image(data=np.zeros((H, W)), inverr=np.full((H, W), 1.0 / NOISE),
                  psf=psf, wcs=NullWCS(), sky=ConstantSky(0.0))
    tractor = Tractor([image], catalog)

    clean = np.asarray(tractor.getModelImage(0))       # noiseless truth
    rng = np.random.default_rng(seed)
    tractor.images[0].data = clean + rng.normal(0.0, NOISE, (H, W))
    return tractor, np.array([f for _, _, f, _ in TRUTH])


def fit(solver, **kw):
    """Fit fluxes with one solver -> (fluxes, sigmas, fitted model image)."""
    tractor, _ = make_scene()
    fluxes, variances = optimize_fluxes(
        tractor, solver=solver, return_variances=True,
        update_catalog=True, use_sharding=False, **kw)[0]
    model = np.asarray(tractor.getModelImage(0))
    return (np.asarray(fluxes),
            np.sqrt(np.maximum(np.asarray(variances), 0.0)), model)


def plot_scene(data, model, path):
    """data / fitted model / residual triptych."""
    resid = (data - model) / NOISE
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    panels = ((axs[0], data, "data", dict(vmin=data.min(), vmax=data.max())),
              (axs[1], model, "fitted model",
               dict(vmin=data.min(), vmax=data.max())),
              (axs[2], resid, r"residual / $\sigma$",
               dict(vmin=-4, vmax=4, cmap="RdBu_r")))
    for ax, img, title, kw in panels:
        m = ax.imshow(img, origin="lower", interpolation="nearest", **kw)
        ax.set_title(title)
        fig.colorbar(m, ax=ax, shrink=0.85)
    for x, y, _, _ in TRUTH:
        for ax in axs[:2]:
            ax.scatter([x], [y], s=110, facecolors="none", edgecolors="w",
                       linewidths=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solvers(truth, results, path):
    """Recovered/true flux per source, one series per solver."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    idx = np.arange(len(truth))
    for k, (solver, _, color, marker) in enumerate(SOLVERS):
        f, s, _ = results[solver]
        off = (k - 1) * 0.22
        ax.errorbar(idx + off, f / truth, yerr=s / truth, fmt=marker, ms=6,
                    lw=0, elinewidth=1.3, capsize=3, color=color, label=solver)
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.axhline(0.0, color="0.85", lw=1)
    ax.set_xticks(idx)
    ax.set_xticklabels([lab for _, _, _, lab in TRUTH], fontsize="small")
    ax.set_ylabel("recovered / true flux")
    ax.set_ylim(-0.6, 4.6)
    ax.legend(title="solver", loc="upper left")

    i, j = BLEND
    sums = {s: results[s][0][i] + results[s][0][j] for s, _, _, _ in SOLVERS}
    ax.text(0.98, 0.96,
            "blend A+B (the constrained quantity):\n"
            + "   ".join(f"{s} {v:.3f}" for s, v in sums.items())
            + f"\ntruth {truth[i] + truth[j]:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize="small",
            color="0.25")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tractor, truth = make_scene()
    data = np.asarray(tractor.images[0].data)

    results = {}
    for solver, kw, _, _ in SOLVERS:
        results[solver] = fit(solver, **kw)

    print(f"{'source':>22} {'truth':>7} " +
          " ".join(f"{s:>16}" for s, _, _, _ in SOLVERS))
    for n, (_, _, t, lab) in enumerate(TRUTH):
        row = " ".join(f"{results[s][0][n]:8.3f}±{results[s][1][n]:6.3f}"
                       for s, _, _, _ in SOLVERS)
        print(f"{lab.replace(chr(10), ' '):>22} {t:7.2f} {row}")
    i, j = BLEND
    print(f"{'blend A+B (sum)':>22} {truth[i] + truth[j]:7.2f} " +
          " ".join(f"{results[s][0][i] + results[s][0][j]:8.3f}{'':7}"
                   for s, _, _, _ in SOLVERS))

    plot_scene(data, results["eigfloor"][2], OUT / "example_scene.png")
    plot_solvers(truth, results, OUT / "example_solvers.png")
    print(f"\nwrote {OUT / 'example_scene.png'}")
    print(f"wrote {OUT / 'example_solvers.png'}")


if __name__ == "__main__":
    main()
