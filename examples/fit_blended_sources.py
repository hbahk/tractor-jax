"""Worked example: forced photometry on an undersampled, blended scene.

A SPHEREx-like setup — 6.15"/px detector, a 6.2" FWHM PSF, so the pixel is
*wider* than the PSF and every source is a two-or-three-pixel blob.  The PSF is
handed to the engine as a 5x **oversampled** ``PixelizedPSF`` (``sampling=0.2``);
sources are rendered on the fine grid and integrated back to native pixels.

The catalog mixes stars and Sersic galaxies and contains the two cases that make
solver choice matter:

* **the blend** — a compact Sersic galaxy and a star ~0.9" apart (0.15 native
  px).  Their unit-flux templates overlap at rho ~ 0.99, so the data pin the
  pair's *total* flux and say almost nothing about the *split*;
* **a faint star** at S/N ~ 0.4 — below any sensible detection threshold.

Three figures are written for the docs:

    docs/_static/example_scene.png      truth / data / model / chi
    docs/_static/example_solvers.png    recovered/true flux per solver
    docs/_static/example_stability.png  120 noise realizations, linear vs eigfloor

Runs on a laptop CPU in a couple of minutes (most of it the 240 solves of the
stability run):

    JAX_PLATFORMS=cpu python examples/fit_blended_sources.py
"""

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

# fp32 is the production default and is calibrated; x64 here only so the
# printed numbers are exactly reproducible. Must precede any array creation.
jax.config.update("jax_enable_x64", True)

from tractor_jax import (Catalog, ConstantSky, Flux, GalaxyShape,
                         GaussianMixturePSF, Image, NullWCS, PixelizedPSF,
                         PixPos, PointSource, Tractor)
from tractor_jax.jax.optimizer import optimize_fluxes
from tractor_jax.sersic import SersicGalaxy, SersicIndex

OUT = Path(__file__).resolve().parents[1] / "docs" / "_static"

# ---------------------------------------------------------------- instrument
PIXSCALE = 6.15          # native detector pixel, arcsec
PSF_FWHM = 6.2           # delivered PSF FWHM, arcsec  -> sigma = 0.43 native px
OVERSAMP = 5             # PSF stamp oversampling (PixelizedPSF sampling = 1/5)
H = W = 24               # native cutout, pixels
FINE = 8                 # truth-image refinement factor (display only)
NOISE = 0.05             # per-pixel sigma, same flux units as the catalog
SEED = 37                # a +1.4 sigma excursion along the blend's split mode

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493

# ------------------------------------------------------------------- catalog
# Native-pixel positions.  The blend sits at (BX, BY); its stellar companion is
# SEP native px away along ANG.  SEP was tuned so that the pair's normalized
# template overlap lands at rho ~ 0.99, just inside the eigfloor's engagement
# threshold rho > (1 - floor) / (1 + floor) = 0.980 at floor = 1e-2.
BX, BY, SEP, ANG = 11.2, 7.15, 0.15, 35.0

# (kind, x, y, flux, shape, label);  shape = (re_arcsec, ab, phi_deg, sersic_n)
TRUTH = [
    ("star", 5.5, 18.4, 5.00, None, "bright star"),
    ("gal", 17.6, 18.2, 4.00, (6.0, 0.60, 35.0, 1.0), "isolated galaxy"),
    ("gal", BX, BY, 2.50, (1.6, 0.70, 60.0, 2.0), "blend A (galaxy)"),
    ("star", BX + SEP * np.cos(np.deg2rad(ANG)),
     BY + SEP * np.sin(np.deg2rad(ANG)), 1.00, None, "blend B (star)"),
    ("star", 18.9, 5.6, 0.04, None, "faint star"),
]
BLEND = (2, 3)
ZOOM = 4.5               # half-width of the blend close-up, native pixels

SOLVERS = [
    ("linear", {}, "C3", "o"),
    ("eigfloor", dict(eig_floor=1e-2), "C2", "^"),
    ("lasso", dict(penalty={"alpha": "auto"}), "C0", "s"),
]
EIG_FLOOR = 1e-2


# ------------------------------------------------------------------ building
def oversampled_psf_stamp(n=41):
    """Gaussian PSF sampled OVERSAMP x finer than the native pixel."""
    sigma = PSF_FWHM * FWHM_TO_SIGMA / PIXSCALE * OVERSAMP   # in stamp pixels
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma ** 2))
    return (p / p.sum()).astype(np.float32)


def sources(scale=1, fluxes=None):
    """Catalog on a grid `scale` times finer than the native detector.

    Positions are converted (a native pixel centre at x maps to
    ``scale * x + (scale - 1) / 2`` on the fine grid); galaxy shapes are in
    arcsec and are converted by each image's own WCS, so they need no change.
    """
    out = []
    for k, (kind, x, y, f, shp, _) in enumerate(TRUTH):
        if fluxes is not None:
            f = fluxes[k]
        xs = scale * x + (scale - 1) / 2.0
        ys = scale * y + (scale - 1) / 2.0
        if kind == "star":
            out.append(PointSource(PixPos(xs, ys), Flux(f)))
        else:
            re, ab, phi, n = shp
            out.append(SersicGalaxy(PixPos(xs, ys), Flux(f),
                                    GalaxyShape(re, ab, phi), SersicIndex(n)))
    return out


def native_image(data=None):
    """The image we actually fit: undersampled, oversampled PixelizedPSF."""
    psf = PixelizedPSF(oversampled_psf_stamp(), sampling=1.0 / OVERSAMP)
    return Image(data=np.zeros((H, W)) if data is None else data,
                 inverr=np.full((H, W), 1.0 / NOISE), psf=psf,
                 wcs=NullWCS(pixscale=PIXSCALE), sky=ConstantSky(0.0))


def fine_templates():
    """Unit-flux render of every source on a FINE x finer, WELL-SAMPLED grid.

    This is the little instrument simulator that produces the "sky": the scene
    is built at 0.77"/px, where a Gaussian-mixture PSF is perfectly adequate,
    and the detector image is obtained by summing FINE x FINE blocks.  Nothing
    here uses the oversampled-PixelizedPSF machinery that the fit exercises, so
    the fit is not being scored against its own renderer.
    """
    sigma = PSF_FWHM * FWHM_TO_SIGMA / (PIXSCALE / FINE)      # in fine pixels
    psf = GaussianMixturePSF(np.array([1.0]), np.zeros((1, 2)),
                             np.array([[[sigma ** 2, 0.0], [0.0, sigma ** 2]]]))
    img = Image(data=np.zeros((H * FINE, W * FINE)),
                inverr=np.ones((H * FINE, W * FINE)), psf=psf,
                wcs=NullWCS(pixscale=PIXSCALE / FINE), sky=ConstantSky(0.0))
    unit = [1.0] * len(TRUTH)
    return np.array([np.asarray(Tractor([img], [s]).getModelImage(0))
                     for s in sources(FINE, unit)])


def bin_to_native(fine):
    """Sum FINE x FINE blocks: the detector integrating over its pixels."""
    return fine.reshape(*fine.shape[:-2], H, FINE, W, FINE).sum(axis=(-3, -1))


def make_data(templates, truth_flux, seed=SEED):
    """One noisy realization of the native image."""
    clean = np.tensordot(truth_flux, templates, axes=1)
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, NOISE, (H, W))


# ------------------------------------------------------------------- fitting
def fit(data, solver, **kw):
    """Fit fluxes on one realization -> (fluxes, sigmas, fitted model)."""
    tractor = Tractor([native_image(data)], Catalog(*sources(1)))
    fluxes, variances = optimize_fluxes(
        tractor, solver=solver, return_variances=True, update_catalog=True,
        oversample_rendering=True, use_sharding=False, **kw)[0]
    model = np.asarray(tractor.getModelImage(0))
    return (np.asarray(fluxes),
            np.sqrt(np.maximum(np.asarray(variances), 0.0)), model)


def mode_analysis(T):
    """The 2x2 blend theory, evaluated on the real Gram matrix."""
    T = T.reshape(len(TRUTH), -1)
    G = (T @ T.T) / NOISE ** 2                    # A^T W A
    D = np.sqrt(np.diag(G))                       # 1/sigma of an isolated fit
    Ghat = G / np.outer(D, D)                     # correlation matrix
    lam = np.linalg.eigvalsh(Ghat)
    i, j = BLEND
    rho = Ghat[i, j]
    lam_p, lam_m = 1.0 + rho, 1.0 - rho
    lam_f = EIG_FLOOR * lam.max()
    return dict(G=G, D=D, Ghat=Ghat, lam=lam, rho=rho, lam_p=lam_p,
                lam_m=lam_m, lam_max=lam.max(), lam_f=lam_f,
                phi_m=min(1.0, lam_m / lam_f), phi_p=min(1.0, lam_p / lam_f))


# ------------------------------------------------------------------- figures
def to_grid(x, y, scale):
    """Native-pixel coordinates -> coordinates on a `scale` x finer grid."""
    return scale * x + (scale - 1) / 2.0, scale * y + (scale - 1) / 2.0


def plot_scene(truth, data, model, path):
    """truth (fine, noiseless) / data / fitted model / chi, wide and zoomed."""
    truth_native = truth * FINE ** 2          # -> flux per native pixel area
    hi = float(max(truth_native.max(), data.max()))
    norm = AsinhNorm(linear_width=0.06 * hi, vmin=-4 * NOISE, vmax=hi)
    ticks = [0.0] + [t for t in (0.1, 0.3, 1.0, 3.0) if t < hi]
    chi = (data - model) / NOISE

    fig, axs = plt.subplots(2, 4, figsize=(13.6, 7.0), constrained_layout=True)
    panels = [
        (truth_native, FINE, f"truth: the sky at {PIXSCALE / FINE:.2f}\"/px "
         "(noiseless)", dict(norm=norm)),
        (data, 1, f"data: {PIXSCALE:.2f}\"/px detector + noise", dict(norm=norm)),
        (model, 1, "fitted model (eigfloor)", dict(norm=norm)),
        (chi, 1, r"$\chi$ = (data $-$ model) / $\sigma$",
         dict(vmin=-4, vmax=4, cmap="RdBu_r")),
    ]
    for col, (img, scale, title, kw) in enumerate(panels):
        for row in (0, 1):
            ax = axs[row][col]
            m = ax.imshow(img, origin="lower", interpolation="nearest", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize="medium")
            else:
                x0, y0 = to_grid(BX - ZOOM, BY - ZOOM, scale)
                x1, y1 = to_grid(BX + ZOOM, BY + ZOOM, scale)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                for s in ax.spines.values():
                    s.set(color="C1", linewidth=1.6)
            if col == 3:
                continue
            for kind, x, y, _, _, _ in TRUTH:
                xs, ys = to_grid(x, y, scale)
                r = (1.3 if row == 1 else 1.1) * scale       # ~native px radius
                ax.add_patch(plt.Circle((xs, ys), r, fill=False, color="w",
                                        lw=0.9, alpha=0.85))
        if col < 3:
            m_flux = m
    cb = fig.colorbar(m_flux, ax=axs[:, :3].ravel().tolist(), shrink=0.45,
                      location="bottom", pad=0.015, ticks=ticks,
                      format=lambda v, _: f"{v:g}")
    cb.set_label("flux per native pixel (asinh stretch)", fontsize="small")
    cb.ax.tick_params(labelsize="small")
    cb2 = fig.colorbar(m, ax=axs[:, 3].ravel().tolist(), shrink=0.85,
                       location="bottom", pad=0.015)
    cb2.ax.tick_params(labelsize="small")

    # source labels, on the wide truth panel only
    for (kind, x, y, _, _, _), (dx, dy, lab) in zip(
            TRUTH, [(0, 2.0, "bright star"), (0, 2.0, "isolated galaxy"),
                    (0, -2.2, "blend A + B"), (0, 0, None),
                    (0, -2.2, "faint star")]):
        if lab is None:
            continue
        xs, ys = to_grid(x + dx, y + dy, FINE)
        axs[0][0].annotate(lab, (xs, ys), color="w", fontsize="small",
                           ha="center", va="bottom" if dy > 0 else "top")

    # the blend, spelled out on the zoomed truth panel
    xa, ya = to_grid(*TRUTH[BLEND[0]][1:3], FINE)
    axs[1][0].annotate("galaxy + star,\n"
                       f"{SEP * PIXSCALE:.2f}\" = {SEP:.2f} px apart:\n"
                       "one blob at any resolution",
                       (xa, ya), xytext=(xa - 1.0 * FINE, ya - 3.2 * FINE),
                       color="w", fontsize="small", ha="center", va="top",
                       arrowprops=dict(arrowstyle="->", color="w", lw=0.9))
    axs[1][1].annotate(f"{2 * ZOOM:.0f} x {2 * ZOOM:.0f} native pixels",
                       (0.5, 0.03), xycoords="axes fraction", color="w",
                       fontsize="small", ha="center", va="bottom")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_solvers(truth, results, modes, path):
    """Recovered/true flux per source, one series per solver.

    The four detected sources share a "recovered / true" axis; the
    undetected one gets its own absolute-flux panel, because dividing a
    noise value by a tiny truth flux is meaningless.
    """
    det = [k for k in range(len(truth)) if modes["snr"][k] > 1.0]
    und = [k for k in range(len(truth)) if modes["snr"][k] <= 1.0]
    fig, (ax, axf) = plt.subplots(
        1, 2, figsize=(9.6, 4.6), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[len(det), 1.15]))

    for k, (solver, _, color, marker) in enumerate(SOLVERS):
        f, s, _ = results[solver]
        ax.errorbar(np.arange(len(det)) + (k - 1) * 0.22, f[det] / truth[det],
                    yerr=s[det] / truth[det], fmt=marker, ms=6, lw=0,
                    elinewidth=1.3, capsize=3, color=color, label=solver)
        # a zeroed (selected-out) source has infinite reported variance: no bar
        axf.errorbar([(k - 1) * 0.22], f[und],
                     yerr=np.where(np.isfinite(s[und]), s[und], 0.0),
                     fmt=marker, ms=6, lw=0, elinewidth=1.3, capsize=3,
                     color=color)
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.axhline(0.0, color="0.85", lw=1)
    ax.set_xticks(np.arange(len(det)))
    ax.set_xticklabels([f"{TRUTH[k][5]}\nS/N {modes['snr'][k]:.0f}" for k in det],
                       fontsize="small")
    ax.set_ylabel("recovered / true flux")
    ax.set_ylim(-0.78, 1.95)
    ax.legend(title="solver", loc="upper left", fontsize="small", ncol=3)

    axf.axhline(truth[und[0]], color="0.6", lw=1, ls="--")
    axf.axhline(0.0, color="0.85", lw=1)
    axf.set_xticks([0.0])
    axf.set_xticklabels([f"{TRUTH[und[0]][5]}\nS/N {modes['snr'][und[0]]:.1f}"],
                        fontsize="small")
    axf.set_xlim(-0.55, 0.55)
    lim = 3.6 * float(results["linear"][1][und[0]])
    axf.set_ylim(-lim, lim)
    axf.set_ylabel("flux (truth dashed)")
    axf.yaxis.set_label_position("right")
    axf.yaxis.tick_right()
    axf.text(0.5, 0.03, "lasso returns exactly 0\n(selected out, no error bar)",
             transform=axf.transAxes, fontsize="small", color="C0",
             ha="center", va="bottom")

    i, j = BLEND
    sums = {s: results[s][0][i] + results[s][0][j] for s, _, _, _ in SOLVERS}
    ax.text(0.985, 0.965,
            r"blend A+B, the measured direction ($\lambda_+$ = "
            f"{modes['lam_p']:.2f}):\n"
            + "   ".join(f"{s} {v:.3f}" for s, v in sums.items())
            + f"\ntruth {truth[i] + truth[j]:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize="small",
            color="0.25")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_stability(truth, split, ssum, modes, path):
    """120 noise realizations: the split explodes, the sum does not."""
    i, j = BLEND
    n = split["linear"].size
    x = np.arange(n)
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 6.4), sharey="row",
                            gridspec_kw=dict(width_ratios=[3.2, 1.0]),
                            constrained_layout=True)
    rows = [
        (axs[0], split, truth[j], "blend B (star) flux", True),
        (axs[1], ssum, truth[i] + truth[j], "blend A + B (sum)", False),
    ]
    for (axl, axr), series, tval, ylab, spiky in rows:
        if spiky:
            axl.axhline(0.0, color="0.85", lw=1)
        lo = min(v.min() for v in series.values())
        hi = max(v.max() for v in series.values())
        pad = 0.08 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 26)
        for solver, color in (("linear", "C3"), ("eigfloor", "C2")):
            v = series[solver]
            axl.plot(x, v, color=color, lw=0.9 if spiky else 1.1,
                     marker=".", ms=3.2, alpha=0.9,
                     label=f"{solver}  (std {v.std(ddof=1):.3f})")
            axr.hist(v, bins=bins, orientation="horizontal", color=color,
                     alpha=0.5, histtype="stepfilled")
        axl.axhline(tval, color="0.25", lw=1.2, ls="--", label="truth")
        axr.axhline(tval, color="0.25", lw=1.2, ls="--")
        axl.set_ylabel(ylab)
        axl.set_xlim(-1, n)
        axl.legend(fontsize="small", ncol=3, loc="upper left",
                   framealpha=0.9)
        axr.set_xticks([])
        axr.tick_params(labelleft=False)
    axs[0][0].set_title(
        r"the unmeasured direction: $\lambda_-$ = "
        f"{modes['lam_m']:.4f}, damped by $\\varphi_-$ = {modes['phi_m']:.2f}",
        fontsize="medium")
    axs[1][0].set_title(
        r"the measured direction: $\lambda_+$ = "
        f"{modes['lam_p']:.2f}, untouched by the floor", fontsize="medium")
    axs[1][0].set_xlabel("noise realization "
                         "(the synthetic twin of one field's spectral channels)")
    axs[0][1].set_title("distribution", fontsize="small")
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth_flux = np.array([f for _, _, _, f, _, _ in TRUTH])

    U = fine_templates()                       # (n, H*FINE, W*FINE), unit flux
    T = bin_to_native(U)                       # (n, H, W), the native templates
    truth_fine = np.tensordot(truth_flux, U, axes=1)
    modes = mode_analysis(T)
    modes["snr"] = truth_flux * modes["D"]

    data = make_data(T, truth_flux)
    results = {s: fit(data, s, **kw) for s, kw, _, _ in SOLVERS}

    # -------------------------------------------------- per-source results
    print(f"\nundersampled scene: {H}x{W} px at {PIXSCALE}\"/px, "
          f"PSF FWHM {PSF_FWHM}\" = {PSF_FWHM * FWHM_TO_SIGMA / PIXSCALE:.2f} px "
          f"sigma, PixelizedPSF(sampling={1.0 / OVERSAMP})")
    print(f"blend separation {SEP:.2f} px = {SEP * PIXSCALE:.2f}\" = "
          f"{SEP * PIXSCALE / PSF_FWHM:.2f} FWHM\n")
    print(f"{'source':>18} {'truth':>7} {'S/N':>6} " +
          " ".join(f"{s:>17}" for s, _, _, _ in SOLVERS))
    for n, (_, _, _, t, _, lab) in enumerate(TRUTH):
        row = " ".join(f"{results[s][0][n]:8.3f}±{results[s][1][n]:7.3f}"
                       for s, _, _, _ in SOLVERS)
        print(f"{lab:>18} {t:7.2f} {modes['snr'][n]:6.1f} {row}")
    i, j = BLEND
    print(f"{'blend A+B (sum)':>18} {truth_flux[i] + truth_flux[j]:7.2f} "
          f"{modes['snr'][i] + modes['snr'][j]:6.1f} " +
          " ".join(f"{results[s][0][i] + results[s][0][j]:8.3f}{'':8}"
                   for s, _, _, _ in SOLVERS))

    # ------------------------------------------------------ mode analysis
    print("\neigenmode analysis (Jacobi-normalized Gram; coordinates are S/N)")
    print("  eigenvalues of Ghat:      " +
          " ".join(f"{v:.4f}" for v in modes["lam"]))
    print(f"  blend template overlap    rho      = {modes['rho']:.4f}"
          f"   (engagement threshold {(1 - EIG_FLOOR) / (1 + EIG_FLOOR):.4f})")
    print(f"  sum mode    (1, 1)/sqrt2  lam_+    = {modes['lam_p']:.4f}")
    print(f"  split mode  (1,-1)/sqrt2  lam_-    = {modes['lam_m']:.4f}")
    print(f"  floor       {EIG_FLOOR:g} * lam_max  lam_f    = {modes['lam_f']:.4f}"
          f"   (lam_max = {modes['lam_max']:.4f})")
    print(f"  filter factors            phi_+    = {modes['phi_p']:.4f}, "
          f"phi_- = {modes['phi_m']:.4f}")

    lin, eig = results["linear"][0], results["eigfloor"][0]
    D = modes["D"]
    sum_lin = (D[i] * lin[i] + D[j] * lin[j]) / np.sqrt(2.0)
    sum_eig = (D[i] * eig[i] + D[j] * eig[j]) / np.sqrt(2.0)
    spl_lin = (D[i] * lin[i] - D[j] * lin[j]) / np.sqrt(2.0)
    spl_eig = (D[i] * eig[i] - D[j] * eig[j]) / np.sqrt(2.0)
    print(f"\n  in S/N coordinates, this realization:")
    print(f"    sum   component  linear {sum_lin:8.3f} -> eigfloor {sum_eig:8.3f}"
          f"   ratio {sum_eig / sum_lin:.4f}  (predicted phi_+ = "
          f"{modes['phi_p']:.4f})")
    print(f"    split component  linear {spl_lin:8.3f} -> eigfloor {spl_eig:8.3f}"
          f"   ratio {spl_eig / spl_lin:.4f}  (predicted phi_- = "
          f"{modes['phi_m']:.4f})")

    # ------------------------------------------------------ stability run
    n_real = 120
    two = ("linear", "eigfloor")
    split = {s: np.empty(n_real) for s in two}
    ssum = {s: np.empty(n_real) for s in two}
    other = {s: np.empty(n_real) for s in two}
    rep = {s: np.empty((n_real, 2)) for s in two}     # reported sigma: B, star
    for r in range(n_real):
        d = make_data(T, truth_flux, seed=1000 + r)
        for solver, kw in (("linear", {}), ("eigfloor", dict(eig_floor=EIG_FLOOR))):
            f, sig, _ = fit(d, solver, **kw)
            split[solver][r] = f[j]
            ssum[solver][r] = f[i] + f[j]
            other[solver][r] = f[0]
            rep[solver][r] = (sig[j], sig[0])

    print(f"\nstability over {n_real} independent noise realizations")
    print(f"{'quantity':>22} {'truth':>7} " +
          " ".join(f"{s:>22}" for s in two))
    for name, series, tval in (
            ("blend B (star)", split, truth_flux[j]),
            ("blend A+B (sum)", ssum, truth_flux[i] + truth_flux[j]),
            ("bright star (isolated)", other, truth_flux[0])):
        print(f"{name:>22} {tval:7.2f} " +
              " ".join(f"{series[s].mean():8.3f} +- {series[s].std(ddof=1):8.3f}"
                       for s in two))
    print(f"{'variance ratio':>22} {'':7} "
          f"linear/eigfloor split "
          f"{split['linear'].std(ddof=1) / split['eigfloor'].std(ddof=1):.2f}x, "
          f"sum "
          f"{ssum['linear'].std(ddof=1) / ssum['eigfloor'].std(ddof=1):.3f}x")

    print("\nare the error bars honest? (reported sigma vs the actual scatter)")
    for name, series, col, tval in (
            ("blend B (star)", split, 0, truth_flux[j]),
            ("bright star", other, 1, truth_flux[0])):
        for s in two:
            emp = series[s].std(ddof=1)
            r_ = rep[s][:, col].mean()
            print(f"  {name:>15} {s:>9}: reported {r_:.3f}  actual {emp:.3f}  "
                  f"reported/actual {r_ / emp:.2f}   "
                  f"bias (mean - truth)/reported "
                  f"{(series[s].mean() - tval) / r_:+.2f} sigma")

    # ------------------------------------------------------------ figures
    plot_scene(truth_fine, data, results["eigfloor"][2],
               OUT / "example_scene.png")
    plot_solvers(truth_flux, results, modes, OUT / "example_solvers.png")
    plot_stability(truth_flux, split, ssum, modes, OUT / "example_stability.png")
    for name in ("example_scene.png", "example_solvers.png",
                 "example_stability.png"):
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
