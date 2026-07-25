# Changelog

All notable changes to Tractor-JAX. This project follows
[Semantic Versioning](https://semver.org/); while the major version is 0 the
public API may still change between minor releases.

## [0.1.0] — 2026-07-25

First versioned release. The engine had been developed continuously since the
initial extraction from The Tractor; this release marks the point where the
batched solver stack, its test suite, and the documentation describe the same
package.

### Added

- **Batched solver stack.** `build_padded_batches` pads and stacks many views
  (halo tiles or postage stamps) sharing one parent catalog into a `vmap`-ready
  bundle; `make_batched_solver` memoizes the compiled `jit(vmap(...))` and takes
  penalty/prior arrays as *runtime* arguments so per-view weights no longer
  force a recompilation. `pad_bucket` rounds natural widths to a multiple as a
  middle ground between fixed caps and per-view retracing.
- **`eigfloor` solver** — eigenvalue floor on the Jacobi-normalized Gram. Damps
  only correlation-degenerate directions (the flux splits of blended groups),
  sign-free and selection-free, so faint sources keep their negative
  excursions. The recommended default for blind photometry.
- **`eigfloor_prior` solver** — `eigfloor` plus per-source Gaussian flux priors
  (ridge toward an externally predicted flux). Protected sources
  (`lambda = 0`) reduce exactly to `eigfloor`.
- **`lasso` improvements** — public `lasso_fista` / `lasso_fista_jit`,
  `alpha="auto"` (universal-threshold rule in S/N units), and
  `debias_signfree` ∈ {`none`, `protected`, `all`} to remove the rectification
  bias on reported targets.
- **`prefetch_pipeline` / `lagged_collect`** — overlap the host-side build of
  the next item with the device solve of the current one, with bounded
  look-ahead and error propagation at the consumer.
- **`autotune_batch_size`** and `estimate_solve_bytes_per_view` for sizing the
  `vmap` batch empirically.
- **Tiling** (`use_tiling=True`) — core tiles with a PSF-sized halo, merged by
  core-box ownership into one catalog-layout flux vector per image.
- **Documentation**: a worked example that actually fits
  (`examples/fit_blended_sources.py`), plus guides on choosing a solver, how the
  engine works, measured performance, and migrating from The Tractor.
- Memoized `SersicMixture.getProfile`, and a `PixelizedPSF` FFT cache stapled to
  the PSF object.

### Fixed

- **Unrecognised PSFs silently returned zero fluxes.** `extract_model_data`
  dispatched only on `PixelizedPSF` and `GaussianMixturePSF` with a silent
  fall-through, so an `NCircularGaussianPSF` or a `HybridPixelizedPSF` rendered
  as an all-zero template and every source came back with flux 0 and infinite
  variance — no error, no warning. PSF classification now lives in
  `psf_kind()`, accepts any PSF exposing a mixture-of-Gaussians, routes hybrids
  through their pixelized model, and raises `TypeError` on anything else.
- **Oversampled Fourier renders were `1/sampling²` too faint.** The oversampled
  FFT path omitted the pixel-area factor that the point-source path applies, so
  FFT-convolved galaxies rendered ~25× too faint at `sampling=0.2` while point
  sources were correct.
- **Non-integer oversampled rendering** — registration drift of ~0.05 native px
  and a ±6–10% matched-filter amplitude swing, from an odd high-resolution width
  in the `rfft2` reconstruction.
- **Downsampling window** — non-integer factors dropped the native-pixel
  integration window, breaking flux conservation.
- Galaxy position-angle sign convention; a `vmap`/`cond` GPU bug; numpy fallback
  for the legacy `tractor.mix` C grid evaluators.

### Changed

- Python ≥ 3.11 is required (the documentation previously claimed 3.9).
- Relicensed to GPL-3.0-or-later under The Tractor's "or later" clause.
