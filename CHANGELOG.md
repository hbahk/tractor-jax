# Changelog

All notable changes to Tractor-JAX. This project follows
[Semantic Versioning](https://semver.org/); while the major version is 0 the
public API may still change between minor releases.

## [Unreleased]

### Added

- **Compact-stamp template rendering (opt-in).** `build_padded_batches(...,
  render_stamp=S)` additionally emits the PSF transform on an `S x S` high-res
  stamp (`images_data["psf"]["fft_stamp"]`) and classifies every galaxy as
  compact or large from its enclosed-flux radius (`stamp_flux_frac`, default
  0.9999) plus the kernel's; the Galaxy batch then carries `stamp_mask` and a
  padded `large_idx` / `large_mask` sub-batch (`stamp_large_bucket`).
  `_render_source_templates` (and therefore every `solve_fluxes_*`) draws
  point sources and compact galaxies on the stamp — one `S^2` transform per
  source instead of the padded tile grid's, with no full-grid image
  materialized — and only the large galaxies on the full grid. New solver
  kwargs `render_mode` (`"auto"` default: stamp iff the bundle carries it;
  `"full"`; `"stamp"`) and `psf_type` (`None` keeps the historical
  both-branches `jnp.where` dispatch; `"fft"` / `"mog"` trace one branch).
  Inside the weighted image the stamp path agrees with the full grid to
  ~1e-6 of the template peak; it differs by construction only in the
  zero-weight padding, where the full grid's periodic wrap parks light from
  sources near the low edge. Bundles built without `render_stamp` and
  solvers with the default kwargs are unchanged (`tests/test_render_stamp.py`
  pins both the agreement and the bit-identity of the default path).
  `batches_in_axes` includes the optional Galaxy keys when present.
- **Host eigensolver for the eigfloor family (opt-in).** `solve_fluxes_eigfloor`
  and `solve_fluxes_eigfloor_prior` take `eig_method` (`"cusolver"`, the
  default, is the historical `jnp.linalg.eigh`; `"host"`) and
  `eig_host_threads`. `"host"` ships the batch of Jacobi-normalized Grams to
  the host through `jax.pure_callback` (`vmap_method="expand_dims"`: one
  callback per vmapped solve) and diagonalizes them with numpy's LAPACK
  (`ssyevd`, releases the GIL) on a thread pool, one matrix per task; pin
  BLAS to one thread (`OPENBLAS/OMP/MKL_NUM_THREADS=1`) so the pool, not
  the library, provides the parallelism. Motivation: on L40S-class cards
  cuSOLVER's per-matrix `syevd` is host-synchronous and dominates the
  eigfloor solve (~45 of 49 ms per 49-tile cutout at `m_z<21`, ~95 of 111 ms
  at full depth with the stamp renderer), identically on every co-scheduled
  worker. Measured on one L40S: solve per cutout 49 → 39 ms at `m_z<21`
  (4 threads); card rate 14.9 → 20.0 cutouts/s with one worker and
  20.7 → 54.0 with three; at full depth (334×334 Grams) it loses
  (111 → 141 ms, 7.8 → 5.8 cutouts/s on a loaded host) and on an H100
  (cuSOLVER ~10 ms per cutout) it is not worth its threads — keep the
  default there. The host path is an fp32 eigensolver-level equivalent
  (fluxes agree to 1e-3, floored inverse ~2e-4 relative), not bit-identical;
  the default path is unchanged (`tests/test_eig_method.py` pins array
  equality). Evaluated and rejected on the way: the pure-JAX QDWH eigensolver
  (`jax._src.lax.eigh`), 10–90× slower at these sizes and inaccurate at fp32
  termination sizes; scipy's `eigh(driver="evd")`, faster serially but
  GIL-bound in the pool. Found afterwards: the slow per-matrix eigensolver
  is jaxlib 0.5's, not the card's — jax 0.11.1 on the same L40S runs the
  batched `eigh` 51× faster at n = 102 (0.96 ms per 49 matrices) and 7.5×
  at n = 385, so on a current jax the default path is the right one
  everywhere and `"host"` is only a remedy for old stacks
  (`docs/performance.md`).

### Changed

- **Requires jax >= 0.8.0.** From that release `jnp.linalg.eigh` on NVIDIA
  GPUs uses cuSOLVER's batched `syev` (jax issue #31368 / PR #31375); on
  older jax a batched eigh ran one matrix at a time with a host sync each,
  and the eigfloor family was 4–50× slower on L40S-class cards (49 ms
  instead of 0.96 ms per 49 × 102² matrices; `docs/performance.md`). The
  engine still runs on older jax — the pin is a performance contract, not
  an API break — so relax it locally if you must, and expect eigfloor to be
  eigh-bound there. Prefer **jax 0.9.0**: from 0.10.0 on XLA:GPU renders the
  compact-stamp templates 25–30% slower (bisected on the same inputs across
  0.5.3–0.10.0; `docs/performance.md`), with the eigh unaffected; no upper
  bound is pinned so a future fix is not blocked. Cause: jax 0.10.0's GPU
  lowering rule for `fft` (`_fft_lowering_gpu`) decomposes every
  multi-dimensional IRFFT into an outer-axes complex IFFT plus a 1-D IRFFT
  with two transposes, so that GPU results match NumPy for inputs that are
  Hermitian only along the last axis; our spectra are fully symmetric, so
  it is pure cost here (`jax.lax.fft` included) and the renderer cannot
  work around it.
- **Relicensed to GPL-2.0-only.** The Tractor is licensed under the GPLv2
  *without* an "or later" clause (its `COPYING` grants "version 2" only), so
  the 0.1.0 relicense to GPL-3.0-or-later was not permitted for this
  derivative work. `LICENSE` now carries the GPLv2 text and `COPYING` the
  grant notice, matching upstream.

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
