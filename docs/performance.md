# Performance and hardware

Measured numbers from the SPHEREx production configuration (tiled driver, tile
15 / halo 3, `pad_bucket=32`, float32, full-catalog depth: ~3,700 fitted
sources per 100×100-pixel cutout, 49 tiles of 21×21 pixels), and what they
imply for sizing hardware. Two protocols appear below; do not mix them:

* **service** — wall time per completed cutout per card, host read +
  construction + record extraction included, measured as one single-worker
  pass plus `M` co-scheduled worker processes on one card;
* **GPU solve** — the jitted vmapped solve alone (template rendering +
  normal equations + factorization) on prebuilt inputs, warm, per cutout.

## Service rates (2026-08, production configuration)

| card | estimator | single worker | `M=3` on one card | 
|---|---|---:|---:|
| NVIDIA H100 80 GB | `linear` | 13.6 cutouts/s (74 ms) | 31.5 cutouts/s (31.7 ms) |
| NVIDIA H100 80 GB | `eigfloor` | 14.0 (72 ms) | 29.1 (34.4 ms) |
| NVIDIA H100 80 GB | `eigfloor_prior` | 11.8 (85 ms) | 22.7 (44.1 ms) |
| NVIDIA H100 80 GB | `lasso` | 13.3 (75 ms) | 23.1 (43.3 ms) |
| NVIDIA L40S | `linear` | 12.3 (81 ms) | 19.9 (50.3 ms) |
| NVIDIA L40S | `eigfloor` | 6.2 (162 ms) | 6.8 (146 ms) |

A single worker is **host-bound** on either card (the host stage is ~55 ms of
the 74 ms: FITS read, background, catalog projection, batch construction,
record extraction); `M=3` is the operated point, not the card's limit (the
H100 sweep reaches 31.3 cutouts/s at `M=5`). Against the tiled legacy Tractor
on one CPU core (4.2–4.8 s per cutout at the same configuration) one H100
keeps pace with about 130 cores at full depth — a throughput equivalence,
not a latency speedup, and a full-depth statement: at the `m_z<21` science
depth the CPU path falls 6.1× while the card falls only 2.0–2.4×, so the
card/core ratio is about three times smaller there.

## Where the GPU time goes, and the rendering options

On the L40S the `linear` GPU solve of a full-depth cutout is 44.5 ms, of which
**39 ms (88%) is template rendering**: every source drawn on the full padded
160×160 high-res tile grid by an inverse FFT, both PSF branches evaluated and
selected with `jnp.where`, the stamps materialized (~3 GB transient per
cutout). Batching tiles of several cutouts into one launch changes nothing
(≤ 1%); the kernels are already bandwidth-bound.

Two opt-in options (2026-08-22) attack that directly; the defaults are the
historical path, bit for bit.

| option | where | effect (L40S, per cutout, `linear`) |
|---|---|---|
| `psf_type="fft"` (solver kwarg) | trace only the pixelized-PSF branch instead of both | full depth 44.5 → 33.0 ms; `m_z<21` 12.1 → 9.1 ms |
| `render_stamp=80` (`build_padded_batches`) + `render_mode="auto"` | point sources and compact galaxies on an 80×80 high-res stamp, large galaxies on the full grid | render 39 → 12 ms, solve **44.5 → 16.4 ms** at full depth; **12.1 → 4.4 ms** at `m_z<21` |

`S=80` is the recommended stamp (`S=60` loses on large-galaxy fallbacks,
`S=100` on per-source work). The stamp path is numerically equivalent to the
full grid, not bit-identical: templates agree to ~2e-6 of the peak inside the
weighted image on real cutouts, and the resulting fluxes move by 40–140× less
than float32 arithmetic itself moves them (median), with the same tails; it
differs by construction only in the zero-weight padding, where the full grid's
periodic wrap parks light from sources near the low tile edge. See
`tests/test_render_stamp.py` and the CHANGELOG for the contract.

The `eigfloor` family is additionally bound by cuSOLVER's symmetric
eigensolver on L40S-class cards (~95 ms per cutout at full depth, ~45 ms at
`m_z<21`, host-synchronous), which neither option touches; on the H100 that
term is ~10 ms. A batched Jacobi eigensolver is the open engine item there.

## Multi-GPU and co-scheduling

Cutout-parallel work scales linearly across cards (two L40S: 99.9–100.6% at
`M=3`). Co-scheduling several worker processes on one card fills the host
gaps: `linear` +38%, `lasso` +22%, `eigfloor_prior` +22%, `eigfloor` +9% at
`M=3` on the L40S; on the H100 the `M` sweep is 13.3 / 25.2 / 30.9 / 31.2 /
31.3 / 31.3 cutouts/s for `M=1…6`.

## Choosing hardware

**GPU memory** is set by the batch: the dense solve is $O(p^2)$ in stored
matrices and $O(p^3)$ in work for $p$ free fluxes per view, and the full-grid
renderer materializes one high-res stamp per source (`render_stamp` removes
most of that). `estimate_solve_bytes_per_view` gives a per-view estimate and
`autotune_batch_size` finds the throughput knee empirically. If you share a
card, cap the pool: JAX otherwise preallocates most of it.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
```

**Precision.** float32 (with TF32 GEMMs, the JAX GPU default) is the
production default for the regularized estimators and is validated against
simulation truth (pull σ 1.003 / 1.001 for the plain solve in the whole-cutout
and tiled geometries). It is **not** licensed for the unregularized `linear`
solve at full catalog depth, where fp32 departs from fp64 by more than the
target flux (5.1 mJy on a 3.5 mJy source) while the eigenvalue-floored solve
stays within 8e-3 mJy. float64 costs ~1.7× on the H100 and ~3.5× on the L40S
(its FP64 rate is 1/64 of FP32).

## CPU is not a fallback

Running the same code with `JAX_PLATFORMS=cpu` is slower than the legacy CPU
Tractor (34 s per cutout on one core against 4.9 s): the dense, padded,
full-grid formulation does roughly 7× more arithmetic than the sparse
compact-stamp legacy path, and only the GPU pays for that cheaply. Use CPU
execution for development and the test suite (a couple of minutes), the
legacy Tractor for CPU-only production.

## Reproducing

The benchmark harness lives in the analysis project that drives this engine
(`analysis/bench_multigpu_point.py`, `analysis/bench_batch_across_cutouts.py`,
`analysis/run_depth_service_bench.sh`, `analysis/run_render_host_e2e_bench.sh`
in `proj-spherex-gpupipe`), not in this repository. Two measurement cautions:

- **Warm up before timing.** With `pad_bucket`, later cutouts introduce new
  padded shapes and the mid-run recompiles inflate a two-point wall-clock fit
  by ~2.5×. Time a steady state after a full warm-up pass.
- **Compilation is not throughput.** Report the one-time trace cost (~25–40 s
  per shape family) separately from the per-item steady state.
