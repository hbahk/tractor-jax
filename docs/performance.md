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

## Service rates (2026-08-23, engine as released: compact stamps, host fast paths, jax 0.9.0)

Best rate found over an `M` sweep (1–16 co-scheduled streams), host work
included; shared node, so the host-bound full-depth rows move by up to ±25%
between sessions — best same-day row quoted.

| card | estimator | full depth | `m_z<21` |
|---|---|---:|---:|
| NVIDIA H100 80 GB | `linear` | 181 cutouts/s (5.5 ms, `M=8`) | 488 (2.0 ms, `M=16`) |
| NVIDIA H100 80 GB | `eigfloor` | 84.1 (11.9 ms, `M=6`) | 396 (2.5 ms, `M=16`) |
| NVIDIA H100 80 GB | `eigfloor_prior` | 70.4 (14.2 ms, `M=8`) | 382 (2.6 ms, `M=12`) |
| NVIDIA H100 80 GB | `lasso` | 69.6 (14.4 ms, `M=6`) | 134 (7.5 ms, `M=6`) |
| NVIDIA L40S (quiet) | `linear` | 76.0 (13.2 ms, `M=4`) | 325 (3.1 ms, `M=16`) |
| NVIDIA L40S (quiet) | `eigfloor` | 54.5 (18.4 ms, `M=8`) | 303 (3.3 ms, `M=16`) |

A single stream is **host-bound** at either depth (H100: ~43 cutouts/s full,
~55 at `m_z<21`); six to sixteen streams fill the card. Against the tiled
legacy Tractor on one CPU core of the same node (4.27 s per full-depth
cutout, 0.70 s at `m_z<21`, same day) one H100 keeps pace with roughly 770
cores for the linear-to-linear pair and ~360 for `eigfloor` at full depth —
a throughput equivalence, not a latency speedup, and a full-depth statement:
at the `m_z<21` depth the CPU path falls ~6× while the card falls ~2–5×, so
the card/core ratio is smaller there (~340 linear, ~280 `eigfloor`).

The pre-improvement rates (legacy rendering, jax 0.5.3, `M=3` protocol:
H100 `eigfloor` 29.1 cutouts/s / 34.4 ms, ≈130-core equivalence) are the
numbers earlier notes and drafts quote; the paper's Figure 1(b) uses the
table above.

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
eigensolver on L40S-class cards (~45 of the 49 ms solve per cutout at
`m_z<21`, ~95 of 111 ms at full depth with the stamp; host-synchronous and
per-matrix, so co-scheduled workers queue on it), which neither rendering
option touches; on the H100 that term is ~10 ms. The opt-in
`eig_method="host"` (solver kwarg, 2026-08-22) runs that eigendecomposition
on host LAPACK instead — numpy `ssyevd` on `eig_host_threads` threads through
`jax.pure_callback`, BLAS pinned to one thread — and is an fp32
eigensolver-level equivalent, not bit-identical. On one L40S at `m_z<21` it
takes the solve from 49 to 39 ms per cutout (4 threads), one worker from 14.9
to 20.0 cutouts/s and three co-scheduled workers from 20.7 to 54.0 cutouts/s,
because each stream's eigh now runs on its own host threads; at full depth
(334×334 Grams) it loses (141 vs 111 ms; 5.8 vs 7.8 cutouts/s on a
loaded host) and on an H100 it is not worth its threads. At full depth the
driver-side lever is *per-tile size bucketing* (the SPHEREx driver's
`TILE_SIZE_BUCKETS=3`): tiles are grouped by live-source count and each
group is built and solved as its own bundle padded to its own maxima, so
the 150-source tiles stop paying the 385-wide eigendecomposition of the
330-source ones — 40% off the eigfloor GPU solve per cutout at the float32
level, with three fixed shapes per cutout. A batched GPU Jacobi eigensolver
remains the open engine item for the rest.

## Multi-GPU and co-scheduling

Cutout-parallel work scales linearly across cards (two L40S: 99.9–100.6% at
`M=3`). Co-scheduling several worker processes on one card fills the host
gaps: `linear` +38%, `lasso` +22%, `eigfloor_prior` +22%, `eigfloor` +9% at
`M=3` on the L40S; on the H100 the `M` sweep is 13.3 / 25.2 / 30.9 / 31.2 /
31.3 / 31.3 cutouts/s for `M=1…6` (full grid).

With `render_stamp=80` the GPU solve is no longer what limits the card, and
the operating `M` should follow the host cores available. Measured on one
L40S with the driver's host fast paths (aggregate cutouts/s, card occupancy
from `nvidia-smi`):

| | M=1 | M=2 | M=3 | M=4 | M=6 | M=8 |
|---|---:|---:|---:|---:|---:|---:|
| `linear`, full depth | 20.5 (43%) | 34.6 | 44.2 | 52.3 | 49.4 | 53.7 (99%) |
| `linear`, `m_z<21` | 25.1 (13%) | 48.0 | 67.7 | 90.0 | 124.6 | 146.6 (31%) |
| `eigfloor`, full depth | 7.9 (97%) | 7.7 | 9.7 | 9.5 | 8.7 | 8.2 (100%) |

The linear family saturates the card at full depth around `M=4` (~54
cutouts/s, 2.5× the full-grid card rate) and at `m_z<21` is still scaling
linearly with the number of host streams at `M=8` with the card 31% occupied;
the eigfloor family reports 95–100% occupancy already at `M=1` and does not
move with `M` — that is cuSOLVER's eigensolver, not the schedule. Memory is
not the constraint with the stamp (0.9/M of the card per worker is ample).

Two caveats on reading such sweeps. `utilization.gpu` is the fraction of time
*any* kernel is resident: many time-sliced contexts keep it near 100% while
the SMs are far from busy, so it cannot tell a saturated card from a busy
scheduler. And the per-stream rate is set by the host work per cutout, so the
curve moves whenever that work moves: after the SPHEREx driver rebuilt its
WCS from header values and this engine kept the ramped PSF-basis transforms
in the caller-owned cache (`f443145`), one stream delivered 40.8 cutouts/s at
`m_z<21` (25.1 before) and 31.3 at full depth (20.5); after a third host round
(dict headers, one `device_put` per bundle — `34bef60` — exact binned medians,
cached catalog arrays) 50.0 and 38.0, with `M=3` 146 cutouts/s at `m_z<21` on
three host cores, `M=8` 300, `M=16` 430 (card 94% occupied) and `M=6` 76
cutouts/s at full depth (GPU-bound). Fewer host streams, not bigger launches,
is how the card gets filled: batching several cutouts into one launch changes
the per-cutout GPU time by ≤ 7% even with the stamp.

The same options on one **H100 80 GB** (jax 0.11, 2026-08-22, node CPU load
52–77): GPU solve per cutout full grid → stamp 80 + fft, `linear` 21.0 → 9.3 ms
at full depth and 6.4 → 2.9 ms at `m_z<21`, `eigfloor` 31.7 → 20.1 and 6.7 →
3.1 ms — the eigendecomposition is ~11 ms per cutout at full depth and ~0.2 ms
at `m_z<21` there, so eigfloor costs within 7% of linear at the science depth.
Card rates with the stamp and the host fast paths (cutouts/s, `M`=1/3/6):
`linear` full depth 42.5 / 89.7 / 111.4, `m_z<21` 55.2 / 162.9 / 292.7;
`eigfloor` full depth 41.4 / 46.9 / 51.5 (GPU-bound by its eigh), `m_z<21`
55.4 / 162.6 / 251.9; the paper-protocol `eigfloor` full-depth `M=3` row
reproduces (29.1). Neither `eig_method="host"` nor the driver's size
bucketing pays on the H100 (39.0 / 111.7 and 25.3 / 31.7 against the rows
above): both are remedies for cards whose per-matrix eigensolver is slow.

**That slowness is a jax version, not the card.** On the same L40S,
`jnp.linalg.eigh` on 49 matrices takes 49.0 / 106.8 / 124.3 ms (n = 102 /
334 / 385) under jax 0.5.3 and **0.96 / 13.9 / 16.6 ms** under jax 0.11.1
(cuSOLVER 11.7.5; 51× at n = 102, 7.5× at n = 385; same reconstruction
error): jaxlib 0.5 ran a batched `eigh` one matrix at a time with
`syevd` and a host sync each, the current jaxlib uses cuSOLVER's batched
path. Upgrading jax removes the L40S eigfloor wall outright — eigfloor then
costs about what linear costs at `m_z<21` — and makes `eig_method="host"`
and the driver's bucketing unnecessary (they remain as opt-ins for old
stacks). One gotcha: an `LD_LIBRARY_PATH` that points at an older CUDA
(`/usr/local/cuda/lib64`) makes the 0.11 CUDA plugin fail to load cuSPARSE
and fall back to CPU silently; clear it or fix it in the env's activate.d.
End to end on the L40S (production env cloned with only jax upgraded, same
CUDA libraries, stamp 80): eigfloor GPU solve per cutout 49.9 → 5.5 ms at
`m_z<21` and 123 → 32 ms at full depth; card rates `M`=1/3 12.4/15.6 →
48.7/125.9 cutouts/s and 6.4/7.9 → 23.1/32.6. The one regression is the
stamp render itself, 25–37% slower under the newer XLA (3.3 → 4.2 ms at
`m_z<21`, 12.4 → 17.0 at full depth; no XLA flag recovers it), which costs
the render-bound `linear` family 15–26%. Bisected on an H100 with the same
captured inputs (2026-08-23): render 1.52 / 4.13 ms (`m_z<21` / full) under
jax 0.5.3, 1.43–1.45 / 3.85–3.87 under 0.7–0.9, **1.82 / 5.05 under 0.10.0**
(+27% / +31%, the same effect as 0.11) — the step is at jax 0.10.0, while
the batched eigh is fast from 0.8.0 on. **jax 0.9.0 has both**, and is the
version to prefer until the XLA:GPU change is understood; 0.10+ works but
renders 25–30% slower (eigh unaffected). The cause, from the optimized HLO
and the jax source: up to jax 0.9 a 2-D inverse real FFT is one HLO `fft` op
(`IRFFT, fft_length={80,80}`, a multi-dimensional cuFFT C2R plan); jax
0.10.0 added a GPU lowering rule (`_fft_lowering_gpu` in
`jax/_src/lax/fft.py`) that decomposes every multi-dimensional IRFFT into a
complex IFFT over the outer axes plus a 1-D IRFFT, with two transposes, so
that GPU results match NumPy for inputs that are Hermitian only along the
last axis (cuFFT's C2R assumes symmetry on all axes). Our inputs satisfy
the full symmetry, so the two lowerings agree bit for bit and the
decomposition is pure cost; `jax.lax.fft` goes through the same rule, so
there is no engine-side workaround, only the jax version (or an upstream
opt-out for Hermitian-symmetric inputs).

## Service rates on the improved stack (2026-08-23)

Stamp 80 + fft, host fast paths, jax 0.11; the card's best measured rate
over a sweep in M (cutouts/s at that M; per-cutout service time in
parentheses; the "knee" pick — smallest M within 5% of the maximum — differs
by less than 5% by construction). CPU = the tiled legacy Tractor on one core
of the same node, K=1.

| card, depth | `linear` | `eigfloor` | `eigfloor_prior` | `lasso` | CPU one core |
|---|---:|---:|---:|---:|---:|
| H100, full depth (jax 0.9.0, best same-day row) | 181 @M=8 (5.5 ms) | 84.1 @M=6 (11.9 ms) | 70.4 @M=8 (14.2) | 69.6 @M=6 (14.4) | 4.27 s |
| H100, full depth (jax 0.9.0, single sweep) | 111 @M=8 (9.0 ms) | 81.2 @M=6 (12.3 ms) | 67.7 @M=6 (14.8) | 43.1 @M=4 (23.2) | 4.27 s |
| H100, `m_z<21` (jax 0.9.0) | 488 @M=16 (2.0 ms) | 396 @M=16 (2.5 ms) | 382 @M=12 (2.6) | 134 @M=6 (7.5) | 0.70 s |
| H100, full depth (jax 0.11.0, 08-22) | 121 @M=8 (8.3 ms) | 71.5 @M=8 (14.0 ms) | 58.0 @M=8 (17.2) | 54.9 @M=8 (18.2) | 4.25 s |
| H100, `m_z<21` (jax 0.11.0, 08-22) | 387 @M=12 (2.6 ms) | 314 @M=14 (3.2 ms) | 310 @M=12 (3.2) | 113 @M=6 (8.8) | 0.70 s |
| L40S, full depth (jax 0.9.0, quiet card) | 76.0 @M=4 (13.2 ms) | 54.5 @M=8 (18.4 ms) | 38.5 @M=6 (26.0) | 41.4 @M=6 (24.2) | 4.83 s |
| L40S, `m_z<21` (jax 0.9.0, quiet card) | 325 @M=16 (3.1 ms) | 303 @M=16 (3.3 ms) | 195 @M=8 (5.1) | 114 @M=8 (8.8) | 0.78 s |
| L40S, full depth (jax 0.11.1, quiet card) | 68.8 @M=8 (14.5 ms) | 43.7 @M=8 (22.9 ms) | 33.3 @M=6 (30.0) | 33.1 @M=6 (30.2) | 4.83 s |
| L40S, `m_z<21` (jax 0.11.1, quiet card) | 319 @M=16 (3.1 ms) | 331 @M=16 (3.0 ms) | 158 @M=8 (6.3) | 119 @M=8 (8.4) | 0.78 s |

One-core equivalents (card rate over the same node's one-core rate): H100
jax 0.9.0, best same-day row (the paper's convention): full depth `linear`
774 (the paper's "about 770"), `eigfloor` 359, `m_z<21` 344 / 279; the single
sweep gave 473 / 347, the 0.11.0 set 513 / 304 and 271 / 220 — the host-bound
full-depth rows move by up to ±25% between sessions with the shared node's
load (linear M=8: 111 / 121 / 162 / 171 / 181 cutouts/s across five same-day
sessions) while a single stream stays at 40–43; L40S
(0.11.1, 08-23 quiet card) 332 / 211 and 250 / 260. At `m_z<21` the H100
reaches 400–490 cutouts/s at M = 12–16 (16 workers on the partition's
16-CPU-per-job cap; rows noisy above M=10) and the L40S is still
host-stream-bound at M=16 (efficiency 0.4), so the L40S row is a lower bound
on that card. Differences between the two H100 sets at full depth (linear
8.3 vs 9.0 ms, lasso 18.2 vs 23.2) are session noise of host-/GPU-bound rows
on a shared node, not the jax version; the eigfloor family's 12–22% gain is. The H100 rows are the paper's
(`manuscript/macros.tex`, provenance there); the headline table at the top of
this file quotes the jax 0.9.0 rows of this table.

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
