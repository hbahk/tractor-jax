# Performance and hardware

Measured numbers from a production SPHEREx run, and what they imply for sizing
your own hardware. All figures are from the pinned "F3" benchmark (2026-07-17)
on **1× NVIDIA L40S** with a 56-core host; the workload is 100×100 pixel cutouts
at full catalog depth (~3700 sources per cutout), tile 15 / halo 3,
`pad_bucket=32`, float32.

## Against the legacy CPU Tractor

The reference is the legacy pipeline as deployed — plain forced photometry, one
pinned core, float64: **10.77 s per cutout**.

| solver | serial ms/cutout | GPU solve | pipelined | 1 GPU ≈ N legacy cores |
|---|---|---|---|---|
| `linear` | 101.9 | 41.5 | **55.1** | **195** |
| `lasso` | 114.2 | 53.6 | **75.8** | **142** |
| `eigfloor` | 190.0 | 130.0 | **154.6** | **70** |
| `eigfloor_prior` | 200.8 | 130.2 | **171.2** | **63** |

The `linear` row is the strict same-problem comparison: identical estimator on
both engines. The other rows compare *our production estimators* against that
deployed baseline, so they are conservative — a same-estimator CPU
implementation would be slower than the plain solve, raising those numbers.

At the node level, on the same 96 cutouts: **1 GPU = 9.95 cutouts/s** versus a
legacy CPU pool at 1.58 (24 processes) or 2.06 (48 processes; the 56-core node
saturates) — **6.3× / 4.9× per node**, before the pipelining gain above. One-time
compilation is ~133 s of per-shape traces, or ~12 s with fixed caps.

## Multi-GPU

Cutout-parallel work scales essentially linearly: **2× L40S = 26.19 vs 13.17
cutouts/s, 99.4% scaling efficiency**. Independent images are independent solves,
so there is nothing to synchronize (see {doc}`architecture`).

Running **two processes on one card** (memory fraction 0.4) is solver-dependent
and not a general win: **+16% for `lasso`** (it fills read/host-sync idle gaps)
but only **+2% for `linear`** — the dense solve already saturates the card — and
**0% for the eigfloor family**, whose host-synchronous `eigh` serializes the GPU.

## What each knob buys

| change | effect |
|---|---|
| `pad_bucket=32` vs fixed caps | removes padded-width inflation of the host-synchronous solves; products bit-identical |
| halo 3 (from a wider halo) | GPU solve 1.6× faster (68.7→41.4 ms linear, 186→130 eigfloor) |
| `prefetch="thread"` vs sync | 1.29–1.40× end to end; recovers 63–66% of the ideal overlap window |
| float64 vs float32 | ~3.5× slower end to end on the L40S (its FP64 rate is 1/64 of FP32) |
| `eigfloor_prior` vs `eigfloor` | GPU solve identical (130.2 vs 130.0 ms) — the prior terms are free; the +11% is CPU-side prior evaluation |
| tile-chunked `vmap` | **rejected** — up to 2× slower and cancels the prefetch gain; solve whole images in one `vmap` |

Note the shape of these results: after the solve itself is fast, the wins come
from *not recompiling* and from *keeping the host off the critical path*.

## Choosing hardware

**GPU memory** is set by the batch: the dense solve is $O(p^2)$ in stored
matrices and $O(p^3)$ in work for $p$ free fluxes per view.
{func}`~tractor_jax.jax.batching.estimate_solve_bytes_per_view` gives a per-view
estimate, and `autotune_batch_size` finds the throughput knee empirically. If you
share a card, cap the pool: JAX otherwise preallocates most of it.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
```

**Precision.** float32 is the production default, and it is validated rather
than merely tolerated: on an independent image simulation the fp32 production
config recovers bright fluxes to +0.1% with calibrated errors (pull σ 1.00 for
`linear`, 0.92 for `eigfloor`). Note that TF32 is on in every fp32 number here —
it is the JAX GPU default and buys ~2.1× on the normal-equations GEMM at
~1.5×10⁻⁴ relative error on $A^\top A$; the accuracy validation ran that same
path. (Cores-equivalent figures therefore assume an Ampere-or-later card.)

What float64 actually buys is **exactness under padding and batching**: pad/batch
invariance is ~10⁻¹¹ in x64 versus percent-level deviations in fp32 on faint,
near-null-space fluxes. Use it when you need bit-reproducible products across
tiling choices, or for faint work in crowded groups where a stronger regularizer
(`eigfloor`) is the other half of the answer. Budget ~2.5× end-to-end (~3.5× on
the serial solve path) on an L40S, whose FP64 rate is 1/64 of FP32; an
A100/H100-class card largely removes that penalty.

## CPU is not a fallback

Running the same code with `JAX_PLATFORMS=cpu` is **not** a slower-but-usable
mode — it is slower than the legacy CPU Tractor:

| configuration | s/cutout | vs legacy |
|---|---|---|
| tractor-jax, 1 CPU core | 34 | 3.2× **slower** than legacy 1 core |
| tractor-jax, 24 CPU cores | 7.8–9.0 | ~12× slower than legacy 24-proc |
| tractor-jax, full 56-core node | 6.3 | ≈ 1.7 legacy cores |

The speed of this engine is a **GPU + batching co-design**, not "JAX being fast".
XLA-CPU threading recovers only 4.4–5.4× across the whole node, and the workload
is render/GEMM-bound rather than solver-bound (the numbers barely move between
solvers). On CPU-only hardware, use the legacy Tractor.

CPU execution is still the right choice for **development and testing** — the
test suite and the {doc}`worked_example` run on CPU in a couple of minutes.

## Reproducing

The benchmark harness lives in the analysis project that drives this engine
(`analysis/run_paper_bench_final.sh`, `analysis/bench_multigpu_point.py`), not in
this repository. Two measurement cautions carried over from that work:

- **Warm up before timing.** A two-point wall-clock fit is invalid with
  `pad_bucket`: later cutouts introduce new padded shapes, and the mid-run
  recompiles inflate the difference by ~2.5×. Time a steady state after warmup.
- **Compilation is not throughput.** Report the one-time trace cost separately
  from the per-item steady state.
