# How it works

What happens between `optimize_fluxes(tractor)` and a flux vector, and which
knob controls each stage. Read this when you need to tune throughput, understand
a surprising number, or extend the engine.

## The two layers

Tractor-JAX has an **object layer** that mirrors The Tractor's API — `Image`,
`Catalog`, `PointSource`, `SersicGalaxy`, `PixelizedPSF`, `WCS` — and a
**functional JAX layer** (`tractor_jax.jax`) that does the numerical work on
flat arrays. The object layer is Python and runs once per problem; the JAX layer
is `jit`-compiled and runs on the accelerator.

```text
Tractor(images, catalog)                     object layer (Python, per problem)
        │
        ▼
extract_model_data                           flatten to arrays: per-image data,
        │                                    invvar, PSF FFTs; per-source
        │                                    positions, shapes, MoG profiles
        ▼
[ bucketing / tiling ]                       group images by padded shape;
        │                                    optionally split into core+halo tiles
        ▼
vmap( solve_fluxes_* )                       one compiled kernel over the batch
        │                                    (render templates -> AtWA -> solve)
        ▼
fluxes, variances                            written back with update_catalog=True
```

Everything below the flattening step is shape-polymorphic only through
recompilation, which is why so much of the design is about **making shapes
repeat**.

## Rendering: oversample, then bin

A source's template is not evaluated at pixel centres. For an undersampled
instrument that biases the flux — the PSF varies substantially across one pixel.
Instead the engine renders on a finer grid and integrates each native pixel:

1. **Point sources with a pixelized PSF** are drawn by shifting the oversampled
   PSF stamp to the source's sub-pixel phase and summing each $k \times k$ block
   back to native resolution (`render_point_source_pixelized`,
   `rebin_downsample_int_flux` for integer factors, `downsample_image` otherwise).
2. **Galaxies** are convolved in Fourier space: the Sérsic profile's
   mixture-of-Gaussians has an analytic FT (`gaussian_fourier_transform`), which
   is multiplied by the PSF's FFT and inverted (`render_galaxy_fft`). The PSF FFT
   is computed once per geometry and **cached on the PSF object**
   (`psf_to_fft`, `psf._jax_fft_cache`), so repeated views of the same PSF pay
   for it once.
3. **Gaussian-mixture PSFs** skip the FFT entirely — the convolution of two
   Gaussians is a Gaussian, so the model is evaluated in closed form
   (`render_galaxy_mog`, `render_point_source_mog`). This is much faster; use a
   MoG PSF when it describes your instrument well.

`oversample_rendering=True` turns on the oversampled path for a `PixelizedPSF`
whose `sampling != 1`. Flux conservation through this chain is a tested contract:
a unit-flux source must render to a unit-sum model, on both the point-source and
the galaxy path, at integer and non-integer factors.

:::{warning}
An oversampled `PixelizedPSF` carries a `1/sampling²` pixel-area factor. Both
render paths apply it; if you write a new render path, assert that a unit-flux
source sums to 1 — this factor has been the source of two separate 25× flux bugs.
:::

## Batching: making shapes repeat

The solver is `jit(vmap(solve, in_axes=...))` over a stack of images. `vmap`
demands identical shapes, and real images are not identical, so:

- **Bucketing** (`bucket_mode="auto"`, `bucket_base=32`) groups images by padded
  size and runs one `vmap` per bucket. Fewer distinct padded shapes means fewer
  XLA compilations.
- **`build_padded_batches`** does the same for *views* that share one parent
  catalog — halo tiles of a big image, or per-target postage stamps. It pads
  every view's source arrays to a common width and returns a
  {class}`~tractor_jax.jax.batching.BatchBundle` ready for `vmap`.
- **Caps vs. buckets.** `max_ps_cap` / `max_gal_cap` / `max_mog_k_cap` fix the
  padded widths outright: one compilation for the whole run, at the cost of
  padding every view to the field maximum. `pad_bucket=N` instead rounds the
  *natural* widths up to a multiple of `N` — a handful of shapes per field, each
  compiled once, with matrices close to their natural size. Padding is
  output-preserving: masked slots carry no source, and with the Jacobi ridge the
  solution is invariant to them.
- **`make_batched_solver`** memoizes the compiled `jit(vmap(...))` by
  `(solver, in_axes, kwargs)`. Crucially, the lasso penalty weights and the
  eigfloor_prior arrays are **runtime arguments**, not closure constants, so
  changing them per view does not trigger a recompilation.
- **`autotune_batch_size`** finds the throughput knee empirically when you do not
  want to guess.

## Tiling: bounding the dense solve

The dense solve is $O(p^3)$ in the number of free fluxes, and a wide image in a
crowded field has too many. `use_tiling=True` splits each image into `tile_size`
**core** tiles padded by a PSF-sized **halo**, solves every tile independently
with a compact per-tile flux vector, and merges the results back into one
catalog-layout vector per image.

The halo is what makes this valid: a source just outside a core still contributes
its PSF wings to that tile's pixels. Each source is then read from the tile whose
**core box** contains it, so halo overlaps never double-count. The background, if
fit, is reported as the core-area weighted mean of the per-tile backgrounds.

Tiled and untiled results agree — that is a tested contract, not an
approximation. The knob that matters is the halo width: too small and the tile
edges lose flux, too large and every tile solves a bigger problem than it needs.

## Multi-device

`use_sharding=True` (the default) distributes the image batch across every
visible device with GSPMD: `prepare_sharded_inputs` builds a
`Mesh(devices, axis_names=('img_batch',))` and shards image data, batches, and
fluxes along the batch axis, replicating the shared leaves. There is no `pmap` and
no multi-process setup — it is single-process data parallelism, and the batch
axis must be divisible by the device count.

Because the images in a batch are independent solves, this scales almost
perfectly; see {doc}`performance`.

## Overlapping the host

On a real pipeline the GPU is idle while Python reads the next image and builds
its arrays. {func}`~tractor_jax.jax.pipeline.prefetch_pipeline` runs that build in
a worker thread with bounded look-ahead, so cutout $N+1$ is prepared while
cutout $N$ solves:

```python
from tractor_jax.jax import prefetch_pipeline

for item, built in prefetch_pipeline(items, build_fn, depth=2):
    fluxes = solve(built)          # GPU; build of the next item already running
```

Build errors surface at the consumer's `yield`, and abandoning the loop early
cancels and joins the workers. In production this is worth ~1.3–1.4× end to end.

## Where the numbers come from

Every stage above has a test that pins its contract — padded batches solve
identically to per-view builds, tiled matches untiled, the solver factory is
bit-identical to a hand-rolled `jit(vmap(...))`, the downsampler conserves flux,
FFT-rendered galaxies carry unit flux. Those tests are the specification; the
measured throughput they enable is in {doc}`performance`.
