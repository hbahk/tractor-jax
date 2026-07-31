"""Batched-solve support: vmap ``in_axes`` construction and a cached
jit(vmap(solver)) factory.

The ``solve_fluxes_*`` functions in :mod:`tractor_jax.jax.optimizer` are
single-image solvers designed to be vmapped over a leading batch axis. Every
downstream driver ends up writing the same wrapper::

    jax.jit(jax.vmap(partial(solve_fluxes_linear, ...), in_axes=(0, 0, bia)))

and, when it forgets to reuse the wrapper across calls, pays a full XLA
recompile per call. This module centralizes that pattern:

- :func:`batches_in_axes` derives the ``in_axes`` pytree from a ``batches``
  dict (``Background.flux_idx`` is a shared scalar slot and maps to ``None``).
- :func:`make_batched_solver` returns the jitted vmapped solver, memoized on
  ``(solver, static kwargs, in_axes structure)`` so repeated requests reuse
  one compiled function; within one returned function XLA still caches per
  input shape, so fixed-shape batches (see the ``max_*_cap`` padding in the
  batch builders) compile exactly once per run.
- For ``solver="lasso"`` the per-image ``penalty_weights`` are a *runtime*
  vmapped argument of the returned callable, not a closure constant — weights
  that differ per call (e.g. per-cutout protected-target sets) reuse the same
  compiled executable instead of forcing a retrace.
- :func:`penalty_weights_from_slots` builds those weights from the
  per-image ``{catalog index -> flux slot}`` maps the batch builders return.

The PSF side of the batch builder also lives here: :func:`psf_to_fft` produces
the ``images_data['psf']['fft']`` layout, and :func:`psf_fft_phase_ramp` /
:func:`shift_psf_fft` re-register a kernel by a SUB-PIXEL amount as a linear
phase in that same Fourier domain — one elementwise multiply on an array the
renderer already holds, no resampling and no extra render cost. Opt in per view
with ``psf_shift`` / ``psf_basis_shifts`` (see
:func:`build_padded_batches`); absent, the transform is bit-identical to the
pre-ramp behavior.
"""
import math
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from tractor_jax.jax.optimizer import (
    _even_hr_width_pad,
    solve_fluxes_linear,
    solve_fluxes_eigfloor,
    solve_fluxes_eigfloor_prior,
    solve_fluxes_lasso,
)
from tractor_jax.sersic import SersicMixture

__all__ = [
    "BatchBundle",
    "build_padded_batches",
    "psf_to_fft",
    "psf_fft_phase_ramp",
    "shift_psf_fft",
    "slice_fluxes",
    "batches_in_axes",
    "make_batched_solver",
    "clear_solver_cache",
    "penalty_weights_from_slots",
    "prior_arrays_from_slots",
    "pad_normal_eq",
    "autotune_batch_size",
    "estimate_solve_bytes_per_view",
]

_SOLVER_FNS = {
    "linear": solve_fluxes_linear,
    "eigfloor": solve_fluxes_eigfloor,
    "eigfloor_prior": solve_fluxes_eigfloor_prior,
    "lasso": solve_fluxes_lasso,
}

# in_axes template per source type; Background.flux_idx is a single shared
# slot index (identical across images), so it is not vmapped.
_IN_AXES_TEMPLATES = {
    "PointSource": {"flux_idx": 0, "pos_pix": 0, "mask": 0},
    "Galaxy": {"flux_idx": 0, "pos_pix": 0, "wcs_cd_inv": 0,
               "shapes": 0, "mask": 0,
               "profile": {"amp": 0, "mean": 0, "var": 0}},
    "Background": {"flux_idx": None},
}


def batches_in_axes(batches):
    """vmap ``in_axes`` pytree matching a ``batches`` dict.

    Only the source types present in ``batches`` get an entry, so the result
    is a valid ``in_axes`` for ``jax.vmap`` over exactly that structure.
    """
    return {k: _IN_AXES_TEMPLATES[k] for k in batches}


def penalty_weights_from_slots(src_slot_per_image, n_images, n_flux,
                               protected):
    """(n_images, n_flux) lasso penalty multipliers: 0 for protected catalog
    sources, 1 elsewhere.

    ``src_slot_per_image`` maps, per image, catalog index -> slot in that
    image's flux vector (as returned by the batch builders). Empty/masked
    slots and the background keep weight 1 here; the solver unpenalizes the
    background itself, and dead slots render a zero template so their weight
    is irrelevant. A protected source absent from an image is simply not set
    there.
    """
    pw = np.ones((n_images, n_flux), dtype=np.float64)
    protected = set(protected)
    for i, slots in enumerate(src_slot_per_image):
        for ci, slot in slots.items():
            if ci in protected:
                pw[i, slot] = 0.0
    return jnp.asarray(pw)


def prior_arrays_from_slots(src_slot_per_image, n_images, n_flux,
                            f_prior_per_source, sigma_prior_per_source,
                            protected=None):
    """Build the runtime prior arrays for the eigfloor_prior solver.

    Returns ``(lambda_diag, f_prior)``, each of shape (n_images, n_flux).

    Scatters per-CATALOG-source Gaussian flux priors into the padded
    per-image flux slots, mirroring :func:`penalty_weights_from_slots`:
    ``src_slot_per_image`` maps, per image, catalog index -> slot in that
    image's flux vector (as returned by the batch builders).

    Slot semantics (0/inf-safe by construction):

    - padding slots, the background slot, and sources absent from an image
      never appear in the slot maps -> ``lambda_diag = 0`` there (no prior;
      dead padding slots are additionally pinned by the solver itself);
    - ``protected`` catalog sources get ``lambda_diag = 0`` (exact eigfloor
      behavior, unbiased);
    - a source whose ``f_prior`` or ``sigma_prior`` is non-finite, or whose
      ``sigma_prior <= 0``, gets ``lambda_diag = 0`` — never an inf/NaN
      precision (map "pin this source" to a small positive sigma instead).

    Parameters
    ----------
    src_slot_per_image : sequence of dict
        Per image, ``{catalog index: flux slot}``.
    n_images, n_flux : int
        Output array shape.
    f_prior_per_source, sigma_prior_per_source : array
        Predicted flux and prior width per CATALOG row (same flux units as
        the fit; typical ``sigma_prior ~ 0.5-1 x f_prior``).
    protected : iterable of int or bool array, optional
        Catalog indices (or a per-catalog-row boolean mask) to leave
        unregularized.

    Returns
    -------
    (lambda_diag, f_prior) : (jax.numpy.ndarray, jax.numpy.ndarray)
        Both (n_images, n_flux); ``lambda_diag = 1/sigma_prior^2``.
    """
    lam = np.zeros((n_images, n_flux), dtype=np.float64)
    fpr = np.zeros((n_images, n_flux), dtype=np.float64)
    fp_arr = np.asarray(f_prior_per_source, dtype=np.float64)
    sp_arr = np.asarray(sigma_prior_per_source, dtype=np.float64)
    if protected is None:
        prot = set()
    else:
        p = np.asarray(protected)
        if p.dtype == bool:
            prot = set(np.flatnonzero(p).tolist())
        else:
            prot = set(int(x) for x in np.ravel(p))
    for i, slots in enumerate(src_slot_per_image):
        for ci, slot in slots.items():
            if ci in prot:
                continue
            f0 = fp_arr[ci]
            s0 = sp_arr[ci]
            if not (np.isfinite(f0) and np.isfinite(s0) and s0 > 0):
                continue
            lam[i, slot] = 1.0 / (s0 * s0)
            fpr[i, slot] = f0
    return jnp.asarray(lam), jnp.asarray(fpr)


def _freeze(obj):
    """Recursively convert dicts/lists to sorted tuples for use as cache keys."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return ("__seq__",) + tuple(_freeze(v) for v in obj)
    return obj


_solver_cache = {}


def clear_solver_cache():
    """Drop all memoized solver callables (frees their XLA executables)."""
    _solver_cache.clear()


def make_batched_solver(solver="linear", *, in_axes, return_variances=True,
                        cache=True, **solver_kwargs):
    """Return a jitted vmapped flux solver.

    Parameters
    ----------
    solver : {"linear", "eigfloor", "eigfloor_prior", "lasso"}
        Which ``solve_fluxes_*`` backend to wrap.
    in_axes : dict
        vmap ``in_axes`` for the ``batches`` argument, matching its structure
        exactly (see :func:`batches_in_axes`).
    return_variances : bool
        Passed through to the solver.
    cache : bool
        Memoize the returned callable on ``(solver, return_variances,
        solver_kwargs, in_axes)``. All kwargs must then be hashable; pass
        ``cache=False`` for unhashable kwargs (e.g. array-valued ``grid``).
    **solver_kwargs
        Static solver options (``rcond``, ``floor``, ``alpha``,
        ``penalty_mode``, ``nonneg``, ``debias``, ``debias_signfree``,
        ``n_iter``, ...). They are baked into the trace; vary them via a new
        factory call, not per solve.

    Returns
    -------
    callable
        ``fn(initial_fluxes, images_data, batches, penalty_weights=None)``
        -> whatever the solver returns (``(fluxes, variances)`` stacks when
        ``return_variances=True``). ``penalty_weights`` is only accepted for
        ``solver="lasso"``, where it is a per-image runtime argument of shape
        ``(n_images, n_flux)``; ``None`` means unpenalized weights of 1
        (identical to the solver's own default).

        For ``solver="eigfloor_prior"`` the callable instead accepts
        ``fn(initial_fluxes, images_data, batches, lambda_diag=None,
        f_prior=None)`` where ``lambda_diag`` and ``f_prior`` are per-image
        RUNTIME arguments of shape ``(n_images, n_flux)`` (build them with
        :func:`prior_arrays_from_slots`); ``None`` means all-zero, i.e.
        exactly the ``"eigfloor"`` solve. Like lasso's ``penalty_weights``,
        varying their values reuses one compiled executable.
    """
    if solver not in _SOLVER_FNS:
        raise ValueError(f"unknown solver {solver!r}; "
                         f"expected one of {sorted(_SOLVER_FNS)}")
    key = None
    if cache:
        key = (solver, bool(return_variances), _freeze(solver_kwargs),
               _freeze(in_axes))
        try:
            hit = _solver_cache.get(key)
        except TypeError as exc:
            raise TypeError(
                "make_batched_solver kwargs must be hashable when cache=True; "
                "pass cache=False for array-valued options") from exc
        if hit is not None:
            return hit

    base = _SOLVER_FNS[solver]
    if solver == "lasso":
        def _solve(init, imgd, bat, pw):
            return base(init, imgd, bat, penalty_weights=pw,
                        return_variances=return_variances, **solver_kwargs)
        jfn = jax.jit(jax.vmap(_solve, in_axes=(0, 0, in_axes, 0)))

        def fn(initial_fluxes, images_data, batches, penalty_weights=None):
            if penalty_weights is None:
                penalty_weights = jnp.ones_like(initial_fluxes)
            return jfn(initial_fluxes, images_data, batches, penalty_weights)
    elif solver == "eigfloor_prior":
        def _solve(init, imgd, bat, lam, fp):
            return base(init, imgd, bat, lambda_diag=lam, f_prior=fp,
                        return_variances=return_variances, **solver_kwargs)
        jfn = jax.jit(jax.vmap(_solve, in_axes=(0, 0, in_axes, 0, 0)))

        def fn(initial_fluxes, images_data, batches, lambda_diag=None,
               f_prior=None):
            if lambda_diag is None:
                lambda_diag = jnp.zeros_like(initial_fluxes)
            if f_prior is None:
                f_prior = jnp.zeros_like(initial_fluxes)
            return jfn(initial_fluxes, images_data, batches,
                       lambda_diag, f_prior)
    else:
        vfn = partial(base, return_variances=return_variances,
                      **solver_kwargs)
        jfn = jax.jit(jax.vmap(vfn, in_axes=(0, 0, in_axes)))

        def fn(initial_fluxes, images_data, batches, penalty_weights=None):
            if penalty_weights is not None:
                raise ValueError(
                    f"penalty_weights is lasso-only (solver={solver!r})")
            return jfn(initial_fluxes, images_data, batches)

    fn._jitted = jfn   # exposed for cache/trace-count introspection in tests
    if cache:
        _solver_cache[key] = fn
    return fn


def estimate_solve_bytes_per_view(n_flux, target_shape, dtype_bytes=4):
    """Rough per-view device-memory bound for one vmapped flux solve.

    The dominant transient is the unit-flux template stack ``A`` rendered at
    the oversampled target grid (``n_flux x target_H x target_W``), plus the
    Gram/normal-equation blocks (``~2 x n_flux^2``) and the padded
    data/invvar/model planes (``~4 x target_H x target_W``). This is a
    heuristic upper-ish bound for sizing batches (see
    :func:`autotune_batch_size`'s ``per_item_bytes``), not an exact
    accounting — cuFFT workspace in particular can add a comparable amount,
    so budget conservatively on shared GPUs.
    """
    t_h, t_w = target_shape
    n_flux = int(n_flux)
    return int(dtype_bytes * (n_flux * t_h * t_w + 2 * n_flux * n_flux
                              + 4 * t_h * t_w))


def autotune_batch_size(run_batch, *, start=1, max_batch=1024, min_gain=0.10,
                        repeats=3, per_item_bytes=None,
                        mem_budget_bytes=None):
    """Pick the smallest vmap batch size at the throughput knee.

    Small views (few sources, small stamps) under-utilize the device, so
    batching more of them per solve raises throughput — but only until the
    kernels saturate; past that knee a bigger batch adds latency and memory
    pressure without more items/s (and near the memory limit it REGRESSES:
    the measured B=16/32 slowdown-then-OOM of the 100x100 vmap experiments).
    The right size is therefore the knee, not "as large as memory allows".

    Parameters
    ----------
    run_batch : callable
        ``run_batch(B)`` builds-or-reuses a size-``B`` batch, runs ONE solve
        to completion (block until ready), and returns. The first call at
        each ``B`` is the untimed warmup (jit compile); the following
        ``repeats`` calls are timed. Each distinct ``B`` is a new XLA trace,
        so candidates are the doubling sequence ``start, 2*start, ...`` —
        production batches should be padded to the chosen size (masked pad
        views are output-preserving under the engine's Jacobi ridge).
    start, max_batch : int
        Candidate range (inclusive; ``max_batch`` also caps the answer).
    min_gain : float
        Stop once doubling improves items/s by less than this fraction (or
        regresses); the previous candidate is returned.
    repeats : int
        Timed repetitions per candidate (min is taken — robust to node
        contention).
    per_item_bytes, mem_budget_bytes : int, optional
        Analytic memory cap: candidates are limited to
        ``mem_budget_bytes // per_item_bytes`` (see
        :func:`estimate_solve_bytes_per_view`). On shared GPUs pass a
        conservative budget (~half the free memory).

    Returns
    -------
    (best_B, report) : (int, dict)
        The chosen batch size and ``{B: measured items/s}`` for every
        candidate tried.
    """
    if start < 1:
        raise ValueError(f"start must be >= 1, got {start}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    mem_cap = None
    if mem_budget_bytes is not None:
        if not per_item_bytes or per_item_bytes <= 0:
            raise ValueError("mem_budget_bytes requires per_item_bytes > 0")
        mem_cap = max(1, int(mem_budget_bytes // per_item_bytes))

    import time as _time

    report = {}
    prev_b = None
    prev_tp = None
    b = start
    while b <= max_batch and (mem_cap is None or b <= mem_cap):
        run_batch(b)                    # warmup: compile, untimed
        best_t = float("inf")
        for _ in range(repeats):
            t0 = _time.perf_counter()
            run_batch(b)
            best_t = min(best_t, _time.perf_counter() - t0)
        tp = b / best_t
        report[b] = tp
        if prev_tp is not None and tp < prev_tp * (1.0 + min_gain):
            # Plateau or regression: the previous (smaller) candidate gives
            # the same-or-better items/s with less latency and memory.
            return prev_b, report
        prev_b, prev_tp = b, tp
        b *= 2
    return prev_b, report


def pad_normal_eq(G, b, lam, free=None, *, bucket=128):
    """Zero-pad a normal-equation lasso problem to a bucket multiple.

    Padding the problem to the next ``bucket`` multiple of its size means
    repeated :func:`~tractor_jax.jax.optimizer.lasso_fista_jit` calls with
    varying ``n`` hit a small set of compiled shapes instead of recompiling
    per problem.

    Dead pad slots have ``G_jj = 0`` and ``lam_j = 0``, which the solver's
    dead-slot convention pins to exactly zero — slice the solution back with
    ``f[:n]``.

    Returns ``(G_p, b_p, lam_p, free_p, n)`` as numpy arrays plus the
    original size ``n``.
    """
    G = np.asarray(G)
    n = G.shape[0]
    n_pad = ((n + bucket - 1) // bucket) * bucket
    Gp = np.zeros((n_pad, n_pad), dtype=G.dtype)
    Gp[:n, :n] = G
    bp = np.zeros(n_pad, dtype=np.asarray(b).dtype)
    bp[:n] = b
    lamp = np.zeros(n_pad, dtype=np.asarray(lam).dtype)
    lamp[:n] = lam
    freep = np.zeros(n_pad, dtype=(np.asarray(free).dtype if free is not None
                                   else np.float64))
    if free is not None:
        freep[:n] = free
    return Gp, bp, lamp, freep, n


# --------------------------------------------------------------------------- #
# Padded multi-view batch builder
# --------------------------------------------------------------------------- #
class BatchBundle(NamedTuple):
    """Everything :func:`make_batched_solver` needs for one padded batch.

    ``meta`` keys: ``max_ps``, ``max_gal``, ``max_mog_k``, ``n_flux``,
    ``bg_idx``, ``src_slot`` (per view: dict catalog index -> flux slot),
    ``counts`` (per view: ``(n_ps, n_gal)`` real, unpadded source counts).
    """
    images_data: dict
    batches: dict
    initial_fluxes: jnp.ndarray
    in_axes: dict
    meta: dict


def psf_fft_phase_ramp(fft_shape, shift_hr):
    """Unit-modulus phase ramp that translates an ``rfft2``-ed PSF.

    A translation is exactly a linear phase in the Fourier domain, so a
    sub-pixel PSF re-registration costs one elementwise multiply on an array
    the renderer already holds — no resampling, no interpolation kernel, no
    extra render cost.

    Parameters
    ----------
    fft_shape : tuple of int
        Shape of the ``rfft2`` array the ramp will multiply, ``(H, W // 2 + 1)``
        (only the last two entries are read). The real-space grid width is
        recovered as ``W = (fft_shape[-1] - 1) * 2``, the same inversion the
        renderers use — so this is only correct for the EVEN ``W`` that
        :func:`build_padded_batches` guarantees via ``_even_hr_width_pad``.
    shift_hr : (dy, dx)
        Translation in **oversampled (high-resolution) grid pixels** — the
        units of the FFT grid itself, NOT native image pixels. One native pixel
        is ``target_sampling`` HR pixels (5.0 in production). ``(dy, dx)``
        order, i.e. (axis 0, axis 1).

    Returns
    -------
    numpy.ndarray
        Complex128 ``(H, W // 2 + 1)`` ramp, built in float64 regardless of the
        dtype it will be applied to (cast at the multiply, see
        :func:`shift_psf_fft`).

    Notes
    -----
    Sign convention: the returned ramp is ``exp(-2i pi (dx f_x + dy f_y))``,
    which moves PSF content TOWARD LARGER x/y for positive ``dx``/``dy`` —
    the same sense (and the same expression) that
    :func:`~tractor_jax.jax.rendering.render_point_source_fft` uses to place a
    source at ``pos``. So ``shift_hr`` is "where to move the kernel", in the
    same direction convention as a source position.

    Consequently, if a delivered kernel's core sits at offset ``c`` relative to
    the array center that the pipeline pins on the catalog position, the shift
    to APPLY is ``-c``: for the SPHEREx L2 planes, whose core is measured at
    ``dx = dy = -0.05`` native px, the correction is ``+0.05`` native px on
    both axes.

    Frequency conventions for the ``ifftshift``-ed kernel layout: axis 0 gets
    the full ``fftfreq(H)`` set (the kernel is not halved on that axis), axis 1
    gets ``rfftfreq(W)``. The ramp is built as an outer product of two 1-D
    exponentials, which is both cheaper than a 2-D ``exp`` and exact at DC:
    ``fftfreq(H)[0] == rfftfreq(W)[0] == 0`` so ``ramp[0, 0]`` is exactly
    ``1 - 0j`` and total flux (the DC bin) is preserved BIT-EXACTLY, not
    approximately.

    One caveat, harmless at the sub-0.1-native-px scale this exists for: on an
    EVEN axis the Nyquist bin's phase factor is ``exp(-i pi d)``, whose
    imaginary part ``irfft2`` cannot represent in a real output, so a shift of
    exactly +-0.5 px annihilates that bin. For a properly oversampled kernel
    there is nothing there — measured centroid residual 0.0 for a clean 50x50
    Gaussian at ``d = 0.5``, and 2.8e-6 high-res px for a deliberately
    hard-truncated 24x24 one.
    """
    h = int(fft_shape[-2])
    wf = int(fft_shape[-1])
    w = (wf - 1) * 2
    dy = float(shift_hr[0])
    dx = float(shift_hr[1])
    # float64 throughout: the ramp is a pure geometric quantity and is cast to
    # the PSF FFT's dtype only at the multiply, so nothing already flowing
    # through the batch changes dtype.
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    ry = np.exp(-2j * np.pi * (dy * fy))
    rx = np.exp(-2j * np.pi * (dx * fx))
    return ry[:, None] * rx[None, :]


def shift_psf_fft(psf_fft, shift_hr):
    """Translate an ``rfft2``-ed PSF by ``shift_hr`` = ``(dy, dx)`` HIGH-RES
    (oversampled) pixels, via :func:`psf_fft_phase_ramp`.

    ``psf_fft`` may be a single ``(H, Wf)`` transform or a leading-axis stack
    (e.g. a ``(K, H, Wf)`` PSF basis); the ramp broadcasts. The ramp is cast to
    ``psf_fft.dtype`` before the multiply, so the result keeps the input's
    dtype (complex64 or complex128, following ``jax_enable_x64``).

    ``shift_hr = (0.0, 0.0)`` is an exact identity: the ramp is then exactly
    ``1 - 0j`` everywhere and the complex multiply is bit-preserving (the sign
    of a zero component may flip, which no numeric comparison sees).
    """
    ramp = psf_fft_phase_ramp(psf_fft.shape, shift_hr)
    return psf_fft * jnp.asarray(ramp, dtype=psf_fft.dtype)


_EVEN_PARITY_MODES = ("raise", "warn", "fix", "allow")


def _even_parity_message(ph, pw, target_sampling):
    """Message naming the exact mis-centring an even-sized kernel causes."""
    axes = []
    if ph % 2 == 0:
        axes.append(f"axis 0 (height {ph})")
    if pw % 2 == 0:
        axes.append(f"axis 1 (width {pw})")
    native = 0.5 / float(target_sampling) if target_sampling else float("nan")
    return (
        f"psf_to_fft: EVEN post-resize PSF size on {' and '.join(axes)} "
        f"(shape ({ph}, {pw})). The center-pad offset `cy - ph // 2` puts the "
        f"kernel's geometric center exactly 0.5 oversampled pixel BELOW the "
        f"ifftshift origin on each even axis, so every source rendered with "
        f"this PSF is mis-registered by exactly -0.5 high-res px "
        f"(= {native:.4g} native px at target_sampling={target_sampling}) on "
        f"that axis. Pass even_parity='fix' to remove it exactly with a "
        f"+0.5 high-res px phase ramp, or 'warn'/'allow' to keep the legacy "
        f"(mis-centered) behavior. Odd sizes are unaffected."
    )


def psf_to_fft(psf_img, *, psf_sampling, target_shape, target_sampling,
               shift_hr=None, even_parity="raise"):
    """rfft2 of a PSF resampled to ``target_sampling``, center-padded to
    ``target_shape`` and ``ifftshift``-ed — the layout ``images_data['psf']
    ['fft']`` expects. Resampling (lanczos3, flux-renormalized) only happens
    when the PSF's own oversampling ``1/psf_sampling`` differs from
    ``target_sampling``.

    Parameters
    ----------
    psf_img : array
        PSF kernel, oversampled by ``1 / psf_sampling``.
    psf_sampling : float
        PSF pixel size in native image pixels (0.2 => 5x oversampled).
    target_shape : (int, int)
        Output high-res grid ``(H, W)``; ``W`` must be even (the renderers
        recover it as ``(rfft_width - 1) * 2``).
    target_sampling : float
        Oversampling factor of the target grid (5.0 in production).
    shift_hr : (dy, dx), optional
        Sub-pixel re-registration, in **oversampled (high-res) grid pixels**
        — see :func:`psf_fft_phase_ramp` for the units and the sign
        convention (positive moves the kernel toward larger x/y; the shift to
        apply is MINUS the measured core offset). ``None`` (the default) skips
        the ramp entirely and is bit-identical to the pre-ramp behavior.
    even_parity : {"raise", "warn", "fix", "allow"}
        What to do when the (post-resize) kernel has an EVEN size on either
        axis, which makes the ``cy - ph // 2`` center-pad mis-place the kernel
        by exactly -0.5 high-res px on that axis:

        - ``"raise"`` (default) — ``ValueError`` naming the consequence;
        - ``"warn"`` — ``RuntimeWarning``, legacy (mis-centered) result;
        - ``"fix"`` — add ``+0.5`` high-res px on each even axis via the phase
          ramp, composing with ``shift_hr``; the kernel center then lands
          exactly on the ifftshift origin;
        - ``"allow"`` — silent legacy behavior.

        Odd sizes take exactly the legacy code path in every mode.
    """
    if even_parity not in _EVEN_PARITY_MODES:
        raise ValueError(f"even_parity must be one of {_EVEN_PARITY_MODES}, "
                         f"got {even_parity!r}")
    target_h, target_w = target_shape
    psf = np.asarray(psf_img, dtype=np.float64)
    ph, pw = psf.shape
    local_factor = 1.0 / psf_sampling if psf_sampling < 1.0 else 1.0
    if abs(local_factor - target_sampling) > 1e-3:
        ratio = target_sampling / local_factor
        new_shape = (int(round(ph * ratio)), int(round(pw * ratio)))
        psf_j = jax.image.resize(jnp.asarray(psf), new_shape, method="lanczos3")
        s_in = float(psf.sum())
        s_out = float(jnp.sum(psf_j))
        if s_out > 0 and s_in > 0:
            psf_j = psf_j * (s_in / s_out)
        psf = np.asarray(psf_j)
        ph, pw = psf.shape

    # LATENT PARITY BUG, made explicit. `y0 = cy - ph // 2` lands the kernel's
    # geometric center on `cy` (which ifftshift sends to the origin) only for
    # ODD ph: for even ph the center is at ph/2 - 0.5, so it lands at cy - 0.5,
    # i.e. every source is rendered -0.5 high-res px off. Production is safe
    # only because the delivered kernels happen to be odd (51, 73) and
    # psf_sampling=0.2 / target_sampling=5.0 make 1.0/0.2 == 5.0 bit-exactly,
    # so the resize branch above (which could produce an even shape) never runs.
    fix_y = fix_x = 0.0
    if ph % 2 == 0 or pw % 2 == 0:
        msg = _even_parity_message(ph, pw, target_sampling)
        if even_parity == "raise":
            raise ValueError(msg)
        elif even_parity == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        elif even_parity == "fix":
            fix_y = 0.5 if ph % 2 == 0 else 0.0
            fix_x = 0.5 if pw % 2 == 0 else 0.0

    pad_psf = np.zeros((target_h, target_w), dtype=np.float64)
    cy, cx = target_h // 2, target_w // 2
    y0 = cy - ph // 2
    x0 = cx - pw // 2
    pad_psf[y0:y0 + ph, x0:x0 + pw] = psf
    pad_psf = np.fft.ifftshift(pad_psf)
    out = jfft.rfft2(jnp.asarray(pad_psf))
    if shift_hr is not None or fix_y or fix_x:
        dy = fix_y + (float(shift_hr[0]) if shift_hr is not None else 0.0)
        dx = fix_x + (float(shift_hr[1]) if shift_hr is not None else 0.0)
        out = shift_psf_fft(out, (dy, dx))
    return out


def slice_fluxes(fluxes, meta):
    """Per-view flux arrays in ``[ps..., gal...]`` order, real sources only
    (padding slots dropped), from a solved ``(n_views, n_flux)`` stack."""
    max_ps = meta["max_ps"]
    out = []
    for c, (n_ps, n_gal) in enumerate(meta["counts"]):
        fc = np.asarray(fluxes[c])
        out.append(np.concatenate([fc[:n_ps], fc[max_ps:max_ps + n_gal]]))
    return out


def build_padded_batches(
    views,
    catalog,
    sx,
    sy,
    *,
    psf_sampling,
    fixed_max_factor=None,
    fit_background=True,
    profile_lookup_fn=None,
    cd_inv=None,
    max_ps_cap=None,
    max_gal_cap=None,
    max_mog_k_cap=None,
    pad_bucket=None,
    psf_fft_cache=None,
    even_parity="raise",
    dtype=np.float32,
):
    """Pad + stack per-view source problems into one vmap-ready batch.

    Each *view* is an independently-solved postage stamp sharing one parent
    catalog — a halo tile of a larger cutout, or a per-target window. All
    views must share one data shape.

    Parameters
    ----------
    views : list of dict
        Per view: ``data`` (h, w), ``invvar`` (h, w), ``psf`` (2-D array,
        oversampled by ``1/psf_sampling``), ``src_indices`` (catalog rows in
        the parent catalog). Optionally ``psf_basis`` (sequence of K kernels,
        the SAME object across views so its transforms are computed once) and
        ``psf_weights`` (K,), in which case the view's PSF is
        ``sum_k psf_weights[k] * psf_basis[k]``, blended in the Fourier
        domain — a spatially-varying PSF at the cost of one complex weighted
        sum per view instead of one transform per view. ``psf`` is still
        required (it sets nothing but is kept for shape/back-compat), and if
        any view supplies ``psf_basis`` then all must.
        this view), ``origin`` ``(x0, y0)`` of the view in the ``sx``/``sy``
        pixel frame.

        Optionally also, for sub-pixel PSF re-registration (opt-in; absent =
        no ramp = bit-identical to not passing it at all):

        ``psf_shift`` ``(dy, dx)``
            One shift for this view's PSF, in **NATIVE image pixels** (this is
            the caller's unit; the builder multiplies by ``target_sampling`` to
            reach the high-res FFT grid). Positive moves the kernel toward
            larger x/y, so the value to pass is MINUS the measured kernel core
            offset — for the SPHEREx L2 planes, whose core sits at
            ``dx = dy ~ -0.05`` native px, pass ``(+0.05, +0.05)``. On the
            ``psf_basis`` path this is the degenerate "every zone kernel shares
            one shift" case and is applied to every basis element.
        ``psf_basis_shifts`` ``(K, 2)``
            One ``(dy, dx)`` per ``psf_basis`` element, in NATIVE pixels.
            Each zone kernel has its own core offset, so the ramp is applied
            PER BASIS ELEMENT BEFORE the weighted blend:
            ``sum_k w_k * (rfft2(K_k) * phase_k)``. (Ramping after the blend
            is only valid when every element shares one shift — that is what
            ``psf_shift`` is for.) Requires ``psf_basis``.

        Like ``psf_basis``, each key is all-or-none across the batch, and
        ``psf_shift`` and ``psf_basis_shifts`` are mutually exclusive.
    catalog : table
        Needs ``shape_r`` (0 => point source), ``shape_ab``, ``shape_phi``,
        ``sersic`` columns for galaxies.
    sx, sy : array
        Source pixel positions in the parent frame, indexed by catalog row.
    psf_sampling : float
        PSF pixel size in image pixels (e.g. 0.2 for 5x oversampling).
    fixed_max_factor : float, optional
        Oversampled-rendering factor; default ``1/psf_sampling`` (or 1).
    profile_lookup_fn : callable, optional
        ``sersic -> MixtureOfGaussians``; default is the memoized
        :meth:`SersicMixture.getProfile`. Looked up once per galaxy.
    cd_inv : (2, 2) array, optional
        Inverse CD matrix shared by every view (views are translated crops of
        one parent WCS). Identity if omitted; required in practice whenever
        galaxies are fit.
    max_ps_cap, max_gal_cap, max_mog_k_cap : int, optional
        Fixed per-view array widths; a view exceeding a cap raises
        ``ValueError`` instead of silently changing the padded shape (which
        would force an XLA retrace).
    pad_bucket : int, optional
        Round the NATURAL ``max_ps``/``max_gal`` up to a multiple of this
        instead of fixing them with caps: a few padded shapes per field
        (one retrace each) in exchange for near-natural matrix sizes. This
        is the middle ground between caps (one shape, maximal padding —
        e.g. a 465-wide eigfloor eigh paying the field maximum on every
        cutout) and no caps (per-cutout retraces). Caps, if also given,
        still validate/override after bucketing. Masked pad slots carry no source; with
        the Jacobi per-source ridge the padding is output-preserving (exact
        under eigfloor/x64; ~1e-5 bright-source level for float32 linear,
        larger only for near-null-space fluxes in over-crowded views).
    psf_fft_cache : MutableMapping, optional
        Caller-owned cross-call cache for PSF FFTs, keyed on the PSF object's
        identity, the FFT geometry and the requested sub-pixel shift. The
        caller must keep the PSF arrays alive for the cache's lifetime (id
        reuse after gc would alias).
    even_parity : {"raise", "warn", "fix", "allow"}
        Passed to :func:`psf_to_fft`; governs EVEN-sized (post-resize) PSF
        kernels, which the center-pad mis-places by exactly -0.5 high-res px
        per even axis. Default ``"raise"``. Odd kernels are unaffected in every
        mode.

    Returns
    -------
    BatchBundle
    """
    n_views = len(views)
    if n_views == 0:
        raise ValueError("build_padded_batches: views is empty")
    if profile_lookup_fn is None:
        profile_lookup_fn = SersicMixture.getProfile

    base_h, base_w = views[0]["data"].shape
    for v in views:
        if v["data"].shape != (base_h, base_w):
            raise ValueError("All views must share the same data shape")

    # Sub-pixel PSF re-registration keys: opt-in, and all-or-none per batch
    # (same discipline as psf_basis — a half-shifted batch would silently mix
    # two registrations into one solve).
    use_basis = any(v.get("psf_basis") is not None for v in views)
    n_shift = sum(v.get("psf_shift") is not None for v in views)
    n_bshift = sum(v.get("psf_basis_shifts") is not None for v in views)
    if n_shift and n_shift != n_views:
        raise ValueError("build_padded_batches: psf_shift must be given for "
                         "every view or for none")
    if n_bshift and n_bshift != n_views:
        raise ValueError("build_padded_batches: psf_basis_shifts must be given "
                         "for every view or for none")
    if n_shift and n_bshift:
        raise ValueError("build_padded_batches: psf_shift and "
                         "psf_basis_shifts are mutually exclusive")
    if n_bshift and not use_basis:
        raise ValueError("build_padded_batches: psf_basis_shifts requires "
                         "psf_basis")

    def _psf_shapes(v):
        b = v.get("psf_basis")
        if b is not None:
            return [np.shape(k) for k in b]
        return [v["psf"].shape]

    max_psf_h = max(s[0] for v in views for s in _psf_shapes(v))
    max_psf_w = max(s[1] for v in views for s in _psf_shapes(v))
    if fixed_max_factor is None:
        fixed_max_factor = 1.0 / psf_sampling if psf_sampling < 1.0 else 1.0
    max_factor = float(fixed_max_factor)
    target_sampling = max_factor if max_factor > 1.0 else 1.0

    fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))
    padded_h = base_h + fft_pad_h_lr
    # An ODD high-res width is silently narrowed by one column downstream:
    # the renderers recover it from the rfft2 array as `(shape[1]-1)*2`. The
    # grid then no longer maps at an integer HR->LR factor, `downsample_image`
    # drops to its boxcar path, and every template is resampled (~5% in
    # sum(t^2), ~2% in flux). Buy the even width with extra LOW-RES padding,
    # which keeps the factor exact — see `_even_hr_width_pad`.
    padded_w = _even_hr_width_pad(base_w + fft_pad_w_lr, max_factor)
    target_h = int(round(padded_h * max_factor))
    target_w = int(round(padded_w * max_factor))

    # Classify sources per view (vectorized) and look each DISTINCT sersic
    # index's MoG profile up once (catalogs carry ~1e2 distinct values for
    # ~1e4 galaxy slots).
    shape_r_arr = np.asarray(catalog["shape_r"], dtype=np.float64)
    classification = []
    max_ps = 0
    max_gal = 0
    for v in views:
        src = np.asarray(v["src_indices"], dtype=np.intp)
        isgal = shape_r_arr[src] > 0 if src.size else np.zeros(0, bool)
        ps_idx = src[~isgal]
        gal_idx = src[isgal]
        classification.append((ps_idx, gal_idx))
        max_ps = max(max_ps, len(ps_idx))
        max_gal = max(max_gal, len(gal_idx))

    all_gal_ci = (np.concatenate([g for _, g in classification])
                  if max_gal else np.zeros(0, np.intp))
    if all_gal_ci.size:
        sersic_arr = np.asarray(catalog["sersic"], dtype=np.float64)
        uniq_sersic, gal_prof_inv = np.unique(sersic_arr[all_gal_ci],
                                              return_inverse=True)
        uniq_profs = [profile_lookup_fn(float(s)) for s in uniq_sersic]
        max_mog_k = max((len(p.amp) for p in uniq_profs), default=1)
    else:
        uniq_profs, gal_prof_inv = [], np.zeros(0, np.intp)
        max_mog_k = 1

    # Bucketed padding: round natural widths up so a field yields only a
    # few distinct padded shapes while each stays near its natural size.
    if pad_bucket:
        max_ps = int(math.ceil(max(max_ps, 1) / pad_bucket) * pad_bucket)
        max_gal = int(math.ceil(max(max_gal, 1) / pad_bucket) * pad_bucket)

    # Fixed-shape caps: pad the per-view widths so the jitted solver sees one
    # shape across every batch built with the same caps.
    if max_ps_cap is not None:
        if max_ps > max_ps_cap:
            raise ValueError(
                f"max_ps={max_ps} exceeds cap {max_ps_cap}")
        max_ps = max_ps_cap
    if max_gal_cap is not None:
        if max_gal > max_gal_cap:
            raise ValueError(
                f"max_gal={max_gal} exceeds cap {max_gal_cap}")
        max_gal = max_gal_cap
    if max_mog_k_cap is not None:
        if max_mog_k > max_mog_k_cap:
            raise ValueError(
                f"max_mog_K={max_mog_k} exceeds cap {max_mog_k_cap}")
        max_mog_k = max_mog_k_cap

    # Per-view flux layout: [point sources... | galaxies... | (background)]
    n_flux = max_ps + max_gal + (1 if fit_background else 0)
    bg_idx = max_ps + max_gal

    base_dtype = views[0]["data"].dtype
    data_arr = np.zeros((n_views, base_h, base_w), dtype=base_dtype)
    iv_arr = np.zeros((n_views, base_h, base_w),
                      dtype=views[0]["invvar"].dtype)
    for vi, view in enumerate(views):
        data_arr[vi] = view["data"]
        iv_arr[vi] = view["invvar"]
    d_pad = np.zeros((n_views, padded_h, padded_w), dtype=dtype)
    iv_pad = np.zeros((n_views, padded_h, padded_w), dtype=dtype)
    d_pad[:, :base_h, :base_w] = data_arr
    iv_pad[:, :base_h, :base_w] = iv_arr

    # Source POSITIONS are geometry, not pixel data: they are built in float64
    # and left there. Storing them at `dtype` (float32 by default) quantised
    # every requested position to ~1e-7 relative — e.g. x=10.2 became
    # 10.19999980926514 — a 2e-7..1.6e-6 native-px registration error over a
    # 40..100 px stamp, which is pure loss and of the same kind as (though far
    # smaller than) the PSF core offset this module's phase ramp corrects.
    # Nothing downstream depends on their being float32: `pos_pix` is only ever
    # multiplied by the per-axis HR factors (`pos_pix * f_xy + (f_xy - 1) / 2`,
    # a float64 `jnp.array` under x64) and fed to the render phase ramp, so the
    # dtype of the rendered templates is unchanged; and with `jax_enable_x64`
    # off, `jnp.asarray` downcasts these to float32 exactly as before, so the
    # arrays reaching the device are bit-identical there.
    pos_dtype = np.float64
    ps_pos = np.zeros((n_views, max_ps, 2), dtype=pos_dtype)
    ps_fidx = np.zeros((n_views, max_ps), dtype=np.int32)
    ps_mask = np.zeros((n_views, max_ps), dtype=dtype)

    gal_pos = np.zeros((n_views, max_gal, 2), dtype=pos_dtype)
    gal_fidx = np.zeros((n_views, max_gal), dtype=np.int32)
    gal_mask = np.zeros((n_views, max_gal), dtype=dtype)
    gal_cd = np.tile(np.eye(2, dtype=dtype), (n_views, max_gal, 1, 1))
    gal_shape = np.zeros((n_views, max_gal, 3), dtype=dtype)
    gal_amp = np.zeros((n_views, max_gal, max_mog_k), dtype=dtype)
    gal_mean = np.zeros((n_views, max_gal, max_mog_k, 2), dtype=dtype)
    gal_var = np.tile(np.eye(2, dtype=dtype),
                      (n_views, max_gal, max_mog_k, 1, 1))
    init_flux = np.zeros((n_views, n_flux), dtype=dtype)

    cd_inv = (np.eye(2, dtype=dtype) if cd_inv is None
              else np.asarray(cd_inv, dtype=dtype))

    sx_arr = np.asarray(sx, dtype=np.float64)
    sy_arr = np.asarray(sy, dtype=np.float64)
    origins = np.asarray([v["origin"] for v in views], dtype=np.float64)

    # Flat (view, within-view-slot, catalog-row) index triplets for one
    # fancy-indexed scatter per array instead of a Python loop per source.
    def _flat(idx_lists):
        if not any(len(x) for x in idx_lists):
            return (np.zeros(0, np.intp),) * 3
        vids = np.concatenate([np.full(len(x), vi, np.intp)
                               for vi, x in enumerate(idx_lists)])
        ks = np.concatenate([np.arange(len(x), dtype=np.intp)
                             for x in idx_lists])
        cis = np.concatenate([np.asarray(x, dtype=np.intp)
                              for x in idx_lists])
        return vids, ks, cis

    ps_v, ps_k, ps_ci = _flat([c[0] for c in classification])
    gal_v, gal_k, gal_ci = _flat([c[1] for c in classification])

    def _seed(vids, ks, cis, slot_offset):
        """init flux = the data pixel under each in-bounds source."""
        px = sx_arr[cis] - origins[vids, 0]
        py = sy_arr[cis] - origins[vids, 1]
        ix = np.rint(px).astype(np.intp)
        iy = np.rint(py).astype(np.intp)
        ok = (0 <= ix) & (ix < base_w) & (0 <= iy) & (iy < base_h)
        vals = data_arr[vids[ok], iy[ok], ix[ok]].astype(np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        init_flux[vids[ok], slot_offset + ks[ok]] = vals
        return px, py

    if ps_ci.size:
        px, py = _seed(ps_v, ps_k, ps_ci, 0)
        ps_pos[ps_v, ps_k, 0] = px
        ps_pos[ps_v, ps_k, 1] = py
        ps_fidx[ps_v, ps_k] = ps_k
        ps_mask[ps_v, ps_k] = 1.0

    if gal_ci.size:
        px, py = _seed(gal_v, gal_k, gal_ci, max_ps)
        gal_pos[gal_v, gal_k, 0] = px
        gal_pos[gal_v, gal_k, 1] = py
        gal_fidx[gal_v, gal_k] = max_ps + gal_k
        gal_mask[gal_v, gal_k] = 1.0
        gal_cd[gal_v, gal_k] = cd_inv
        gal_shape[gal_v, gal_k, 0] = shape_r_arr[gal_ci]
        gal_shape[gal_v, gal_k, 1] = np.asarray(catalog["shape_ab"],
                                                dtype=np.float64)[gal_ci]
        gal_shape[gal_v, gal_k, 2] = np.asarray(catalog["shape_phi"],
                                                dtype=np.float64)[gal_ci]
        # profiles: one scatter per DISTINCT sersic value
        for u_i, prof in enumerate(uniq_profs):
            sel = gal_prof_inv == u_i
            K = len(prof.amp)
            gal_amp[gal_v[sel], gal_k[sel], :K] = np.asarray(prof.amp,
                                                             dtype=dtype)
            gal_mean[gal_v[sel], gal_k[sel], :K] = np.asarray(prof.mean,
                                                              dtype=dtype)
            gal_var[gal_v[sel], gal_k[sel], :K] = np.asarray(prof.var,
                                                             dtype=dtype)

    src_slot_per_view = []
    counts = []
    for ps_idx, gal_idx in classification:
        slots = {int(ci): k for k, ci in enumerate(ps_idx)}
        slots.update({int(ci): max_ps + k for k, ci in enumerate(gal_idx)})
        src_slot_per_view.append(slots)
        counts.append((len(ps_idx), len(gal_idx)))

    # Sub-pixel PSF re-registration. Callers give shifts in NATIVE image
    # pixels; the FFT grid is oversampled by target_sampling, and the batch
    # path maps at EXACTLY that factor per axis (target_h = padded_h *
    # max_factor and _even_hr_width_pad keeps target_w = padded_w *
    # max_factor), which is the same factor the renderers use to scale source
    # positions. So one native px is target_sampling high-res px, exactly.
    def _shift_hr(shift_native):
        if shift_native is None:
            return None
        s = np.asarray(shift_native, dtype=np.float64).reshape(-1)
        if s.size != 2:
            raise ValueError("psf shift must be a (dy, dx) pair, got shape "
                             f"{np.shape(shift_native)}")
        return (float(s[0]) * target_sampling, float(s[1]) * target_sampling)

    def _shift_key(shift_hr):
        return None if shift_hr is None else (shift_hr[0], shift_hr[1])

    # PSF FFTs: transform once per distinct (PSF object, requested shift) — the
    # object identity holds within this call — and broadcast when every view
    # shares one PSF; the optional caller-owned psf_fft_cache extends the reuse
    # across calls.
    def _fft_for(psf_arr, shift_hr=None):
        if psf_fft_cache is not None:
            key = (id(psf_arr), psf_arr.shape, target_h, target_w,
                   round(target_sampling, 9), round(psf_sampling, 9),
                   _shift_key(shift_hr), even_parity)
            hit = psf_fft_cache.get(key)
            if hit is not None:
                return hit
        fft = psf_to_fft(psf_arr, psf_sampling=psf_sampling,
                         target_shape=(target_h, target_w),
                         target_sampling=target_sampling,
                         shift_hr=shift_hr, even_parity=even_parity)
        if psf_fft_cache is not None:
            psf_fft_cache[key] = fft
        return fft

    # Spatially-varying PSF, blended in the FOURIER domain. A view may carry
    # ``psf_basis`` (K kernels, one per PSF zone, shared object across views)
    # plus ``psf_weights`` (K,), meaning "this view's PSF is sum_k w_k basis_k".
    # Every step of psf_to_fft — lanczos resize, zero-pad, ifftshift, rfft2 —
    # is linear, so FFT(sum_k w_k K_k) = sum_k w_k FFT(K_k) and the K basis
    # transforms can be shared by every view instead of one transform per view.
    # Measured on A2055 (324 tiles, 9 zones): 38 ms/cutout against 177 ms for
    # a per-view transform. The one non-linearity is psf_to_fft's post-resize
    # sum renormalization, whose per-kernel scale factors differ; with
    # sum-normalized zone kernels the residual is ~5e-7 of the DC term.
    #
    # Sub-pixel re-registration rides along for free: each zone kernel carries
    # its OWN core offset, so the ramp goes on PER BASIS ELEMENT and BEFORE the
    # blend, ``sum_k w_k * (rfft2(K_k) * phase_k)``. (One post-blend ramp is
    # only equivalent when every element shares one shift — that is what
    # ``psf_shift`` is for.) Because the ramped basis depends on the basis and
    # the shift table but NOT on a view's weights, it is cached exactly like the
    # unramped one: a batch whose views share one per-(detector, zone) shift
    # table pays for the ramp once, not once per view (measured on 100 views x
    # K=9 at a 255x131 rfft grid: 33.0 ms unshifted, 33.9 ms with one shared
    # psf_shift, 36.2 ms with a per-zone table — against 83.3 ms when the
    # multiply is redone per view).
    basis_fft_cache = {}
    ramp_cache = {}

    def _ramp(shift_hr, dtype):
        key = (_shift_key(shift_hr), np.dtype(dtype).str)
        hit = ramp_cache.get(key)
        if hit is None:
            hit = jnp.asarray(
                psf_fft_phase_ramp((target_h, target_w // 2 + 1), shift_hr),
                dtype=dtype)
            ramp_cache[key] = hit
        return hit

    shift_keys_cache = {}

    def _basis_shift_keys(bshifts, n_basis):
        """Per-element high-res shifts of one ``psf_basis_shifts`` table.

        Memoized on the table's identity — like ``psf_basis`` itself, views are
        expected to hand over the SAME per-(detector, zone) object — so the
        native->high-res conversion runs once per table, not once per view.
        """
        cid = id(bshifts)
        hit = shift_keys_cache.get(cid)
        if hit is None:
            sh = np.asarray(bshifts, dtype=np.float64)
            if sh.shape != (n_basis, 2):
                raise ValueError(
                    "psf_basis_shifts must have shape (K, 2) matching "
                    f"psf_basis; got {sh.shape} for K={n_basis}")
            hit = tuple(_shift_key(_shift_hr(s)) for s in sh)
            shift_keys_cache[cid] = hit
        return hit

    def _basis_fft(basis, shift_keys=None):
        """(K, H, W//2+1) transforms of one basis, optionally phase-ramped.

        ``shift_keys`` is ``None`` (no ramp), a 1-tuple (one shift shared by
        every element) or a K-tuple (one per element), each entry a high-res
        ``(dy, dx)``.
        """
        key = (id(basis), shift_keys)
        hit = basis_fft_cache.get(key)
        if hit is not None:
            return hit
        base = basis_fft_cache.get((id(basis), None))
        if base is None:
            base = jnp.stack([_fft_for(k) for k in basis])
            basis_fft_cache[(id(basis), None)] = base
        if shift_keys is None:
            hit = base
        elif len(shift_keys) == 1:
            hit = base * _ramp(shift_keys[0], base.dtype)
        else:
            hit = base * jnp.stack([_ramp(s, base.dtype) for s in shift_keys])
        basis_fft_cache[key] = hit
        return hit

    if use_basis:
        if not all(v.get("psf_basis") is not None for v in views):
            raise ValueError("build_padded_batches: psf_basis must be given "
                             "for every view or for none")
        # Blend per GROUP of views sharing (basis, shifts), not per view. In
        # the driver every tile of a cutout hands over the SAME basis object
        # and the SAME shift table, so this is one group and the whole blend
        # is a single (T, K) x (K, H, Wf) tensordot — one dispatch instead of
        # one per tile (~50/cutout), which put the eager per-tile launches on
        # the CPU-bound build stage's critical path (measured +21.6 ms/cutout
        # at K=9-12; see proj research note 2026-07-28-node-comparison, §PSF).
        groups = {}                       # gkey -> [bf, [view idx], [weights]]
        gorder = []
        for i, v in enumerate(views):
            basis = v["psf_basis"]
            n_basis = len(basis)
            w = np.asarray(v["psf_weights"], dtype=np.float64)
            if w.shape[0] != n_basis:
                raise ValueError("psf_weights length must match psf_basis")
            bshifts = v.get("psf_basis_shifts")
            if bshifts is not None:
                shift_keys = _basis_shift_keys(bshifts, n_basis)
            elif v.get("psf_shift") is not None:
                shift_keys = (_shift_key(_shift_hr(v["psf_shift"])),)
            else:
                shift_keys = None
            gkey = (id(basis), shift_keys)
            g = groups.get(gkey)
            if g is None:
                bf = (_basis_fft(basis) if shift_keys is None
                      else _basis_fft(basis, shift_keys))
                g = groups[gkey] = [bf, [], []]
                gorder.append(gkey)
            g[1].append(i)
            g[2].append(w)
        # precision="highest": the grouped blend is a real GEMM, which XLA
        # would otherwise run in TF32 on Ampere+ (10-bit mantissa, ~5e-4
        # relative on the kernel -- most of the fp32 parity budget). The
        # per-view path was a matvec and never hit TF32, so pin full fp32.
        if len(groups) == 1:
            bf, _, ws = groups[gorder[0]]
            wmat = jnp.asarray(np.stack(ws), dtype=bf.real.dtype)   # (T, K)
            psf_fft_stack = jnp.tensordot(wmat, bf, axes=(1, 0),
                                          precision="highest")
        else:
            blended, idx = [], []
            for gkey in gorder:
                bf, idxs, ws = groups[gkey]
                wmat = jnp.asarray(np.stack(ws), dtype=bf.real.dtype)
                blended.append(jnp.tensordot(wmat, bf, axes=(1, 0),
                                              precision="highest"))
                idx.extend(idxs)
            # invert the group-major ordering back to view order
            perm = np.argsort(np.asarray(idx))
            psf_fft_stack = jnp.concatenate(blended, axis=0)[perm]
    else:
        unique_fft = {}
        for v in views:
            key = (id(v["psf"]), _shift_key(_shift_hr(v.get("psf_shift"))))
            if key not in unique_fft:
                unique_fft[key] = _fft_for(v["psf"],
                                           _shift_hr(v.get("psf_shift")))
        if len(unique_fft) == 1:
            one = next(iter(unique_fft.values()))
            psf_fft_stack = jnp.broadcast_to(one[None], (n_views,) + one.shape)
        else:
            psf_fft_stack = jnp.stack([
                unique_fft[(id(v["psf"]),
                            _shift_key(_shift_hr(v.get("psf_shift"))))]
                for v in views])

    images_data = {
        "data": jnp.asarray(d_pad),
        "invvar": jnp.asarray(iv_pad),
        "psf": {
            "type_code": jnp.zeros(n_views, dtype=jnp.int32),
            "sampling": jnp.full(n_views, target_sampling, dtype=jnp.float32),
            "fft": psf_fft_stack,
            "amp": jnp.zeros((n_views, 1)),
            "mean": jnp.zeros((n_views, 1, 2)),
            "var": jnp.tile(jnp.eye(2), (n_views, 1, 1, 1)),
        },
    }

    batches = {}
    if max_ps > 0:
        batches["PointSource"] = {
            "flux_idx": jnp.asarray(ps_fidx),
            "pos_pix": jnp.asarray(ps_pos),
            "mask": jnp.asarray(ps_mask),
        }
    if max_gal > 0:
        batches["Galaxy"] = {
            "flux_idx": jnp.asarray(gal_fidx),
            "pos_pix": jnp.asarray(gal_pos),
            "wcs_cd_inv": jnp.asarray(gal_cd),
            "shapes": jnp.asarray(gal_shape),
            "mask": jnp.asarray(gal_mask),
            "profile": {
                "amp": jnp.asarray(gal_amp),
                "mean": jnp.asarray(gal_mean),
                "var": jnp.asarray(gal_var),
            },
        }
    if fit_background:
        batches["Background"] = {"flux_idx": jnp.asarray([bg_idx],
                                                         dtype=jnp.int32)}

    initial_fluxes = jnp.asarray(init_flux, dtype=dtype)

    meta = {"max_ps": max_ps, "max_gal": max_gal, "max_mog_k": max_mog_k,
            "n_flux": n_flux, "bg_idx": bg_idx,
            "src_slot": src_slot_per_view, "counts": counts}
    return BatchBundle(images_data, batches, initial_fluxes,
                       batches_in_axes(batches), meta)
