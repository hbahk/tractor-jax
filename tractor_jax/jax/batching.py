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
"""
import math
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft

from tractor_jax.jax.optimizer import (
    solve_fluxes_linear,
    solve_fluxes_eigfloor,
    solve_fluxes_lasso,
)
from tractor_jax.sersic import SersicMixture

__all__ = [
    "BatchBundle",
    "build_padded_batches",
    "psf_to_fft",
    "slice_fluxes",
    "batches_in_axes",
    "make_batched_solver",
    "clear_solver_cache",
    "penalty_weights_from_slots",
    "pad_normal_eq",
]

_SOLVER_FNS = {
    "linear": solve_fluxes_linear,
    "eigfloor": solve_fluxes_eigfloor,
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
    solver : {"linear", "eigfloor", "lasso"}
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


def pad_normal_eq(G, b, lam, free=None, *, bucket=128):
    """Zero-pad a normal-equation lasso problem to the next ``bucket``
    multiple of its size, so repeated :func:`~tractor_jax.jax.optimizer.
    lasso_fista_jit` calls with varying ``n`` hit a small set of compiled
    shapes instead of recompiling per problem.

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


def psf_to_fft(psf_img, *, psf_sampling, target_shape, target_sampling):
    """rfft2 of a PSF resampled to ``target_sampling``, center-padded to
    ``target_shape`` and ``ifftshift``-ed — the layout ``images_data['psf']
    ['fft']`` expects. Resampling (lanczos3, flux-renormalized) only happens
    when the PSF's own oversampling ``1/psf_sampling`` differs from
    ``target_sampling``."""
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
    pad_psf = np.zeros((target_h, target_w), dtype=np.float64)
    cy, cx = target_h // 2, target_w // 2
    y0 = cy - ph // 2
    x0 = cx - pw // 2
    pad_psf[y0:y0 + ph, x0:x0 + pw] = psf
    pad_psf = np.fft.ifftshift(pad_psf)
    return jfft.rfft2(jnp.asarray(pad_psf))


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
    psf_fft_cache=None,
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
        this view), ``origin`` ``(x0, y0)`` of the view in the ``sx``/``sy``
        pixel frame.
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
        would force an XLA retrace). Masked pad slots carry no source; with
        the Jacobi per-source ridge the padding is output-preserving (exact
        under eigfloor/x64; ~1e-5 bright-source level for float32 linear,
        larger only for near-null-space fluxes in over-crowded views).
    psf_fft_cache : MutableMapping, optional
        Caller-owned cross-call cache for PSF FFTs, keyed on the PSF object's
        identity and the FFT geometry. The caller must keep the PSF arrays
        alive for the cache's lifetime (id reuse after gc would alias).

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

    max_psf_h = max(v["psf"].shape[0] for v in views)
    max_psf_w = max(v["psf"].shape[1] for v in views)
    if fixed_max_factor is None:
        fixed_max_factor = 1.0 / psf_sampling if psf_sampling < 1.0 else 1.0
    max_factor = float(fixed_max_factor)
    target_sampling = max_factor if max_factor > 1.0 else 1.0

    fft_pad_h_lr = int(math.ceil(max_psf_h / max_factor))
    fft_pad_w_lr = int(math.ceil(max_psf_w / max_factor))
    padded_h = base_h + fft_pad_h_lr
    padded_w = base_w + fft_pad_w_lr
    target_h = int(round(padded_h * max_factor))
    target_w = int(round(padded_w * max_factor))

    # Classify sources per view; look each galaxy's MoG profile up ONCE here
    # and stash it for the fill loop below.
    classification = []
    max_ps = 0
    max_gal = 0
    max_mog_k = 1
    shape_r = catalog["shape_r"]
    for v in views:
        ps_idx, gal_idx, gal_profs = [], [], []
        for ci in v["src_indices"]:
            if shape_r[ci] == 0:
                ps_idx.append(ci)
            else:
                gal_idx.append(ci)
                prof = profile_lookup_fn(float(catalog["sersic"][ci]))
                gal_profs.append(prof)
                max_mog_k = max(max_mog_k, len(prof.amp))
        classification.append((ps_idx, gal_idx, gal_profs))
        max_ps = max(max_ps, len(ps_idx))
        max_gal = max(max_gal, len(gal_idx))

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

    data_list = []
    invvar_list = []
    init_flux_list = []
    src_slot_per_view = []
    counts = []

    ps_pos = np.zeros((n_views, max_ps, 2), dtype=dtype)
    ps_fidx = np.zeros((n_views, max_ps), dtype=np.int32)
    ps_mask = np.zeros((n_views, max_ps), dtype=dtype)

    gal_pos = np.zeros((n_views, max_gal, 2), dtype=dtype)
    gal_fidx = np.zeros((n_views, max_gal), dtype=np.int32)
    gal_mask = np.zeros((n_views, max_gal), dtype=dtype)
    gal_cd = np.tile(np.eye(2, dtype=dtype), (n_views, max_gal, 1, 1))
    gal_shape = np.zeros((n_views, max_gal, 3), dtype=dtype)
    gal_amp = np.zeros((n_views, max_gal, max_mog_k), dtype=dtype)
    gal_mean = np.zeros((n_views, max_gal, max_mog_k, 2), dtype=dtype)
    gal_var = np.tile(np.eye(2, dtype=dtype),
                      (n_views, max_gal, max_mog_k, 1, 1))

    cd_inv = (np.eye(2, dtype=dtype) if cd_inv is None
              else np.asarray(cd_inv, dtype=dtype))

    shape_ab = catalog["shape_ab"] if max_gal else None
    shape_phi = catalog["shape_phi"] if max_gal else None

    for vi, view in enumerate(views):
        d = view["data"]
        iv = view["invvar"]
        h, w = d.shape

        d_pad = np.zeros((padded_h, padded_w), dtype=dtype)
        iv_pad = np.zeros((padded_h, padded_w), dtype=dtype)
        d_pad[:h, :w] = d
        iv_pad[:h, :w] = iv
        data_list.append(jnp.asarray(d_pad))
        invvar_list.append(jnp.asarray(iv_pad))

        ps_indices, gal_indices, gal_profs = classification[vi]
        counts.append((len(ps_indices), len(gal_indices)))

        slots = {}
        for k, ci in enumerate(ps_indices):
            slots[ci] = k
        for k, ci in enumerate(gal_indices):
            slots[ci] = max_ps + k
        src_slot_per_view.append(slots)

        init_f = np.zeros(n_flux, dtype=dtype)
        x0, y0 = view["origin"]

        def _seed_init(px, py, slot):
            ix, iy = int(round(px)), int(round(py))
            if 0 <= iy < h and 0 <= ix < w:
                val = float(d[iy, ix])
                init_f[slot] = val if np.isfinite(val) else 0.0

        for k, ci in enumerate(ps_indices):
            px = float(sx[ci]) - x0
            py = float(sy[ci]) - y0
            ps_pos[vi, k, 0] = px
            ps_pos[vi, k, 1] = py
            ps_fidx[vi, k] = slots[ci]
            ps_mask[vi, k] = 1.0
            _seed_init(px, py, slots[ci])

        for k, ci in enumerate(gal_indices):
            px = float(sx[ci]) - x0
            py = float(sy[ci]) - y0
            gal_pos[vi, k, 0] = px
            gal_pos[vi, k, 1] = py
            gal_fidx[vi, k] = slots[ci]
            gal_mask[vi, k] = 1.0
            gal_cd[vi, k] = cd_inv
            gal_shape[vi, k, 0] = float(shape_r[ci])
            gal_shape[vi, k, 1] = float(shape_ab[ci])
            gal_shape[vi, k, 2] = float(shape_phi[ci])
            prof = gal_profs[k]
            K = len(prof.amp)
            gal_amp[vi, k, :K] = np.asarray(prof.amp, dtype=dtype)
            gal_mean[vi, k, :K] = np.asarray(prof.mean, dtype=dtype)
            gal_var[vi, k, :K] = np.asarray(prof.var, dtype=dtype)
            _seed_init(px, py, slots[ci])

        init_flux_list.append(init_f)

    # PSF FFTs: transform once per distinct PSF object (identity within this
    # call) and broadcast when every view shares one PSF; the optional
    # caller-owned psf_fft_cache extends the reuse across calls.
    def _fft_for(psf_arr):
        if psf_fft_cache is not None:
            key = (id(psf_arr), psf_arr.shape, target_h, target_w,
                   round(target_sampling, 9), round(psf_sampling, 9))
            hit = psf_fft_cache.get(key)
            if hit is not None:
                return hit
        fft = psf_to_fft(psf_arr, psf_sampling=psf_sampling,
                         target_shape=(target_h, target_w),
                         target_sampling=target_sampling)
        if psf_fft_cache is not None:
            psf_fft_cache[key] = fft
        return fft

    unique_fft = {}
    for v in views:
        pid = id(v["psf"])
        if pid not in unique_fft:
            unique_fft[pid] = _fft_for(v["psf"])
    if len(unique_fft) == 1:
        one = next(iter(unique_fft.values()))
        psf_fft_stack = jnp.broadcast_to(one[None], (n_views,) + one.shape)
    else:
        psf_fft_stack = jnp.stack([unique_fft[id(v["psf"])] for v in views])

    images_data = {
        "data": jnp.stack(data_list),
        "invvar": jnp.stack(invvar_list),
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

    if max_ps > 0 or max_gal > 0:
        init = np.stack(init_flux_list)
    else:
        init = np.zeros((n_views, n_flux), dtype=dtype)
    initial_fluxes = jnp.asarray(init, dtype=dtype)

    meta = {"max_ps": max_ps, "max_gal": max_gal, "max_mog_k": max_mog_k,
            "n_flux": n_flux, "bg_idx": bg_idx,
            "src_slot": src_slot_per_view, "counts": counts}
    return BatchBundle(images_data, batches, initial_fluxes,
                       batches_in_axes(batches), meta)
