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
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from tractor_jax.jax.optimizer import (
    solve_fluxes_linear,
    solve_fluxes_eigfloor,
    solve_fluxes_lasso,
)

__all__ = [
    "batches_in_axes",
    "make_batched_solver",
    "clear_solver_cache",
    "penalty_weights_from_slots",
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
