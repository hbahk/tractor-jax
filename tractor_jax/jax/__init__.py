try:
    from . import tree
    tree.register_pytree_nodes()
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to register JAX PyTree nodes: {e}")

from .optimizer import (
    optimize_fluxes,
    extract_model_data, extract_model_data_direct,
    solve_fluxes_linear, solve_fluxes_eigfloor,
    solve_fluxes_lasso, solve_fluxes_lasso_batched,
    lasso_fista, lasso_fista_jit,
)
from .batching import (
    BatchBundle, build_padded_batches, psf_to_fft, slice_fluxes,
    batches_in_axes, make_batched_solver, clear_solver_cache,
    penalty_weights_from_slots, pad_normal_eq,
)
from .pipeline import prefetch_pipeline, lagged_collect
from .rendering import (
    render_pixelized_psf, render_galaxy_fft, render_point_source_pixelized,
    render_galaxy_mog, render_point_source_mog
)

__all__ = [
    "optimize_fluxes",
    "extract_model_data", "extract_model_data_direct",
    "solve_fluxes_linear", "solve_fluxes_eigfloor",
    "solve_fluxes_lasso", "solve_fluxes_lasso_batched",
    "lasso_fista", "lasso_fista_jit",
    "BatchBundle", "build_padded_batches", "psf_to_fft", "slice_fluxes",
    "batches_in_axes", "make_batched_solver", "clear_solver_cache",
    "penalty_weights_from_slots", "pad_normal_eq",
    "prefetch_pipeline", "lagged_collect",
    "render_pixelized_psf", "render_galaxy_fft",
    "render_point_source_pixelized", "render_galaxy_mog",
    "render_point_source_mog",
]
