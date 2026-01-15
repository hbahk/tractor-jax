try:
    from . import tree
    tree.register_pytree_nodes()
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to register JAX PyTree nodes: {e}")

from .optimizer import optimize_fluxes
from .rendering import (
    render_pixelized_psf, render_galaxy_fft, render_point_source_pixelized,
    render_galaxy_mog, render_point_source_mog
)
