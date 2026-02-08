
import jax
import jax.numpy as jnp
import numpy as np

def test_scatter():
    H, W = 10, 10
    grid = jnp.zeros((H, W), dtype=jnp.float32)

    # 2 points to add
    # Point 1 at (2, 2)
    # Point 2 at (5, 5)

    indices = jnp.array([[2, 2], [5, 5]], dtype=jnp.int32)
    updates = jnp.array([1.0, 2.0], dtype=jnp.float32)

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1)
    )

    out = jax.lax.scatter_add(grid, indices, updates, dnums)

    print("Out sum:", out.sum())
    print("Out[2,2]:", out[2,2])
    print("Out[5,5]:", out[5,5])

    # Test flat indices logic
    N = 2
    Sh, Sw = 2, 2

    # Source 1 at (2,2). Stamp 2x2.
    # Pixels: (2,2), (2,3), (3,2), (3,3)
    # Values: 10, 10, 10, 10

    y0 = jnp.array([2, 5])
    x0 = jnp.array([2, 5])

    sy, sx = jnp.meshgrid(jnp.arange(Sh), jnp.arange(Sw), indexing='ij')

    base_y = y0[:, None, None]
    base_x = x0[:, None, None]

    flat_y = (base_y + sy[None, :, :]).reshape(-1)
    flat_x = (base_x + sx[None, :, :]).reshape(-1)

    flat_indices = jnp.stack([flat_y, flat_x], axis=-1)

    # Updates: (N, Sh, Sw) -> (N*Sh*Sw)
    stamps = jnp.ones((N, Sh, Sw)) * 10.0
    flat_updates = stamps.reshape(-1)

    grid2 = jnp.zeros((H, W), dtype=jnp.float32)

    out2 = jax.lax.scatter_add(grid2, flat_indices, flat_updates, dnums)

    print("Out2 sum:", out2.sum()) # Should be 2 * 4 * 10 = 80
    print(out2)

if __name__ == "__main__":
    test_scatter()
