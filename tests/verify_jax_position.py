import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt
import numpy as np
from tractor.jax.rendering import render_point_source_fft

def create_gaussian_psf(shape, sigma):
    """Creates a simple Gaussian PSF kernel."""
    H, W = shape
    y, x = np.mgrid[:H, :W]
    cy, cx = H // 2, W // 2
    # Normalize to sum to 1
    psf = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    psf /= psf.sum()
    return jnp.array(psf)

def run_test():
    print("Setting up JAX Position Optimization Test...")

    # Parameters
    H, W = 32, 32
    sigma_psf = 1.5
    flux_true = 1000.0
    pos_true = jnp.array([15.5, 15.5]) # Sub-pixel offset

    # Model start
    flux_model = 1000.0 # Assume flux known/fixed for this test
    pos_init = jnp.array([14.5, 14.5]) # Start closer to avoid local minima/flat regions if any
    # (Gaussian has wide basin so 13.0 should be fine too, but let's be safe with LR)

    # Create PSF
    psf_img = create_gaussian_psf((H, W), sigma_psf)

    # Shift PSF to (0,0) for FFT
    psf_shifted = jnp.fft.ifftshift(psf_img)
    psf_fft = jfft.rfft2(psf_shifted)

    # Generate True Image (Data)
    print(f"Generating True Image at {pos_true}...")
    true_image = render_point_source_fft(flux_true, pos_true, psf_fft, (H, W))

    data = true_image

    # Define Loss Function
    def loss_fn(pos):
        model = render_point_source_fft(flux_model, pos, psf_fft, (H, W))
        diff = data - model
        # Simple Sum of Squared Errors
        return 0.5 * jnp.sum(diff**2) # 0.5 factor standard

    # Optimization Loop
    # Gradient is roughly Flux^2 * something. 1000^2 = 1e6.
    # Gradient was ~1e4.
    # To move ~0.1 pixel, we need lr * 1e4 = 0.1 => lr = 1e-5.
    lr = 1e-5
    n_steps = 100

    pos_current = pos_init

    print(f"Starting optimization from {pos_init}...")
    print("Step | Loss       | Position        | Gradient")

    # JIT compile value_and_grad
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for i in range(n_steps):
        val, grads = loss_and_grad(pos_current)

        if i % 10 == 0:
            p_str = f"[{pos_current[0]:.4f}, {pos_current[1]:.4f}]"
            g_str = f"[{grads[0]:.4f}, {grads[1]:.4f}]"
            print(f"{i:4d} | {val:.4e} | {p_str} | {g_str}")

        pos_current = pos_current - lr * grads

    final_pos = pos_current
    final_loss = loss_fn(final_pos)
    print(f"Final: {final_pos}, Loss: {final_loss:.4e}")
    print(f"Target: {pos_true}")

    # Generate Final Images
    model_init = render_point_source_fft(flux_model, pos_init, psf_fft, (H, W))
    model_final = render_point_source_fft(flux_model, final_pos, psf_fft, (H, W))
    residual = data - model_final

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    vmin, vmax = data.min(), data.max()

    im0 = axes[0].imshow(np.array(data), origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Data (True Pos: {pos_true})")

    im1 = axes[1].imshow(np.array(model_init), origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Initial Model (Pos: {pos_init})")

    im2 = axes[2].imshow(np.array(model_final), origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Final Model (Pos: [{final_pos[0]:.2f}, {final_pos[1]:.2f}])")

    im3 = axes[3].imshow(np.array(residual), origin='lower', vmin=-vmax/10, vmax=vmax/10, cmap='bwr')
    axes[3].set_title(f"Residual (MSE: {final_loss:.2e})")

    plt.colorbar(im3, ax=axes[3])

    outfile = "jax_position_optimization_test.png"
    plt.savefig(outfile)
    print(f"Saved visualization to {outfile}")

    # Verification
    dist = jnp.linalg.norm(final_pos - pos_true)
    if dist < 0.05:
        print(f"SUCCESS: Position converged to {final_pos} (within tolerance).")
    else:
        print(f"FAILURE: Position did not converge. Distance: {dist}")

if __name__ == "__main__":
    run_test()
