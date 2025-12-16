# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "scipy==1.16.3",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.integrate import quad
    return mo, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem of Hard Functions

    Sometimes we want to optimise functions that are hard to optimise. Here are two examples:

    $$f(x) = \text{sinc}(x) \quad \text{and} \quad f(x) = \lfloor 10 \cdot \text{sinc}(x) + 4 \sin(x) \rfloor$$
    """)
    return


@app.cell
def _(np):
    # Shared x values for all plots
    x_vals = np.linspace(-15, 15, 1000)
    s_vals = np.linspace(0.01, 5.0, 40)
    return s_vals, x_vals


@app.cell
def _(np):
    def sinc_func(x):
        """Standard sinc function: sin(x)/x with sinc(0)=1"""
        return np.sinc(x / np.pi)

    def hard_func(x):
        """Non-differentiable function with many local optima"""
        return np.floor(10 * np.sinc(x / np.pi) + 4 * np.sin(x))
    return hard_func, sinc_func


@app.cell
def _(hard_func, plt, sinc_func, x_vals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x_vals, sinc_func(x_vals), color='#1f77b4')
    ax1.set_xlabel('x')
    ax1.set_title(r'$f(x) = \mathrm{sinc}(x)$')
    ax1.set_ylim(-0.3, 1.1)

    ax2.plot(x_vals, hard_func(x_vals), color='#1f77b4')
    ax2.set_xlabel('x')
    ax2.set_title(r'$f(x) = \lfloor 10 \cdot \mathrm{sinc}(x) + 4 \sin(x) \rfloor$')

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both functions have multiple peaks, making them hard to optimise via gradient descent. The right function is also non-differentiable.

    **Idea:** Add a smoothing parameter $s$ to turn the 1D problem into a 2D problem:

    $$g(x, s) = \int_{-\infty}^{\infty} f(t) \cdot \mathcal{N}(t; \mu=x, \sigma=s) \, dt$$

    When $s = 0$: $g(x, 0) = f(x)$. When $s \gg 0$: $g(x, s)$ becomes a smooth average.
    """)
    return


@app.cell
def _(norm, np):
    def smoothed_value(f, x, sigma, n_points=500):
        """Compute Gaussian-smoothed value of f at x with smoothing sigma."""
        if sigma < 0.01:
            return f(x)
        # Integrate over ±4 sigma
        t = np.linspace(x - 4*sigma, x + 4*sigma, n_points)
        weights = norm.pdf(t, loc=x, scale=sigma)
        values = f(t)
        return np.trapezoid(values * weights, t)

    def compute_landscape(f, x_arr, s_arr):
        """Compute the smoothed landscape g(x, s) over a grid."""
        Z = np.zeros((len(s_arr), len(x_arr)))
        for i, s in enumerate(s_arr):
            for j, x in enumerate(x_arr):
                Z[i, j] = smoothed_value(f, x, s)
        return Z
    return compute_landscape, smoothed_value


@app.cell
def _(compute_landscape, hard_func, s_vals, sinc_func, x_vals):
    # Compute landscapes (this may take a moment)
    Z_sinc = compute_landscape(sinc_func, x_vals, s_vals)
    Z_hard = compute_landscape(hard_func, x_vals, s_vals)
    return Z_hard, Z_sinc


@app.cell
def _(Z_hard, Z_sinc, np, plt, s_vals, x_vals):
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))

    X, S = np.meshgrid(x_vals, s_vals)

    cf1 = ax3.contourf(X, S, Z_sinc, levels=20, cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('smoothing')
    ax3.set_title(r'$g(x, s)$ for sinc')
    plt.colorbar(cf1, ax=ax3)

    cf2 = ax4.contourf(X, S, Z_hard, levels=20, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('smoothing')
    ax4.set_title(r'$g(x, s)$ for floor function')
    plt.colorbar(cf2, ax=ax4)

    plt.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient Field Visualization

    The arrows show the gradient direction at each point. Notice how at high smoothing (top),
    gradients consistently point toward the global maximum at $x=0$. At low smoothing (bottom),
    the gradients become chaotic with many local optima.
    """)
    return


@app.cell
def _(Z_sinc, np, plt, s_vals, sinc_func, smoothed_value, x_vals):
    # Compute gradient field for visualization
    # Use a coarser grid for the arrows, derived from x_vals and s_vals
    x_arrow = np.linspace(x_vals.min() + 1, x_vals.max() - 1, 20)
    s_arrow = np.linspace(s_vals.min() + 0.1, s_vals.max() - 0.1, 15)

    U = np.zeros((len(s_arrow), len(x_arrow)))  # gradient in x direction
    V = np.zeros((len(s_arrow), len(x_arrow)))  # gradient in s direction
    eps = 1e-4

    for i, s in enumerate(s_arrow):
        for j, x in enumerate(x_arrow):
            # Gradient in x
            g_x_plus = smoothed_value(sinc_func, x + eps, s)
            g_x_minus = smoothed_value(sinc_func, x - eps, s)
            U[i, j] = (g_x_plus - g_x_minus) / (2 * eps)

            # Gradient in s - this shows how the landscape wants to move in s direction
            g_s_plus = smoothed_value(sinc_func, x, s + eps)
            g_s_minus = smoothed_value(sinc_func, x, max(0.01, s - eps))
            V[i, j] = (g_s_plus - g_s_minus) / (2 * eps)

    fig_quiver, ax_quiver = plt.subplots(figsize=(10, 6))

    _X_grid, _S_grid = np.meshgrid(x_vals, s_vals)
    ax_quiver.contourf(_X_grid, _S_grid, Z_sinc, levels=20, cmap='viridis', alpha=0.8)

    _X_arrow, _S_arrow = np.meshgrid(x_arrow, s_arrow)
    # Normalize for display - show direction only
    magnitude = np.sqrt(U**2 + V**2 + 0.001)
    ax_quiver.quiver(_X_arrow, _S_arrow, U/magnitude, V/magnitude, color='white', alpha=0.9, scale=30)

    ax_quiver.set_xlabel('x')
    ax_quiver.set_ylabel('smoothing (σ)')
    ax_quiver.set_title('Gradient field: arrows show steepest ascent direction in (x, σ) space')
    plt.colorbar(ax_quiver.collections[0], ax=ax_quiver)

    fig_quiver
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient Descent in Smoothed Space

    Key insight:
    - When $s \approx 0$: $g(x, s) \approx f(x)$ (original function)
    - When $s \gg 0$: $g(x, s)$ is smooth, gradients point toward global optimum region
    - Starting at high $s$ and descending toward $s \approx 0$ helps escape local optima
    """)
    return


@app.cell(hide_code=True)
def _(np, smoothed_value):
    def gradient_descent_smoothed(f, x0, s0, lr_x=0.5, lr_s=0.1, steps=100, eps=1e-4, s_bias=-0.01):
        """
        Gradient descent on the smoothed landscape g(x, s).
        Follows gradients in BOTH x and s directions.

        The key insight: s is a searchable dimension, not just an annealing schedule.
        We maximize g in x direction, and follow gradient in s direction with a small
        bias toward lower s (to eventually converge to the true optimum).
        """
        trajectory = [(x0, s0)]
        x, s = x0, s0

        for step in range(steps):
            # Numerical gradient in x (for maximization)
            g_x_plus = smoothed_value(f, x + eps, s)
            g_x_minus = smoothed_value(f, x - eps, s)
            grad_x = (g_x_plus - g_x_minus) / (2 * eps)

            # Numerical gradient in s - this is the key!
            # Follow the gradient in s direction too
            g_s_plus = smoothed_value(f, x, s + eps)
            g_s_minus = smoothed_value(f, x, max(0.01, s - eps))
            grad_s = (g_s_plus - g_s_minus) / (2 * eps)

            # Update x (gradient ascent to find maximum)
            x = x + lr_x * grad_x

            # Update s: follow gradient + small bias toward lower s
            # The bias ensures we eventually converge to s≈0
            s = s + lr_s * grad_s + s_bias
            s = max(0.01, min(3.0, s))  # clamp to valid range

            trajectory.append((x, s))

        return np.array(trajectory)
    return (gradient_descent_smoothed,)


@app.cell
def _(gradient_descent_smoothed, hard_func, sinc_func):
    # Compute gradient-based trajectories
    traj_grad_sinc_1 = gradient_descent_smoothed(sinc_func, x0=-8.0, s0=3.0, steps=200)
    traj_grad_sinc_2 = gradient_descent_smoothed(sinc_func, x0=6.0, s0=3.0, steps=200)

    traj_grad_hard_1 = gradient_descent_smoothed(hard_func, x0=-8.0, s0=3.0, steps=200)
    traj_grad_hard_2 = gradient_descent_smoothed(hard_func, x0=6.0, s0=3.0, steps=200)
    return (
        traj_grad_hard_1,
        traj_grad_hard_2,
        traj_grad_sinc_1,
        traj_grad_sinc_2,
    )


@app.cell(hide_code=True)
def _(np):
    def sample_based_optimize(f, mu0, sigma0, alpha_mu=0.5, alpha_sigma=0.8, n_samples=50, steps=100, seed=42):
        """
        Evolution strategies style optimization in 2D (x, σ) space.

        Instead of computing gradients numerically, we sample from N(μ, σ)
        and use the samples to estimate how to update both μ and σ.

        Key insight: samples that land in high-f(x) regions "pull" μ toward them.
        This allows escaping local optima through stochastic exploration.
        """
        np.random.seed(seed)
        trajectory = [(mu0, sigma0)]
        mu, sigma = mu0, sigma0

        for _ in range(steps):
            # Sample from current distribution
            samples = np.random.normal(mu, sigma, n_samples)
            f_vals = f(samples)

            # Score function gradient for mu: E[f(x) * (x - mu) / sigma^2]
            d_mu = alpha_mu * np.mean(f_vals * (samples - mu)) / (sigma**2)

            # Score function gradient for sigma: E[f(x) * ((x-mu)^2/sigma^3 - 1/sigma)] / 2
            d_sigma = alpha_sigma * np.mean(f_vals * ((samples - mu)**2 / sigma**3 - 1/sigma)) / 2

            # Update parameters
            mu = mu + d_mu
            sigma = np.clip(sigma + d_sigma, 0.05, 10.0)

            trajectory.append((mu, sigma))

        return np.array(trajectory)
    return (sample_based_optimize,)


@app.cell
def _(hard_func, sample_based_optimize, sinc_func):
    # Compute sample-based trajectories (same starting points)
    traj_sample_sinc_1 = sample_based_optimize(sinc_func, mu0=-8.0, sigma0=3.0, steps=500, seed=42)
    traj_sample_sinc_2 = sample_based_optimize(sinc_func, mu0=6.0, sigma0=3.0, steps=200, seed=43)

    traj_sample_hard_1 = sample_based_optimize(hard_func, mu0=-8.0, sigma0=3.0, steps=200, seed=44)
    traj_sample_hard_2 = sample_based_optimize(hard_func, mu0=6.0, sigma0=3.0, steps=200, seed=45)
    return (
        traj_sample_hard_1,
        traj_sample_hard_2,
        traj_sample_sinc_1,
        traj_sample_sinc_2,
    )


@app.cell(hide_code=True)
def _(
    Z_hard,
    Z_sinc,
    np,
    plt,
    s_vals,
    traj_grad_hard_1,
    traj_grad_hard_2,
    traj_grad_sinc_1,
    traj_grad_sinc_2,
    traj_sample_hard_1,
    traj_sample_hard_2,
    traj_sample_sinc_1,
    traj_sample_sinc_2,
    x_vals,
):
    # Both methods overlaid on same chart for direct comparison
    fig_compare, axes = plt.subplots(2, 2, figsize=(12, 10))

    _X2, _S2 = np.meshgrid(x_vals, s_vals)

    # Top row: sinc function
    # Left: start x=-8
    axes[0, 0].contourf(_X2, _S2, Z_sinc, levels=20, cmap='viridis')
    axes[0, 0].plot(traj_grad_sinc_1[:, 0], traj_grad_sinc_1[:, 1], 'r.-', linewidth=2, markersize=3, label='Gradient')
    axes[0, 0].plot(traj_sample_sinc_1[:, 0], traj_sample_sinc_1[:, 1], 'w.-', linewidth=2, markersize=3, label='Sample')
    axes[0, 0].scatter([traj_grad_sinc_1[0, 0]], [traj_grad_sinc_1[0, 1]], s=100, c='yellow', marker='*', zorder=10, label='Start')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('smoothing (σ)')
    axes[0, 0].set_title('sinc: start x=-8')
    axes[0, 0].legend(loc='upper right', fontsize=8)

    # Right: start x=6
    axes[0, 1].contourf(_X2, _S2, Z_sinc, levels=20, cmap='viridis')
    axes[0, 1].plot(traj_grad_sinc_2[:, 0], traj_grad_sinc_2[:, 1], 'r.-', linewidth=2, markersize=3, label='Gradient')
    axes[0, 1].plot(traj_sample_sinc_2[:, 0], traj_sample_sinc_2[:, 1], 'w.-', linewidth=2, markersize=3, label='Sample')
    axes[0, 1].scatter([traj_grad_sinc_2[0, 0]], [traj_grad_sinc_2[0, 1]], s=100, c='yellow', marker='*', zorder=10, label='Start')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('smoothing (σ)')
    axes[0, 1].set_title('sinc: start x=6')
    axes[0, 1].legend(loc='upper right', fontsize=8)

    # Bottom row: hard (floor) function
    # Left: start x=-8
    axes[1, 0].contourf(_X2, _S2, Z_hard, levels=20, cmap='viridis')
    axes[1, 0].plot(traj_grad_hard_1[:, 0], traj_grad_hard_1[:, 1], 'r.-', linewidth=2, markersize=3, label='Gradient')
    axes[1, 0].plot(traj_sample_hard_1[:, 0], traj_sample_hard_1[:, 1], 'w.-', linewidth=2, markersize=3, label='Sample')
    axes[1, 0].scatter([traj_grad_hard_1[0, 0]], [traj_grad_hard_1[0, 1]], s=100, c='yellow', marker='*', zorder=10, label='Start')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('smoothing (σ)')
    axes[1, 0].set_title('floor: start x=-8')
    axes[1, 0].legend(loc='upper right', fontsize=8)

    # Right: start x=6
    axes[1, 1].contourf(_X2, _S2, Z_hard, levels=20, cmap='viridis')
    axes[1, 1].plot(traj_grad_hard_2[:, 0], traj_grad_hard_2[:, 1], 'r.-', linewidth=2, markersize=3, label='Gradient')
    axes[1, 1].plot(traj_sample_hard_2[:, 0], traj_sample_hard_2[:, 1], 'w.-', linewidth=2, markersize=3, label='Sample')
    axes[1, 1].scatter([traj_grad_hard_2[0, 0]], [traj_grad_hard_2[0, 1]], s=100, c='yellow', marker='*', zorder=10, label='Start')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('smoothing (σ)')
    axes[1, 1].set_title('floor: start x=6')
    axes[1, 1].legend(loc='upper right', fontsize=8)

    fig_compare.suptitle('Gradient (red) vs Sample-based (white) optimization', fontsize=14)
    plt.tight_layout()
    fig_compare
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intuition

    How does the smoothing work? We convolve $f(x)$ with a Gaussian:

    $$g(x, \sigma) = \int_{-\infty}^{\infty} f(t) \cdot \mathcal{N}(t; x, \sigma) \, dt$$

    This integral computes a weighted average of $f$, where points closer to $x$ get higher weight.

    **Use the sliders below to explore how different $(x, \sigma)$ values produce different pixel values.**
    """)
    return


@app.cell
def _(mo):
    x_slider = mo.ui.slider(start=-14.0, stop=14.0, step=0.1, value=0.0, label="x")
    sigma_slider = mo.ui.slider(start=0.1, stop=3.0, step=0.1, value=1.0, label="σ")
    mo.hstack([x_slider, sigma_slider], justify="start")
    return sigma_slider, x_slider


@app.cell(hide_code=True)
def _(
    Z_hard,
    hard_func,
    mo,
    norm,
    np,
    plt,
    s_vals,
    sigma_slider,
    smoothed_value,
    x_slider,
    x_vals,
):
    x_pos = x_slider.value
    sigma_val = sigma_slider.value

    fig_int, (ax_conv, ax_heat) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Function, Gaussian, product, filled area
    ax_conv.plot(x_vals, hard_func(x_vals), 'C0-', linewidth=1.5, label=r'$f(x)$')

    gaussian = norm.pdf(x_vals, loc=x_pos, scale=sigma_val)
    scale = 5
    ax_conv.plot(x_vals, gaussian * scale, 'C1-', linewidth=1.5,
                 label=rf'$\mathcal{{N}}(\mu={x_pos:.1f}, \sigma={sigma_val:.1f})$')

    product = hard_func(x_vals) * gaussian
    ax_conv.plot(x_vals, product * scale, 'C2-', linewidth=2, label=r'$f(x) \cdot \mathcal{N}$')
    ax_conv.fill_between(x_vals, 0, product * scale, alpha=0.3, color='C2')
    ax_conv.axvline(x=x_pos, color='red', linestyle='--', alpha=0.5)

    ax_conv.set_xlabel('x')
    ax_conv.set_ylim(-8, 12)
    ax_conv.set_xlim(-15, 15)
    ax_conv.legend(loc='upper right', fontsize=9)
    ax_conv.set_title('Gaussian convolution')

    # Right: Heatmap with pixel
    _X_grid, _S_grid = np.meshgrid(x_vals, s_vals)
    ax_heat.contourf(_X_grid, _S_grid, Z_hard, levels=20, cmap='viridis')
    ax_heat.scatter([x_pos], [sigma_val], s=200, c='white', marker='o',
                    edgecolors='red', linewidths=3, zorder=5)

    pixel_val = smoothed_value(hard_func, x_pos, sigma_val)
    ax_heat.set_xlabel('x')
    ax_heat.set_ylabel('smoothing (σ)')
    ax_heat.set_title(f'Pixel value = {pixel_val:.2f}')
    plt.colorbar(ax_heat.collections[0], ax=ax_heat)

    plt.tight_layout()

    mo.vstack([
        fig_int,
        mo.md(f"The green shaded area on the left (integral) equals the pixel value **{pixel_val:.2f}** on the right.")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key insight: instead of optimising $f(x)$ directly, we optimise in the $(x, s)$ space.
    Starting with high smoothing, the landscape is smooth and gradients point toward the global optimum.
    As we reduce $s$ toward 0, we converge to the true optimum of $f(x)$.
    """)
    return


if __name__ == "__main__":
    app.run()
