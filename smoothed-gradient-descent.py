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
    s_vals = np.linspace(0.01, 3.0, 40)
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
    ## Gradient Descent in Smoothed Space

    Key insight:
    - When $s \approx 0$: $g(x, s) \approx f(x)$ (original function)
    - When $s \gg 0$: $g(x, s)$ is smooth, gradients point toward global optimum region
    - Starting at high $s$ and descending toward $s \approx 0$ helps escape local optima
    """)
    return


@app.cell
def _(np, smoothed_value):
    def gradient_descent_smoothed(f, x0, s0, lr_x=0.1, lr_s=0.05, steps=100, eps=1e-4):
        """
        Gradient descent on the smoothed landscape.
        Maximizes in x, decreases s toward 0.
        """
        trajectory = [(x0, s0)]
        x, s = x0, s0

        for _ in range(steps):
            # Numerical gradient in x (for maximization)
            g_plus = smoothed_value(f, x + eps, s)
            g_minus = smoothed_value(f, x - eps, s)
            grad_x = (g_plus - g_minus) / (2 * eps)

            # Update x (gradient ascent to find maximum)
            x = x + lr_x * grad_x

            # Decrease smoothing toward 0
            s = max(0.01, s - lr_s * s)

            trajectory.append((x, s))

        return np.array(trajectory)
    return (gradient_descent_smoothed,)


@app.cell
def _(gradient_descent_smoothed, hard_func, sinc_func):
    # Compute trajectories from different starting points
    traj_sinc_1 = gradient_descent_smoothed(sinc_func, x0=-8.0, s0=0, steps=80)
    traj_sinc_2 = gradient_descent_smoothed(sinc_func, x0=3.0, s0=0, steps=80)

    traj_hard_1 = gradient_descent_smoothed(hard_func, x0=-3.5, s0=2.5, steps=80)
    traj_hard_2 = gradient_descent_smoothed(hard_func, x0=2.5, s0=2.5, steps=80)
    return traj_hard_1, traj_hard_2, traj_sinc_1, traj_sinc_2


@app.cell
def _(
    Z_hard,
    Z_sinc,
    np,
    plt,
    s_vals,
    traj_hard_1,
    traj_hard_2,
    traj_sinc_1,
    traj_sinc_2,
    x_vals,
):
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))

    X2, S2 = np.meshgrid(x_vals, s_vals)

    # Top row: sinc function
    axes[0, 0].contourf(X2, S2, Z_sinc, levels=20, cmap='viridis')
    axes[0, 0].plot(traj_sinc_1[:, 0], traj_sinc_1[:, 1], 'r.-', linewidth=1.5, markersize=3)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('smoothing')
    axes[0, 0].set_title('sinc: start x=-4')

    axes[0, 1].contourf(X2, S2, Z_sinc, levels=20, cmap='viridis')
    axes[0, 1].plot(traj_sinc_2[:, 0], traj_sinc_2[:, 1], 'r.-', linewidth=1.5, markersize=3)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('smoothing')
    axes[0, 1].set_title('sinc: start x=3')

    # Bottom row: hard function
    axes[1, 0].contourf(X2, S2, Z_hard, levels=20, cmap='viridis')
    axes[1, 0].plot(traj_hard_1[:, 0], traj_hard_1[:, 1], 'r.-', linewidth=1.5, markersize=3)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('smoothing')
    axes[1, 0].set_title('floor: start x=-3.5')

    axes[1, 1].contourf(X2, S2, Z_hard, levels=20, cmap='viridis')
    axes[1, 1].plot(traj_hard_2[:, 0], traj_hard_2[:, 1], 'r.-', linewidth=1.5, markersize=3)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('smoothing')
    axes[1, 1].set_title('floor: start x=2.5')

    plt.tight_layout()
    fig3
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
    X_grid, S_grid = np.meshgrid(x_vals, s_vals)
    ax_heat.contourf(X_grid, S_grid, Z_hard, levels=20, cmap='viridis')
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
