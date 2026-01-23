# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import io
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from wigglystuff import ChartPuck
    return ChartPuck, mo, multivariate_normal, np, plt


@app.cell
def _(mo):
    # Script mode detection
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gaussian Distribution Overlap Explorer

    Drag the three pucks to adjust the means of three 2D Gaussian distributions.
    The right chart shows where all three distributions overlap (product of PDFs).
    """)
    return


@app.cell
def _(mo):
    # Fixed covariance matrix (same for all Gaussians)
    # Using a 2x2 identity matrix scaled by sigma^2
    sigma_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        value=0.8,
        step=0.1,
        label="Sigma (std dev)"
    )
    sigma_slider
    return (sigma_slider,)


@app.cell(hide_code=True)
def _(ChartPuck, mo, multivariate_normal, np, sigma_slider):
    # Create ChartPuck with callback to show all three Gaussian distributions
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)

    # Initial puck positions
    initial_x = [-2.0, 0.0, 2.0]
    initial_y = [-1.0, 2.0, -1.0]

    # Container to store widget reference for draw function
    widget_ref = [None]


    def draw_gaussians(ax, x, y):
        """Draw all three Gaussian distributions on the axes."""
        # Access all puck positions from the widget, or use initial values
        puck_widget = widget_ref[0]
        if puck_widget is not None:
            puck_x = puck_widget.x
            puck_y = puck_widget.y
        else:
            # Use initial values if widget not yet available
            puck_x = initial_x
            puck_y = initial_y

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Gaussian Distributions (Drag Pucks to Move)")

        # Create grid for contour plot
        x_range = np.linspace(x_bounds[0], x_bounds[1], 150)
        y_range = np.linspace(y_bounds[0], y_bounds[1], 150)
        X, Y = np.meshgrid(x_range, y_range)
        pos = np.dstack((X, Y))

        # Get sigma and all puck positions
        sigma = sigma_slider.value
        cov = np.eye(2) * (sigma**2)

        # Create and plot each Gaussian
        colors = ["#e63946", "#457b9d", "#2a9d8f"]
        cmaps = ["Reds", "Blues", "Greens"]
        for i, (mu_x, mu_y, color, cmap) in enumerate(
            zip(puck_x, puck_y, colors, cmaps)
        ):
            dist = multivariate_normal(mean=[mu_x, mu_y], cov=cov)
            pdf = dist.pdf(pos)
            pdf_norm = pdf / pdf.max()
            ax.contourf(X, Y, pdf_norm, levels=30, cmap=cmap, alpha=0.4)
            ax.contour(
                X, Y, pdf_norm, levels=8, colors=color, alpha=0.6, linewidths=1.5
            )


    # Create puck with callback
    puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_gaussians,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            figsize=(6, 6),
            x=initial_x,
            y=initial_y,
            puck_radius=12,
            puck_color="#e63946",
        )
    )

    # Store widget reference for draw function
    widget_ref[0] = puck
    return (puck,)


@app.cell
def _(fig_overlap, mo, puck):
    # Display puck widget and overlap chart side by side
    mo.hstack([puck, fig_overlap], justify="start")
    return


@app.cell
def _(np, puck, sigma_slider):
    # Extract puck positions as means
    means = np.array(list(zip(puck.x, puck.y)))  # Shape: (3, 2)

    # Create covariance matrix (isotropic, same for all)
    sigma = sigma_slider.value
    cov = np.eye(2) * (sigma ** 2)
    return cov, means


@app.cell
def _(cov, means, multivariate_normal):
    # Create 3 multivariate normal distributions
    dists = [multivariate_normal(mean=mu, cov=cov) for mu in means]
    return


@app.cell
def _(means, np, plt, sigma_slider, t):
    # Create visualization of overlap - compute log densities numerically
    # Loop over xy coordinates and sum log densities from all Gaussians

    # Grid granularity
    granularity = 200
    x_range = np.linspace(-5, 5, granularity)
    y_range = np.linspace(-5, 5, granularity)

    _sigma_overlap = sigma_slider.value
    t
    # Pre-compute log normalization constant: log(1 / (2π * σ²)) = -log(2π * σ²)
    log_norm = -np.log(2 * np.pi * _sigma_overlap**2)

    # Initialize output array
    overlap_log = np.zeros((granularity, granularity))

    # Loop over all xy coordinates
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            # Sum log densities from all three Gaussians
            log_density_sum = 0.0
            for mu in means:
                # Compute squared distance from mean
                dx = x - mu[0]
                dy = y - mu[1]
                squared_dist = dx**2 + dy**2

                # Compute log density: log_norm - 0.5 * squared_dist / sigma^2
                log_density = log_norm - 0.5 * squared_dist / (_sigma_overlap**2)
                log_density_sum += log_density

            overlap_log[i, j] = log_density_sum

    # Exponentiate to get the product of PDFs
    overlap = np.exp(overlap_log)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_range, y_range)

    # Normalize for better visualization
    overlap_normalized = overlap / overlap.max()

    # Match the puck chart size (6x6 inches) - same as ChartPuck figsize
    fig_overlap, ax_overlap = plt.subplots(figsize=(6, 6), dpi=100)
    ax_overlap.clear()  # Ensure clean slate

    # Show xy values as a heatmap instead of contours
    im = ax_overlap.imshow(
        overlap_normalized,
        extent=[-5, 5, -5, 5],
        origin='lower',
        cmap='viridis',
        aspect='equal',
        interpolation='bilinear'
    )
    ax_overlap.set_xlabel('X')
    ax_overlap.set_ylabel('Y')
    ax_overlap.set_title('Overlap Region (Sum of Log Densities)')
    plt.colorbar(im, ax=ax_overlap, label='Normalized Overlap')
    plt.tight_layout()
    return (fig_overlap,)


@app.cell
def _(mo, puck):
    # Display puck coordinates
    mo.hstack([
        mo.md("**Puck Positions:**"),
        mo.md(f"Puck 1: ({puck.x[0]:.2f}, {puck.y[0]:.2f})"),
        mo.md(f"Puck 2: ({puck.x[1]:.2f}, {puck.y[1]:.2f})"),
        mo.md(f"Puck 3: ({puck.x[2]:.2f}, {puck.y[2]:.2f})"),
    ], justify="start")
    return


if __name__ == "__main__":
    app.run()
