# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
#     "torch>=2.0",
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
    import torch
    import torch.nn as nn
    return mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diffusion Models: Learning to Reverse Noise

    Diffusion models are one of the most powerful generative models today. The core idea:

    1. **Forward process (easy):** Gradually add noise to data until it becomes pure random noise
    2. **Reverse process (the mystery):** Start from noise and somehow recover the original structure

    The forward process is trivial - just add Gaussian noise. But how do we reverse it? **How do we know which direction leads back to real data?**

    That's exactly what this notebook will teach you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameters

    Adjust these to experiment with the diffusion process.
    """)
    return


@app.cell
def _(mo):
    grid_size_slider = mo.ui.slider(
        start=2, stop=6, step=1, value=4,
        label="Grid size", show_value=True
    )
    n_samples_slider = mo.ui.slider(
        start=200, stop=2000, step=200, value=1000,
        label="Number of samples", show_value=True
    )
    mo.hstack([grid_size_slider, n_samples_slider], gap=2)
    return grid_size_slider, n_samples_slider


@app.cell
def _(grid_size_slider, n_samples_slider, np):
    def make_checkerboard(n_samples=1000, grid_size=4, seed=42):
        """
        Generate a checkerboard pattern (alternating filled squares).

        A 4x4 grid has 8 filled squares in a checkerboard pattern.
        Points are uniformly sampled from the filled squares.
        """
        np.random.seed(seed)

        # Determine which squares are "filled" (checkerboard pattern)
        filled_squares = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:  # Checkerboard condition
                    filled_squares.append((i, j))

        n_squares = len(filled_squares)
        samples_per_square = n_samples // n_squares

        points = []
        for (i, j) in filled_squares:
            # Sample uniformly within this square
            x = np.random.uniform(i, i + 1, samples_per_square)
            y = np.random.uniform(j, j + 1, samples_per_square)
            points.append(np.column_stack([x, y]))

        X = np.vstack(points)

        # Center the data around origin
        X = X - grid_size / 2

        return X

    # Generate dataset using hyperparameters
    _grid_size = grid_size_slider.value
    _n_samples = n_samples_slider.value
    data_original = make_checkerboard(n_samples=_n_samples, grid_size=_grid_size, seed=42)
    # Store the grid size for axis limits
    data_grid_size = _grid_size
    return data_grid_size, data_original, make_checkerboard


@app.cell
def _(data_original, np):
    # Pre-compute forward diffusion at multiple timesteps
    def compute_forward_trajectory(data, n_steps=10, seed=42):
        """Compute forward diffusion trajectory."""
        np.random.seed(seed)
        noise = np.random.randn(*data.shape)
        trajectory = []

        for step in range(n_steps + 1):
            t = step / n_steps
            alpha_bar = 1 - t
            data_noisy = np.sqrt(alpha_bar) * data + np.sqrt(1 - alpha_bar + 1e-8) * noise
            trajectory.append(data_noisy)

        return trajectory

    forward_trajectory = compute_forward_trajectory(data_original, n_steps=10, seed=42)
    return compute_forward_trajectory, forward_trajectory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Big Picture: Forward and Reverse

    **Forward process (left to right):** We start with structured data (a checkerboard pattern) and gradually add noise until it becomes random. This is easy - we just add Gaussian noise.

    **Reverse process (right to left):** Starting from noise, we iteratively denoise to recover structure. Watch the grid pattern emerge first, then the sharp square boundaries.

    **The key question:** How does the reverse process know which direction to go? We'll explore different methods to solve this - and ultimately train a neural network to do it.
    """)
    return


@app.cell
def _(data_grid_size, forward_trajectory, plt):
    # Show forward process: data -> noise (5 snapshots)
    _fig, _axes = plt.subplots(1, 5, figsize=(15, 3))
    _steps_to_show = [0, 2, 5, 8, 10]
    _labels = ['Start\n(data)', 't = 0.2', 't = 0.5', 't = 0.8', 'End\n(noise)']
    _lim = data_grid_size / 2 + 1.5

    for _idx, _step in enumerate(_steps_to_show):
        _ax = _axes[_idx]
        _data = forward_trajectory[_step]

        _ax.scatter(_data[:, 0], _data[:, 1], c='#3498db', alpha=0.6, s=8)
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect('equal')
        _ax.set_title(_labels[_idx])
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig.suptitle('Forward Process: Data → Noise', fontsize=14, y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(data_original, np):
    # Pre-compute reverse diffusion using KDE-based score estimation
    def estimate_score_kde(x, data, sigma=0.3):
        """Estimate score using kernel density estimation."""
        diff = data[np.newaxis, :, :] - x[:, np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        weights = np.exp(-dist_sq / (2 * sigma ** 2))
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        weighted_diff = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        score = weighted_diff / (sigma ** 2 * weights_sum)
        return score

    def compute_reverse_trajectory_kde(data, n_steps=50, n_particles=200, seed=42):
        """Compute reverse trajectory using simple Langevin dynamics."""
        np.random.seed(seed)
        x = np.random.randn(n_particles, 2) * 2  # Start from noise

        trajectory = [x.copy()]

        # Simple approach: use decreasing step sizes and KDE-based score
        for step in range(n_steps):
            # Progress from 0 to 1
            progress = step / n_steps

            # Decreasing noise level - start high, end low
            noise_level = 1.0 - progress

            # KDE bandwidth decreases as we get closer to data
            sigma_kde = 0.3 + 0.7 * noise_level

            # Estimate score (direction toward data)
            score = estimate_score_kde(x, data, sigma=sigma_kde)

            # Step size decreases over time
            step_size = 0.5 * (1.0 - 0.8 * progress)

            # Langevin update: move toward data + small noise
            noise_scale = 0.3 * noise_level
            noise = np.random.randn(*x.shape) * noise_scale

            x = x + step_size * score + noise

            trajectory.append(x.copy())

        return trajectory

    reverse_trajectory = compute_reverse_trajectory_kde(data_original, n_steps=50, n_particles=200, seed=42)
    return (
        compute_reverse_trajectory_kde,
        estimate_score_kde,
        reverse_trajectory,
    )


@app.cell
def _(data_grid_size, plt, reverse_trajectory):
    # Show reverse process: noise -> data (5 snapshots)
    _fig, _axes = plt.subplots(1, 5, figsize=(15, 3))
    _steps_to_show = [0, 12, 25, 37, 50]
    _labels = ['Start\n(noise)', 'Step 12', 'Step 25', 'Step 37', 'End\n(data)']
    _lim = data_grid_size / 2 + 1.5

    for _idx, _step in enumerate(_steps_to_show):
        _ax = _axes[_idx]
        _data = reverse_trajectory[_step]

        _ax.scatter(_data[:, 0], _data[:, 1], c='#e74c3c', alpha=0.6, s=8)
        _ax.set_xlim(-_lim, _lim)
        _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect('equal')
        _ax.set_title(_labels[_idx])
        _ax.set_xticks([])
        _ax.set_yticks([])

    _fig.suptitle('Reverse Process: Noise → Data', fontsize=14, y=1.02)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's the core idea! Now let's understand each piece.

    ---

    ## Section 1: The Forward Process

    The forward process adds Gaussian noise to data over time. Use the slider to watch structure dissolve into chaos.
    """)
    return


@app.cell
def _(mo):
    timestep_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.02, value=0.0,
        label="Noise level (t)", show_value=True
    )
    timestep_slider
    return (timestep_slider,)


@app.cell
def _(data_grid_size, data_original, np, plt, timestep_slider):
    _t = timestep_slider.value
    _alpha_bar = 1 - _t
    _lim = data_grid_size / 2 + 1.5

    np.random.seed(42)
    _noise = np.random.randn(*data_original.shape)
    _data_noisy = np.sqrt(_alpha_bar) * data_original + np.sqrt(1 - _alpha_bar + 1e-8) * _noise

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.scatter(_data_noisy[:, 0], _data_noisy[:, 1], c='#3498db', alpha=0.6, s=15)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Checkerboard at t = {_t:.2f} (signal: {_alpha_bar:.0%}, noise: {1-_alpha_bar:.0%})')
    _ax.set_aspect('equal')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At each timestep $t$, we can compute the noisy version directly:

    $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

    where:
    - $x_0$ is the original data point
    - $\epsilon \sim \mathcal{N}(0, I)$ is random Gaussian noise
    - $\bar{\alpha}_t = 1 - t$ controls how much signal vs noise we have
    """)
    return


@app.cell
def _(mo):
    forward_t_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="Timestep t", show_value=True
    )
    forward_t_slider
    return (forward_t_slider,)


@app.cell
def _(data_grid_size, data_original, forward_t_slider, np, plt):
    _t = forward_t_slider.value
    _alpha_bar = 1 - _t
    _lim = data_grid_size / 2 + 1.5

    # Use the full dataset
    np.random.seed(123)
    _x0 = data_original
    _eps = np.random.randn(*_x0.shape)
    _xt = np.sqrt(_alpha_bar) * _x0 + np.sqrt(1 - _alpha_bar + 1e-8) * _eps

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(12, 4))

    _ax1.scatter(_x0[:, 0], _x0[:, 1], c='blue', alpha=0.5, s=15)
    _ax1.set_xlim(-_lim, _lim)
    _ax1.set_ylim(-_lim, _lim)
    _ax1.set_title(r"$x_0$ (original)")
    _ax1.set_aspect('equal')

    _ax2.scatter(_eps[:, 0], _eps[:, 1], c='red', alpha=0.5, s=15)
    _ax2.set_xlim(-_lim, _lim)
    _ax2.set_ylim(-_lim, _lim)
    _ax2.set_title(r"$\epsilon$ (noise)")
    _ax2.set_aspect('equal')

    _ax3.scatter(_xt[:, 0], _xt[:, 1], c='purple', alpha=0.5, s=15)
    _ax3.set_xlim(-_lim, _lim)
    _ax3.set_ylim(-_lim, _lim)
    _ax3.set_title(rf"$x_t = \sqrt{{{_alpha_bar:.2f}}} \cdot x_0 + \sqrt{{{1-_alpha_bar:.2f}}} \cdot \epsilon$")
    _ax3.set_aspect('equal')

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 2: The Noise Schedule

    The **noise schedule** $\bar{\alpha}_t$ determines how quickly we destroy structure. Different schedules have different effects on our checkerboard!

    - **Linear schedule:** $\bar{\alpha}_t = 1 - t$ — destroys structure uniformly
    - **Cosine schedule:** Preserves more structure at early timesteps (better for images)

    Watch how the checkerboard looks at $t = 0.3$ under each schedule:
    """)
    return


@app.cell
def _(mo):
    schedule_t_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="Timestep t", show_value=True
    )
    schedule_t_slider
    return (schedule_t_slider,)


@app.cell
def _(data_grid_size, data_original, np, plt, schedule_t_slider):
    def _linear_schedule(t):
        return 1 - t

    def _cosine_schedule(t):
        s = 0.008
        f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        return f_t / f_0

    # Use slider value for comparison
    _t_compare = schedule_t_slider.value
    _alpha_linear = _linear_schedule(_t_compare)
    _alpha_cosine = _cosine_schedule(_t_compare)
    _lim = data_grid_size / 2 + 1.5

    np.random.seed(42)
    _noise = np.random.randn(*data_original.shape)
    _data_linear = np.sqrt(_alpha_linear) * data_original + np.sqrt(1 - _alpha_linear + 1e-8) * _noise
    _data_cosine = np.sqrt(_alpha_cosine) * data_original + np.sqrt(1 - _alpha_cosine + 1e-8) * _noise

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot schedule curves
    _timesteps = np.linspace(0, 1, 100)
    _axes[0].plot(_timesteps, _linear_schedule(_timesteps), 'b-', linewidth=2, label='Linear')
    _axes[0].plot(_timesteps, _cosine_schedule(_timesteps), 'r-', linewidth=2, label='Cosine')
    _axes[0].axvline(x=_t_compare, color='gray', linestyle='--', alpha=0.7)
    _axes[0].scatter([_t_compare], [_alpha_linear], c='blue', s=100, zorder=5)
    _axes[0].scatter([_t_compare], [_alpha_cosine], c='red', s=100, zorder=5)
    _axes[0].set_xlabel('Timestep t')
    _axes[0].set_ylabel(r'$\bar{\alpha}_t$ (signal retained)')
    _axes[0].set_title('Noise Schedules')
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    # Linear at current t
    _axes[1].scatter(_data_linear[:, 0], _data_linear[:, 1], c='#3498db', alpha=0.5, s=10)
    _axes[1].set_xlim(-_lim, _lim)
    _axes[1].set_ylim(-_lim, _lim)
    _axes[1].set_title(f'Linear at t={_t_compare:.2f}\n(signal: {_alpha_linear:.0%})')
    _axes[1].set_aspect('equal')

    # Cosine at current t
    _axes[2].scatter(_data_cosine[:, 0], _data_cosine[:, 1], c='#e74c3c', alpha=0.5, s=10)
    _axes[2].set_xlim(-_lim, _lim)
    _axes[2].set_ylim(-_lim, _lim)
    _axes[2].set_title(f'Cosine at t={_t_compare:.2f}\n(signal: {_alpha_cosine:.0%})')
    _axes[2].set_aspect('equal')

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 3: The Score Function

    The **score function** is the key to reversing diffusion. It answers: **"Which direction leads to more likely data?"**

    $$s(x) = \nabla_x \log p(x)$$

    Think of it as a vector field that points "uphill" toward regions of higher probability.

    ### Why do we need this?

    If we're at a noisy point and want to denoise, we need to know which direction leads back to real data. The score tells us exactly that!

    ### Score on our checkerboard

    The arrows below show the score field estimated from our checkerboard data. Notice how:
    - **Inside a square:** arrows are small (you're already in a high-probability region)
    - **Outside squares:** arrows point toward the nearest filled square
    - **Between squares:** arrows point toward one of the neighboring squares
    """)
    return


@app.cell
def _(mo):
    score_sigma_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.1, value=0.3,
        label="KDE bandwidth σ", show_value=True
    )
    score_sigma_slider
    return (score_sigma_slider,)


@app.cell
def _(data_grid_size, data_original, np, plt, score_sigma_slider):
    def _estimate_score_kde(x, data, sigma):
        """Estimate score using kernel density estimation."""
        diff = data[np.newaxis, :, :] - x[:, np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        weights = np.exp(-dist_sq / (2 * sigma ** 2))
        weights_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        weighted_diff = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        score = weighted_diff / (sigma ** 2 * weights_sum)
        return score

    _sigma = score_sigma_slider.value
    _lim = data_grid_size / 2 + 1.0

    # Create grid for score field
    _x = np.linspace(-_lim, _lim, 18)
    _y = np.linspace(-_lim, _lim, 18)
    _X, _Y = np.meshgrid(_x, _y)
    _grid_points = np.column_stack([_X.ravel(), _Y.ravel()])

    # Estimate score at each grid point
    _score = _estimate_score_kde(_grid_points, data_original, sigma=_sigma)
    _U = _score[:, 0].reshape(_X.shape)
    _V = _score[:, 1].reshape(_Y.shape)

    # Normalize for visualization
    _magnitude = np.sqrt(_U**2 + _V**2 + 0.01)
    _U_norm = _U / _magnitude
    _V_norm = _V / _magnitude

    _fig, _ax = plt.subplots(figsize=(8, 8))

    # Plot data points
    _ax.scatter(data_original[:, 0], data_original[:, 1], c='#3498db', alpha=0.3, s=10, label='Data')

    # Plot score field
    _ax.quiver(_X, _Y, _U_norm, _V_norm, color='#e74c3c', alpha=0.7, scale=25)

    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Score Field on Checkerboard (σ={_sigma})\nArrows point toward higher probability')
    _ax.set_aspect('equal')
    _ax.legend()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 4: The Score-Noise Connection

    Here's the key insight. The score of the noisy distribution is:

    $$\nabla_{x_t} \log p(x_t | x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

    **The score is proportional to the noise we added!** If we can predict $\epsilon$, we know the score.
    """)
    return


@app.cell
def _(data_grid_size, data_original, np, plt):
    np.random.seed(789)
    _n_show = 30
    _indices = np.random.choice(len(data_original), size=_n_show, replace=False)
    _x0 = data_original[_indices]
    _lim = data_grid_size / 2 + 1.5

    _t = 0.5
    _alpha_bar = 1 - _t

    _eps = np.random.randn(*_x0.shape)
    _xt = np.sqrt(_alpha_bar) * _x0 + np.sqrt(1 - _alpha_bar) * _eps

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='blue', alpha=0.1, s=20, label='All data')
    _ax.scatter(_x0[:, 0], _x0[:, 1], c='blue', alpha=0.8, s=50, label='Clean x₀')
    _ax.scatter(_xt[:, 0], _xt[:, 1], c='red', alpha=0.8, s=50, label='Noisy xₜ')

    for _i in range(_n_show):
        _ax.annotate('', xy=(_x0[_i, 0], _x0[_i, 1]), xytext=(_xt[_i, 0], _xt[_i, 1]),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.6, lw=1.5))

    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Score Direction at t={_t}\nGreen arrows point from noisy (red) toward clean (blue)')
    _ax.legend()
    _ax.set_aspect('equal')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 5: The Reverse Process (with KDE)

    Now that we understand the score, let's see the reverse process in action!

    **One way to estimate the score** is with **Kernel Density Estimation (KDE)**. If we have access to the training data, we can estimate the score at any point by looking at nearby data points.

    This is a valid method, but it has a limitation: it requires access to the original data. In practice, we want to generate **new** data without looking at the training set. That's why we'll later train a neural network instead.

    For now, let's see how reverse diffusion works using KDE. We use **Langevin dynamics**: iteratively move in the direction of the score (toward higher probability) plus a small amount of noise.

    Use the slider to step through the reverse process!
    """)
    return


@app.cell
def _(mo):
    reverse_step_slider = mo.ui.slider(
        start=0, stop=50, step=1, value=0,
        label="Reverse step", show_value=True
    )
    reverse_step_slider
    return (reverse_step_slider,)


@app.cell
def _(data_grid_size, data_original, plt, reverse_step_slider, reverse_trajectory):
    _step = reverse_step_slider.value
    _particles = reverse_trajectory[_step]
    _lim = data_grid_size / 2 + 1.5

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='#3498db', alpha=0.3, s=20, label='Original data')
    _ax.scatter(_particles[:, 0], _particles[:, 1], c='#e74c3c', alpha=0.6, s=20, label='Generated')

    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Reverse Diffusion: Step {_step}/50')
    _ax.set_aspect('equal')
    _ax.legend()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Move the slider from 0 to 50** to watch noise become data!

    ---

    ## Section 6: Training a Neural Network

    KDE works, but requires access to the training data during generation. **A better approach:** train a neural network to predict the score directly!

    ### One Model, All Timesteps

    We train a **single neural network** $\epsilon_\theta(x_t, t)$ that takes:
    - **Input:** A noisy point $x_t$ AND the timestep $t$
    - **Output:** The predicted noise $\hat{\epsilon}$

    The **same network handles all noise levels!** It learns:
    - From $t$: "how much noise is there?"
    - From $x_t$: "where did this point likely come from?"

    ### Training Algorithm

    Each training step:
    1. Sample a clean point $x_0$ from our data
    2. Sample a random timestep $t \sim \text{Uniform}(0, 1)$
    3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
    4. Create noisy point: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
    5. Train the network to predict $\epsilon$ from $(x_t, t)$

    **Training objective:**
    $$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

    Let's implement this with PyTorch!
    """)
    return


@app.cell
def _(nn, torch):
    class NoisePredictor(nn.Module):
        """A simple MLP for predicting noise in 2D diffusion."""

        def __init__(self, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden_dim),    # Input: (x, y, t)
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)     # Output: (noise_x, noise_y)
            )

        def forward(self, x, t):
            """Forward pass. x: (batch, 2), t: (batch,) or (batch, 1)"""
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inp = torch.cat([x, t], dim=-1)
            return self.net(inp)

    def train_diffusion_model(data, n_iterations=3000, batch_size=128, lr=0.001, seed=42):
        """Train the diffusion model with PyTorch."""
        torch.manual_seed(seed)

        # Convert data to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        model = NoisePredictor(hidden_dim=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []

        for _iter in range(n_iterations):
            # Sample batch
            indices = torch.randint(0, len(data_tensor), (batch_size,))
            x0 = data_tensor[indices]

            # Sample timesteps and noise
            t = torch.rand(batch_size) * 0.98 + 0.01  # Uniform(0.01, 0.99)
            alpha_bar = 1 - t
            epsilon = torch.randn(batch_size, 2)

            # Create noisy samples
            xt = torch.sqrt(alpha_bar).unsqueeze(-1) * x0 + torch.sqrt(1 - alpha_bar + 1e-8).unsqueeze(-1) * epsilon

            # Predict noise and compute loss
            epsilon_pred = model(xt, t)
            loss = torch.mean((epsilon_pred - epsilon) ** 2)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return model, losses

    return NoisePredictor, train_diffusion_model


@app.cell
def _(data_original, train_diffusion_model):
    # Train the model (takes ~10-20 seconds)
    trained_model, training_losses = train_diffusion_model(
        data_original,
        n_iterations=3000,
        batch_size=128,
        lr=0.01,
        seed=42
    )
    return trained_model, training_losses


@app.cell
def _(np, plt, training_losses):
    _window = 50
    _smoothed = []
    for _i in range(len(training_losses)):
        _start = max(0, _i - _window)
        _smoothed.append(sum(training_losses[_start:_i+1]) / (_i - _start + 1))

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_smoothed, 'b-', linewidth=1.5)
    _ax.set_xlabel('Iteration')
    _ax.set_ylabel('MSE Loss')
    _ax.set_title('Training Loss (smoothed)')
    _ax.set_yscale('log')
    _ax.grid(True, alpha=0.3)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 7: Sampling with the Learned Model

    Now let's use our trained network to generate new samples!
    """)
    return


@app.cell
def _(data_grid_size, torch, trained_model):
    def sample_with_learned_model(model, n_samples=200, n_steps=50, seed=42, data_scale=2.0):
        """Generate samples using the trained model with Langevin dynamics."""
        torch.manual_seed(seed)
        model.eval()

        with torch.no_grad():
            # Start from noise scaled to match the data range
            x = torch.randn(n_samples, 2) * data_scale

            for step in range(n_steps):
                # Go from t=1 (pure noise) to t=0 (clean data)
                t = 1.0 - step / n_steps
                t_tensor = torch.full((n_samples,), t)

                # Predict the noise
                epsilon_pred = model(x, t_tensor)

                # Convert noise prediction to score
                alpha_bar_t = 1 - t
                score = -epsilon_pred / (torch.sqrt(torch.tensor(1 - alpha_bar_t)) + 1e-4)

                # Langevin step: move in direction of score + add noise
                step_size = 0.1 * (1.0 - 0.5 * (1 - t))  # Larger steps early, smaller late
                noise_scale = (2 * step_size) ** 0.5 * t  # Less noise as we approach t=0

                x = x + step_size * score
                if step < n_steps - 1:
                    x = x + noise_scale * torch.randn(n_samples, 2)

        return x.numpy()

    # Scale based on grid size
    _data_scale = data_grid_size / 2 + 0.5
    generated_samples = sample_with_learned_model(trained_model, n_samples=300, n_steps=50, seed=123, data_scale=_data_scale)
    return generated_samples, sample_with_learned_model


@app.cell
def _(mo):
    sampling_steps_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=50,
        label="Sampling steps", show_value=True
    )
    sampling_steps_slider
    return (sampling_steps_slider,)


@app.cell
def _(
    data_grid_size,
    data_original,
    plt,
    sample_with_learned_model,
    sampling_steps_slider,
    trained_model,
):
    _n_steps = sampling_steps_slider.value
    _data_scale = data_grid_size / 2 + 0.5
    _generated = sample_with_learned_model(trained_model, n_samples=300, n_steps=_n_steps, seed=123, data_scale=_data_scale)
    _lim = data_grid_size / 2 + 1.5

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='#3498db', alpha=0.5, s=20, label='Original')
    _ax.scatter(_generated[:, 0], _generated[:, 1], c='#e74c3c', alpha=0.5, s=20, label='Generated')

    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Generated vs Original ({_n_steps} sampling steps)\nSame model applied {_n_steps} times!')
    _ax.set_aspect('equal')
    _ax.legend()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try different numbers of sampling steps!**

    - Few steps (5-10): Fast but lower quality
    - Many steps (50-100): Slower but better quality

    ---

    ## Section 8: How Sampling Works

    Let's look inside the sampling loop. Remember: we use the **same model** at every step, but with different timestep values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Sampling Algorithm

    ```
    Start with pure noise: x ~ N(0, I)

    for t = 1.0, 0.98, 0.96, ..., 0.02:
        1. Query the model: noise_pred = model(x, t)
        2. Compute score: score = -noise_pred / sqrt(1 - alpha_bar_t)
        3. Take a step toward data: x = x + step_size * score
        4. Add small noise (unless at final step)

    Return x as the generated sample
    ```

    **Key insight:** The model learns to predict noise at ANY noise level because we pass `t` as input. During sampling, we call it ~50 times with decreasing `t` values, gradually refining from noise to data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 9: What the Network Learned

    Let's visualize the score field at different noise levels.
    """)
    return


@app.cell
def _(mo):
    score_t_slider = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.5,
        label="Timestep t", show_value=True
    )
    score_t_slider
    return (score_t_slider,)


@app.cell
def _(data_grid_size, data_original, np, plt, score_t_slider, torch, trained_model):
    _t = score_t_slider.value
    _alpha_bar = 1 - _t
    _lim = data_grid_size / 2 + 1.0

    _x = np.linspace(-_lim, _lim, 20)
    _y = np.linspace(-_lim, _lim, 20)
    _X, _Y = np.meshgrid(_x, _y)
    _grid_points = np.column_stack([_X.ravel(), _Y.ravel()])

    # Use PyTorch for prediction
    trained_model.eval()
    with torch.no_grad():
        _grid_tensor = torch.tensor(_grid_points, dtype=torch.float32)
        _t_tensor = torch.full((len(_grid_points),), _t)
        _epsilon_pred = trained_model(_grid_tensor, _t_tensor).numpy()

    _score = -_epsilon_pred / np.sqrt(1 - _alpha_bar + 1e-8)

    _U = _score[:, 0].reshape(_X.shape)
    _V = _score[:, 1].reshape(_Y.shape)

    _magnitude = np.sqrt(_U**2 + _V**2 + 0.01)
    _U_norm = _U / _magnitude
    _V_norm = _V / _magnitude

    _fig, _ax = plt.subplots(figsize=(9, 7))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='blue', alpha=0.2, s=20, label='Original data')
    _ax.quiver(_X, _Y, _U_norm, _V_norm, color='red', alpha=0.7, scale=30)

    _ax.set_xlim(-_lim - 0.5, _lim + 0.5)
    _ax.set_ylim(-_lim - 0.5, _lim + 0.5)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Learned Score Field at t = {_t:.2f}\nArrows point toward data')
    _ax.set_aspect('equal')
    _ax.legend()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Explore different timesteps!**

    - **High t (near 1):** Simple score field — arrows mostly point toward the center
    - **Low t (near 0):** Complex score field — arrows capture the checkerboard structure

    ---

    ## Summary

    ### Key Equations

    | Concept | Equation |
    |---------|----------|
    | Forward process | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ |
    | Score function | $s(x) = \nabla_x \log p(x)$ |
    | Score-noise connection | $s(x_t, t) = -\epsilon / \sqrt{1 - \bar{\alpha}_t}$ |
    | Training loss | $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ |

    ### Key Insights

    1. **Forward process is easy:** Just add noise according to a schedule
    2. **Score points toward data:** The gradient of log probability tells us where to go
    3. **Predicting noise = knowing the score:** They're mathematically equivalent
    4. **Iterative refinement:** Start from noise, take small steps toward data

    ### What We Didn't Cover

    - **Conditioning:** Adding text/class labels to guide generation
    - **Latent diffusion:** Working in a compressed latent space (Stable Diffusion)
    - **Faster samplers:** DDIM, DPM-Solver for fewer steps
    """)
    return


if __name__ == "__main__":
    app.run()
