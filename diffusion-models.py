# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
#     "scipy>=1.12",
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
    return mo, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diffusion Models: Learning to Reverse Noise

    Diffusion models are one of the most powerful generative models today. The key idea is surprisingly simple:

    1. **Destroy data** by gradually adding noise until it becomes pure random noise
    2. **Learn to reverse** this process, so we can start from noise and generate new data

    Let's see the full picture first, then break it down step by step.
    """)
    return


@app.cell
def _(np):
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

    # Generate our dataset
    data_original = make_checkerboard(n_samples=1000, grid_size=4, seed=42)
    return data_original, make_checkerboard


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

    **Forward process (left to right):** We start with structured data (a checkerboard pattern) and gradually add noise until it becomes random.

    **Reverse process (right to left):** We learn to undo this! Starting from noise, we iteratively denoise to recover structure. Watch the grid pattern emerge first, then the sharp square boundaries.
    """)
    return


@app.cell
def _(forward_trajectory, plt):
    # Show forward process: data -> noise (5 snapshots)
    _fig, _axes = plt.subplots(1, 5, figsize=(15, 3))
    _steps_to_show = [0, 2, 5, 8, 10]
    _labels = ['Start\n(data)', 't = 0.2', 't = 0.5', 't = 0.8', 'End\n(noise)']

    for _idx, _step in enumerate(_steps_to_show):
        _ax = _axes[_idx]
        _data = forward_trajectory[_step]

        _ax.scatter(_data[:, 0], _data[:, 1], c='#3498db', alpha=0.6, s=8)
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
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
def _(plt, reverse_trajectory):
    # Show reverse process: noise -> data (5 snapshots)
    _fig, _axes = plt.subplots(1, 5, figsize=(15, 3))
    _steps_to_show = [0, 12, 25, 37, 50]
    _labels = ['Start\n(noise)', 'Step 12', 'Step 25', 'Step 37', 'End\n(data)']

    for _idx, _step in enumerate(_steps_to_show):
        _ax = _axes[_idx]
        _data = reverse_trajectory[_step]

        _ax.scatter(_data[:, 0], _data[:, 1], c='#e74c3c', alpha=0.6, s=8)
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
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
def _(data_original, np, plt, timestep_slider):
    _t = timestep_slider.value
    _alpha_bar = 1 - _t

    np.random.seed(42)
    _noise = np.random.randn(*data_original.shape)
    _data_noisy = np.sqrt(_alpha_bar) * data_original + np.sqrt(1 - _alpha_bar + 1e-8) * _noise

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.scatter(_data_noisy[:, 0], _data_noisy[:, 1], c='#3498db', alpha=0.6, s=15)
    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
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
def _(data_original, forward_t_slider, np, plt):
    _t = forward_t_slider.value
    _alpha_bar = 1 - _t

    np.random.seed(123)
    _indices = np.random.choice(len(data_original), size=50, replace=False)
    _x0 = data_original[_indices]
    _eps = np.random.randn(*_x0.shape)
    _xt = np.sqrt(_alpha_bar) * _x0 + np.sqrt(1 - _alpha_bar + 1e-8) * _eps

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(12, 4))

    _ax1.scatter(_x0[:, 0], _x0[:, 1], c='blue', alpha=0.7, s=50)
    _ax1.set_xlim(-3, 3)
    _ax1.set_ylim(-3, 3)
    _ax1.set_title(r"$x_0$ (original)")
    _ax1.set_aspect('equal')

    _ax2.scatter(_eps[:, 0], _eps[:, 1], c='red', alpha=0.7, s=50)
    _ax2.set_xlim(-4, 4)
    _ax2.set_ylim(-4, 4)
    _ax2.set_title(r"$\epsilon$ (noise)")
    _ax2.set_aspect('equal')

    _ax3.scatter(_xt[:, 0], _xt[:, 1], c='purple', alpha=0.7, s=50)
    _ax3.set_xlim(-4, 4)
    _ax3.set_ylim(-4, 4)
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

    The **noise schedule** $\bar{\alpha}_t$ determines how quickly we destroy structure.

    **Linear schedule:** $\bar{\alpha}_t = 1 - t$

    **Cosine schedule:** Preserves more structure at early timesteps.
    """)
    return


@app.cell
def _(mo):
    schedule_dropdown = mo.ui.dropdown(
        options=["linear", "cosine"],
        value="linear",
        label="Noise schedule"
    )
    schedule_dropdown
    return (schedule_dropdown,)


@app.cell
def _(np, plt, schedule_dropdown):
    _timesteps = np.linspace(0, 1, 100)

    def _linear_schedule(t):
        return 1 - t

    def _cosine_schedule(t):
        s = 0.008
        f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        return f_t / f_0

    _schedule_name = schedule_dropdown.value
    if _schedule_name == "linear":
        _alpha_bars = _linear_schedule(_timesteps)
    else:
        _alpha_bars = _cosine_schedule(_timesteps)

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.plot(_timesteps, _alpha_bars, 'b-', linewidth=2, label=r'$\bar{\alpha}_t$')
    _ax.plot(_timesteps, np.sqrt(_alpha_bars), 'g-', linewidth=2, label=r'$\sqrt{\bar{\alpha}_t}$ (signal coef)')
    _ax.plot(_timesteps, np.sqrt(1 - _alpha_bars + 1e-8), 'r-', linewidth=2, label=r'$\sqrt{1-\bar{\alpha}_t}$ (noise coef)')

    _ax.set_xlabel('Timestep t')
    _ax.set_ylabel('Coefficient value')
    _ax.set_title(f'{_schedule_name.capitalize()} Schedule')
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    _fig
    return (schedule_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Section 3: The Score Function

    The **score function** is the key to reversing diffusion:

    $$s(x) = \nabla_x \log p(x)$$

    It's the gradient of the log probability density. **The score points toward higher probability regions.**

    For a Gaussian $\mathcal{N}(\mu, \sigma^2 I)$: $s(x) = -\frac{x - \mu}{\sigma^2}$
    """)
    return


@app.cell
def _(mo):
    sigma_slider = mo.ui.slider(
        start=0.3, stop=2.0, step=0.1, value=0.5,
        label="σ (spread)", show_value=True
    )
    sigma_slider
    return (sigma_slider,)


@app.cell
def _(np, plt, sigma_slider):
    _sigma = sigma_slider.value
    _mu = np.array([0.0, 0.0])

    _x = np.linspace(-3, 3, 15)
    _y = np.linspace(-3, 3, 15)
    _X, _Y = np.meshgrid(_x, _y)

    _U = -(_X - _mu[0]) / (_sigma ** 2)
    _V = -(_Y - _mu[1]) / (_sigma ** 2)

    _magnitude = np.sqrt(_U**2 + _V**2 + 0.01)
    _U_norm = _U / _magnitude
    _V_norm = _V / _magnitude

    _fig, _ax = plt.subplots(figsize=(7, 7))

    _xx = np.linspace(-3, 3, 100)
    _yy = np.linspace(-3, 3, 100)
    _XX, _YY = np.meshgrid(_xx, _yy)
    _ZZ = np.exp(-(_XX**2 + _YY**2) / (2 * _sigma**2))
    _ax.contourf(_XX, _YY, _ZZ, levels=20, cmap='Blues', alpha=0.5)

    _ax.quiver(_X, _Y, _U_norm, _V_norm, color='red', alpha=0.8, scale=25)

    _ax.scatter([0], [0], s=200, c='blue', marker='*', zorder=10, label='Mean μ')
    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-3, 3)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Score Field for Gaussian (σ={_sigma})\nArrows point toward higher probability')
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
def _(data_original, np, plt):
    np.random.seed(789)
    _n_show = 30
    _indices = np.random.choice(len(data_original), size=_n_show, replace=False)
    _x0 = data_original[_indices]

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

    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
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

    ## Section 5: The Reverse Process

    The **DDPM sampling algorithm** reverses the diffusion:

    For $t = T, T-1, \ldots, 1$:
    1. Predict the noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
    2. Compute the mean: $\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}\right)$
    3. Add a little noise: $x_{t-1} = \mu + \sigma_t \cdot z$

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
def _(data_original, plt, reverse_step_slider, reverse_trajectory):
    _step = reverse_step_slider.value
    _particles = reverse_trajectory[_step]

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='#3498db', alpha=0.3, s=20, label='Original data')
    _ax.scatter(_particles[:, 0], _particles[:, 1], c='#e74c3c', alpha=0.6, s=20, label='Generated')

    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
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

    ## Section 6: Training the Score Network

    In practice, we train a neural network $\epsilon_\theta(x_t, t)$ to predict the noise.

    **Training objective:**
    $$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

    Let's implement a simple MLP in pure numpy!
    """)
    return


@app.cell
def _(np):
    class SimpleMLP:
        """A simple 2-layer MLP for predicting noise in 2D diffusion."""

        def __init__(self, hidden_dim=64, seed=42):
            np.random.seed(seed)
            scale1 = np.sqrt(2.0 / 3)
            scale2 = np.sqrt(2.0 / hidden_dim)
            scale3 = np.sqrt(2.0 / hidden_dim)

            self.W1 = np.random.randn(3, hidden_dim) * scale1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
            self.b2 = np.zeros(hidden_dim)
            self.W3 = np.random.randn(hidden_dim, 2) * scale3
            self.b3 = np.zeros(2)

        def forward(self, x, t):
            """Forward pass. x: (batch, 2), t: (batch,) or scalar"""
            if np.isscalar(t):
                t = np.full(len(x), t)

            inp = np.column_stack([x, t])
            h1 = np.maximum(0, inp @ self.W1 + self.b1)
            h2 = np.maximum(0, h1 @ self.W2 + self.b2)
            out = h2 @ self.W3 + self.b3

            return out, (inp, h1, h2)

        def backward(self, x, t, target, lr=0.001):
            """Backward pass with gradient descent."""
            out, (inp, h1, h2) = self.forward(x, t)

            batch_size = len(x)
            d_out = 2 * (out - target) / batch_size

            d_W3 = h2.T @ d_out
            d_b3 = d_out.sum(axis=0)

            d_h2 = d_out @ self.W3.T
            d_h2 = d_h2 * (h2 > 0)

            d_W2 = h1.T @ d_h2
            d_b2 = d_h2.sum(axis=0)

            d_h1 = d_h2 @ self.W2.T
            d_h1 = d_h1 * (h1 > 0)

            d_W1 = inp.T @ d_h1
            d_b1 = d_h1.sum(axis=0)

            self.W3 -= lr * d_W3
            self.b3 -= lr * d_b3
            self.W2 -= lr * d_W2
            self.b2 -= lr * d_b2
            self.W1 -= lr * d_W1
            self.b1 -= lr * d_b1

            loss = np.mean((out - target) ** 2)
            return loss

    def train_diffusion_model(data, n_iterations=2000, batch_size=128, lr=0.01, seed=42):
        """Train the diffusion model."""
        np.random.seed(seed)
        model = SimpleMLP(hidden_dim=64, seed=seed)
        losses = []

        for _iter in range(n_iterations):
            indices = np.random.choice(len(data), size=batch_size, replace=True)
            x0 = data[indices]

            t = np.random.uniform(0.01, 0.99, size=batch_size)
            alpha_bar = 1 - t
            epsilon = np.random.randn(batch_size, 2)
            xt = np.sqrt(alpha_bar)[:, np.newaxis] * x0 + np.sqrt(1 - alpha_bar + 1e-8)[:, np.newaxis] * epsilon

            loss = model.backward(xt, t, epsilon, lr=lr)
            losses.append(loss)

        return model, losses

    return SimpleMLP, train_diffusion_model


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
def _(np, trained_model):
    def sample_with_learned_model(model, n_samples=200, n_steps=50, seed=42):
        """Generate samples using the trained model."""
        np.random.seed(seed)
        x = np.random.randn(n_samples, 2)

        for step in range(n_steps):
            t = 1 - step / n_steps
            t_arr = np.full(n_samples, t)

            epsilon_pred, _ = model.forward(x, t_arr)

            alpha_bar_t = 1 - t
            alpha_bar_prev = 1 - max(0, t - 1/n_steps)
            alpha_t = alpha_bar_t / (alpha_bar_prev + 1e-8)
            beta_t = 1 - alpha_t

            coef1 = 1 / np.sqrt(alpha_t + 1e-8)
            coef2 = beta_t / np.sqrt(1 - alpha_bar_t + 1e-8)
            mu = coef1 * (x - coef2 * epsilon_pred)

            if t > 1/n_steps:
                sigma = np.sqrt(beta_t)
                noise = np.random.randn(n_samples, 2)
                x = mu + sigma * noise
            else:
                x = mu

        return x

    generated_samples = sample_with_learned_model(trained_model, n_samples=300, n_steps=50, seed=123)
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
    data_original,
    np,
    plt,
    sample_with_learned_model,
    sampling_steps_slider,
    trained_model,
):
    _n_steps = sampling_steps_slider.value
    _generated = sample_with_learned_model(trained_model, n_samples=300, n_steps=_n_steps, seed=123)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='#3498db', alpha=0.5, s=20, label='Original')
    _ax.scatter(_generated[:, 0], _generated[:, 1], c='#e74c3c', alpha=0.5, s=20, label='Generated')

    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
    _ax.set_xlabel('x')
    _ax.set_ylabel('y')
    _ax.set_title(f'Generated vs Original Data ({_n_steps} sampling steps)')
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

    ## Section 8: What the Network Learned

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
def _(data_original, np, plt, score_t_slider, trained_model):
    _t = score_t_slider.value
    _alpha_bar = 1 - _t

    _x = np.linspace(-3.5, 3.5, 20)
    _y = np.linspace(-3.5, 3.5, 20)
    _X, _Y = np.meshgrid(_x, _y)
    _grid_points = np.column_stack([_X.ravel(), _Y.ravel()])

    _t_arr = np.full(len(_grid_points), _t)
    _epsilon_pred, _ = trained_model.forward(_grid_points, _t_arr)

    _score = -_epsilon_pred / np.sqrt(1 - _alpha_bar + 1e-8)

    _U = _score[:, 0].reshape(_X.shape)
    _V = _score[:, 1].reshape(_Y.shape)

    _magnitude = np.sqrt(_U**2 + _V**2 + 0.01)
    _U_norm = _U / _magnitude
    _V_norm = _V / _magnitude

    _fig, _ax = plt.subplots(figsize=(9, 7))

    _ax.scatter(data_original[:, 0], data_original[:, 1], c='blue', alpha=0.2, s=20, label='Original data')
    _ax.quiver(_X, _Y, _U_norm, _V_norm, color='red', alpha=0.7, scale=30)

    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
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
