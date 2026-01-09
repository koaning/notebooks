# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib>=3.10.0",
#     "numpy==2.4.0",
#     "torch>=2.0.0",
#     "drawdata>=0.3.0",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from drawdata import ScatterWidget
    from sklearn.datasets import make_moons
    return ScatterWidget, make_moons, mo, nn, np, plt, torch


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Lazy Optimizer

    Most optimizers compute gradients at every step. But what if we could be lazier?

    **The idea:** Once we find a good direction to move (via gradient), keep walking in that direction
    until the loss starts increasing. When the loss increases, **undo the step**, halve the step size,
    and try again. Only compute a new gradient when the step size becomes too small.

    This is a form of **line search with backtracking** - we exploit the current direction as long
    as it's useful, potentially saving many gradient computations.

    $$\text{While } L(\theta_{t+1}) < L(\theta_t): \quad \theta_{t+1} = \theta_t - \alpha \cdot d$$

    $$\text{If } L(\theta_{t+1}) > L(\theta_t): \quad \text{undo step}, \quad \alpha \leftarrow \alpha / 2$$

    where $d = -\nabla L / \|\nabla L\|$ is the normalized negative gradient and $\alpha$ is halved when we overshoot.
    """)
    return


@app.cell
def _(torch):
    class LazyOptimizer(torch.optim.Optimizer):
        """
        A "Lazy" optimizer that minimizes gradient computations.

        Strategy:
        1. Compute normalized gradient direction once
        2. Take steps in that direction until loss increases
        3. When loss increases: undo step, halve step size, retry
        4. If step size too small: compute new gradient direction
        """

        def __init__(self, params, step_size=0.01, min_step_size=1e-6):
            defaults = dict(step_size=step_size, min_step_size=min_step_size)
            super().__init__(params, defaults)
            self._direction = None
            self._prev_params = None
            self._prev_loss = float('inf')

        def step(self, closure):
            """Performs a single optimization step with backtracking."""
            loss = closure()
            current_loss = loss.item()
            group = self.param_groups[0]

            # Need new direction if first step or loss got worse and we can't backtrack further
            need_new_direction = self._direction is None

            # If loss increased, try backtracking
            if current_loss > self._prev_loss and self._prev_params is not None:
                self._restore_params()
                group['current_step_size'] = group.get('current_step_size', group['step_size']) / 2

                # If step size too small, get a new direction instead
                if group['current_step_size'] < group['min_step_size']:
                    need_new_direction = True

            if need_new_direction:
                loss = closure()
                current_loss = loss.item()
                self._direction = self._compute_direction()
                group['current_step_size'] = group['step_size']

            self._save_params()
            self._take_step(group['current_step_size'])
            self._prev_loss = current_loss
            return loss

        def _params_list(self):
            """Flat list of all parameters."""
            return [p for group in self.param_groups for p in group['params']]

        def _compute_direction(self):
            """Compute normalized negative gradient direction."""
            grads = [-p.grad.clone() for p in self._params_list()]
            norm = sum(g.norm().item() ** 2 for g in grads) ** 0.5
            if norm > 0:
                return [g / norm for g in grads]
            return grads

        def _save_params(self):
            """Save current parameters for backtracking."""
            self._prev_params = [p.data.clone() for p in self._params_list()]

        def _restore_params(self):
            """Restore parameters to saved values."""
            for p, saved in zip(self._params_list(), self._prev_params):
                p.data.copy_(saved)

        def _take_step(self, step_size):
            """Move parameters in the stored direction."""
            with torch.no_grad():
                for p, d in zip(self._params_list(), self._direction):
                    p.add_(d * step_size)
    return (LazyOptimizer,)


@app.cell
def _(nn, torch):
    class SimpleMLP(nn.Module):
        """MLP with configurable number of hidden layers."""

        def __init__(self, input_dim=2, hidden_dim=16, output_dim=2, n_hidden=3):
            super().__init__()
            self.n_hidden = n_hidden
            self.hidden_dim = hidden_dim

            layers = [nn.Linear(input_dim, hidden_dim)]
            for _ in range(n_hidden - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x))
            return self.layers[-1](x)

    def clone_model(model):
        """Create a clone of a model with the same weights."""
        clone = SimpleMLP(
            input_dim=model.layers[0].in_features,
            hidden_dim=model.hidden_dim,
            output_dim=model.layers[-1].out_features,
            n_hidden=model.n_hidden
        )
        clone.load_state_dict(model.state_dict())
        return clone
    return SimpleMLP, clone_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw Your Dataset

    Use the widget below to draw points. Click to add points of different colors.
    Draw at least two classes (colors) for classification.
    """)
    return


@app.cell
def _(ScatterWidget, mo):
    scatter_widget = mo.ui.anywidget(ScatterWidget())
    scatter_widget
    return (scatter_widget,)


@app.cell
def _(is_script_mode, make_moons, np, scatter_widget, torch):
    def get_data_from_widget():
        """Extract and prepare data from the scatter widget."""
        X, y = scatter_widget.widget.data_as_X_y
        if len(X) == 0:
            return None, None, "No data drawn yet. Please draw some points above."

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return None, None, "Please draw at least two different classes (colors)."

        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y], dtype=np.int64)

        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return torch.tensor(X), torch.tensor(y), None

    def get_synthetic_data(n_samples=200, noise=0.2, seed=42):
        """Generate synthetic data for script mode testing."""
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X = X.astype(np.float32)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return torch.tensor(X), torch.tensor(y, dtype=torch.int64), None

    if is_script_mode:
        X_data, y_data, data_error = get_synthetic_data()
        print(f"Script mode: Generated {len(X_data)} synthetic data points (make_moons)")
    else:
        X_data, y_data, data_error = get_data_from_widget()
    return X_data, data_error, y_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Configuration

    Adjust the parameters below and click "Train Models" to compare the optimizers.
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(start=0.001, stop=0.1, step=0.005, value=0.01, label="Learning Rate (SGD/Adam)")
    epochs_slider = mo.ui.slider(start=50, stop=10000, step=50, value=2000, label="Epochs")
    hidden_dim_slider = mo.ui.slider(start=4, stop=64, step=4, value=16, label="Hidden Dimension")
    n_hidden_slider = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Number of Hidden Layers")
    step_size_slider = mo.ui.slider(start=0.001, stop=0.05, step=0.002, value=0.01, label="Lazy Step Size")
    mo.hstack([lr_slider, step_size_slider], justify="start")
    return (
        epochs_slider,
        hidden_dim_slider,
        lr_slider,
        n_hidden_slider,
        step_size_slider,
    )


@app.cell(hide_code=True)
def _(epochs_slider, hidden_dim_slider, mo, n_hidden_slider):
    mo.hstack([epochs_slider, hidden_dim_slider, n_hidden_slider], justify="start")
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Models")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def _(LazyOptimizer, SimpleMLP, clone_model, nn, torch):
    import time

    def train_with_closure(model, optimizer, X, y, epochs):
        """Training loop for closure-based optimizers (LazyOptimizer)."""
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        times = []
        start = time.perf_counter()

        for _epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                output = model(X)
                loss = loss_fn(output, y)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(loss.item())
            times.append(time.perf_counter() - start)

        return losses, times

    def train_standard(model, optimizer, X, y, epochs):
        """Standard training loop for SGD/Adam."""
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        times = []
        start = time.perf_counter()

        for _epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            times.append(time.perf_counter() - start)

        return losses, times

    def run_training(X, y, lr, step_size, epochs, hidden_dim, n_hidden, seed=42):
        """Train all three optimizers and return results."""
        torch.manual_seed(seed)

        num_classes = len(torch.unique(y))

        model_base = SimpleMLP(
            input_dim=2, hidden_dim=hidden_dim, output_dim=num_classes, n_hidden=n_hidden
        )

        model_sgd = clone_model(model_base)
        model_adam = clone_model(model_base)
        model_lazy = clone_model(model_base)

        opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)
        opt_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)
        opt_lazy = LazyOptimizer(model_lazy.parameters(), step_size=step_size)

        losses_sgd, times_sgd = train_standard(model_sgd, opt_sgd, X, y, epochs)
        losses_adam, times_adam = train_standard(model_adam, opt_adam, X, y, epochs)
        losses_lazy, times_lazy = train_with_closure(model_lazy, opt_lazy, X, y, epochs)

        return {
            'sgd': {'model': model_sgd, 'losses': losses_sgd, 'times': times_sgd},
            'adam': {'model': model_adam, 'losses': losses_adam, 'times': times_adam},
            'lazy': {'model': model_lazy, 'losses': losses_lazy, 'times': times_lazy}
        }
    return (run_training,)


@app.cell(hide_code=True)
def _(
    X_data,
    data_error,
    epochs_slider,
    hidden_dim_slider,
    is_script_mode,
    lr_slider,
    n_hidden_slider,
    run_training,
    step_size_slider,
    train_button,
    y_data,
):
    training_results = None

    if is_script_mode:
        # Auto-run in script mode using slider default values
        if not data_error:
            print("Training models...")
            training_results = run_training(
                X_data, y_data,
                lr=lr_slider.value,
                step_size=step_size_slider.value,
                epochs=epochs_slider.value,
                hidden_dim=hidden_dim_slider.value,
                n_hidden=n_hidden_slider.value
            )
            print("Training complete!")
        else:
            print(f"Cannot train: {data_error}")
    elif train_button.value and not data_error:
        training_results = run_training(
            X_data, y_data,
            lr=lr_slider.value,
            step_size=step_size_slider.value,
            epochs=epochs_slider.value,
            hidden_dim=hidden_dim_slider.value,
            n_hidden=n_hidden_slider.value
        )
    return (training_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loss Comparison

    **Left:** Loss vs epochs - shows convergence behavior per iteration.
    **Right:** Loss vs wall-clock time - shows actual computational efficiency.
    """)
    return


@app.cell(hide_code=True)
def _(plt, training_results):
    fig_loss, (ax_epochs, ax_time) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs Epochs
    ax_epochs.plot(training_results['sgd']['losses'], label='SGD', alpha=0.8, linewidth=2)
    ax_epochs.plot(training_results['adam']['losses'], label='Adam', alpha=0.8, linewidth=2)
    ax_epochs.plot(training_results['lazy']['losses'], label='Lazy Optimizer', alpha=0.8, linewidth=2)
    ax_epochs.set_xlabel('Epoch')
    ax_epochs.set_ylabel('Loss')
    ax_epochs.set_title('Loss vs Epochs')
    ax_epochs.legend()
    ax_epochs.grid(True, alpha=0.3)
    ax_epochs.set_yscale('log')

    # Loss vs Time
    ax_time.plot(training_results['sgd']['times'], training_results['sgd']['losses'], label='SGD', alpha=0.8, linewidth=2)
    ax_time.plot(training_results['adam']['times'], training_results['adam']['losses'], label='Adam', alpha=0.8, linewidth=2)
    ax_time.plot(training_results['lazy']['times'], training_results['lazy']['losses'], label='Lazy Optimizer', alpha=0.8, linewidth=2)
    ax_time.set_xlabel('Time (seconds)')
    ax_time.set_ylabel('Loss')
    ax_time.set_title('Loss vs Wall-Clock Time')
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_time.set_yscale('log')

    plt.tight_layout()
    fig_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Boundaries

    Each panel shows the decision boundary learned by each optimizer.
    The colored regions show which class would be predicted for points in that area.
    """)
    return


@app.cell(hide_code=True)
def _(X_data, np, plt, torch, training_results, y_data):
    fig_db, axes = plt.subplots(1, 3, figsize=(15, 5))

    models_info = [
        ('SGD', training_results['sgd']['model']),
        ('Adam', training_results['adam']['model']),
        ('Lazy', training_results['lazy']['model'])
    ]

    X_np = X_data.numpy()
    y_np = y_data.numpy()

    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    for ax, (_name, _model) in zip(axes, models_info):
        with torch.no_grad():
            Z = _model(grid).argmax(dim=1).numpy()
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='RdYlBu', edgecolors='black', s=50)
        ax.set_title(_name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    fig_db
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations

    The **Lazy Optimizer** takes a different approach than traditional optimizers:

    - **SGD** and **Adam** compute gradients at every step
    - **Lazy** computes gradients only when the loss starts increasing

    This can be beneficial when:
    - Gradient computation is expensive (large models, complex loss functions)
    - The loss landscape is relatively smooth (the same direction stays useful for many steps)

    It may struggle when:
    - The loss landscape is highly curved (the optimal direction changes rapidly)
    - The step size is too large (overshoots before detecting the increase)

    Try adjusting the **Lazy Step Size** to see how it affects convergence!
    """)
    return


if __name__ == "__main__":
    app.run()
