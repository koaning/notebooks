# /// script
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "polars==1.39.3",
#     "scikit-learn==1.7.2",
#     "torch==2.11.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import polars as pl
    import altair as alt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)
    return F, nn, np, pl, torch


@app.cell
def _(np, torch):
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X_all = (mnist.data.astype("float32") / 255.0).reshape(-1, 1, 28, 28)
    y_all = mnist.target.astype(int)

    rng = np.random.default_rng(42)
    train_idx = rng.choice(60000, size=1024, replace=False)
    test_idx = rng.choice(np.arange(60000, 70000), size=512, replace=False)

    X_train = torch.from_numpy(X_all[train_idx])
    y_train = torch.from_numpy(y_all[train_idx]).long()
    X_test = torch.from_numpy(X_all[test_idx])
    y_test = torch.from_numpy(y_all[test_idx]).long()

    N_CLASSES = 10
    TRAIN_SHAPE = tuple(X_train.shape)
    TEST_SHAPE = tuple(X_test.shape)
    return N_CLASSES, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ES Trick on Fashion MNIST

    This notebook uses a small PyTorch CNN:

    - `Conv2d(1, 2, kernel_size=3, padding=1)`
    - `ReLU + MaxPool2d(2)`
    - `Linear(2 * 14 * 14, 16)`
    - `ReLU`
    - `Linear(16, 10)`

    The comparison follows `es-trick.py`:

    - **Dense**: full Gaussian perturbations for every parameter tensor
    - **Low-rank**: factorized perturbations `A @ B / sqrt(rank)`

    The update is the weighted sum of perturbed parameter tensors. After score
    standardization, the constant base weights mostly cancel, which is the trick
    that makes the structured noise family interesting.
    """)
    return


@app.cell
def _(mo):
    rank_slider = mo.ui.slider(1, 16, 1, value=4, label="rank")
    steps_slider = mo.ui.slider(20, 1000, 20, value=1000, label="steps")
    mo.hstack([rank_slider, steps_slider], justify="start")
    return rank_slider, steps_slider


@app.cell(hide_code=True)
def _(dense_result, low_rank_result, mo, rank_slider, steps_slider):
    mo.md(f"""
    ## Test Set Results

    | Method | Best Train Accuracy | Matching Test Accuracy | Random values / member |
    |--------|---------------------|------------------------|------------------------|
    | {dense_result["kind"]} | {dense_result["metrics"]["train_accuracy"]:.3f} | {dense_result["metrics"]["test_accuracy"]:.3f} | {dense_result["random_values"]:,} |
    | {low_rank_result["kind"]} | {low_rank_result["metrics"]["train_accuracy"]:.3f} | {low_rank_result["metrics"]["test_accuracy"]:.3f} | {low_rank_result["random_values"]:,} |

    At rank {rank_slider.value}, the low-rank branch uses {low_rank_result["random_values"] / dense_result["random_values"]:.1%} of the random values of the dense branch across {steps_slider.value} ES steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why This Matches `es-trick.py`

    For each population member we evaluate a perturbed model `theta + eps_i` and then
    update with `sum_i z_i (theta + eps_i)`, where `z_i` are standardized losses.

    Because `sum_i z_i` is approximately zero, the constant `theta` term mostly
    cancels, leaving an ES-like update driven by the perturbations. That means the
    search distribution matters: replacing dense Gaussian noise with structured
    `A @ B / sqrt(rank)` changes the exploration budget without changing the basic
    ES trick.
    """)
    return


@app.cell
def _(
    F,
    N_CLASSES,
    X_test,
    X_train,
    nn,
    np,
    pl,
    rank_slider,
    steps_slider,
    torch,
    y_test,
    y_train,
):
    import marimo as _mo


    class SmallConvNet(nn.Module):
        def __init__(self, conv_channels=2, hidden_dim=16):
            super().__init__()
            self.conv = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
            self.hidden = nn.Linear(conv_channels * 14 * 14, hidden_dim)
            self.out = nn.Linear(hidden_dim, N_CLASSES)

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = F.max_pool2d(x, kernel_size=2)
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.hidden(x))
            return self.out(x)


    def clone_params(model):
        return [param.detach().clone() for param in model.parameters()]


    def set_params(model, params):
        with torch.inference_mode():
            for model_param, new_param in zip(model.parameters(), params):
                model_param.copy_(new_param)


    def add_params(base_params, deltas):
        return [base + delta for base, delta in zip(base_params, deltas)]


    def zeros_like_params(params):
        return [torch.zeros_like(param) for param in params]


    def sample_noise_like(param, sigma, kind, rank, generator):
        if kind == "dense" or param.ndim == 0:
            return torch.randn(param.shape, generator=generator) * sigma

        rows = param.shape[0]
        cols = int(param.numel() // rows)
        effective_rank = max(1, min(rank, rows, cols))
        a = torch.randn((rows, effective_rank), generator=generator) * sigma
        b = torch.randn((effective_rank, cols), generator=generator)
        return ((a @ b) / np.sqrt(effective_rank)).reshape(param.shape)


    def evaluate_model(model):
        with torch.inference_mode():
            train_logits = model(X_train)
            test_logits = model(X_test)
            train_loss = F.cross_entropy(train_logits, y_train).item()
            test_loss = F.cross_entropy(test_logits, y_test).item()
            train_acc = (
                (train_logits.argmax(dim=1) == y_train).float().mean().item()
            )
            test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
        return {
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
        }


    def evaluate_params(params):
        model = SmallConvNet()
        set_params(model, params)
        return evaluate_model(model)


    def random_value_count(params, kind, rank):
        total = 0
        for param in params:
            if kind == "dense" or param.ndim == 0:
                total += int(param.numel())
                continue
            rows = param.shape[0]
            cols = int(param.numel() // rows)
            effective_rank = max(1, min(rank, rows, cols))
            total += effective_rank * (rows + cols)
        return total


    @_mo.cache()
    def train_es(
        kind="dense",
        rank=4,
        n_steps=160,
        n_pop=32,
        lr=0.03,
        sigma=0.02,
        batch_size=256,
        seed=7,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SmallConvNet()
        params = clone_params(model)
        generator = torch.Generator().manual_seed(seed)
        history = []
        best_params = [param.clone() for param in params]
        best_metrics = evaluate_params(params)

        for step in range(n_steps):
            batch_idx = torch.randint(
                0, X_train.shape[0], (batch_size,), generator=generator
            )
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            candidates = []
            losses = []

            for _ in range(n_pop):
                noise = [
                    sample_noise_like(
                        param,
                        sigma=sigma,
                        kind=kind,
                        rank=rank,
                        generator=generator,
                    )
                    for param in params
                ]
                candidate = add_params(params, noise)
                set_params(model, candidate)
                with torch.inference_mode():
                    losses.append(F.cross_entropy(model(xb), yb).item())
                candidates.append(candidate)

            z_scores = (np.asarray(losses, dtype=np.float32) - np.mean(losses)) / (
                np.std(losses) + 1e-6
            )
            update = zeros_like_params(params)

            for score, candidate in zip(z_scores, candidates):
                for idx, tensor in enumerate(candidate):
                    update[idx] += float(score) * tensor

            params = [param - lr * grad for param, grad in zip(params, update)]

            if step % 5 == 0 or step == n_steps - 1:
                metrics = evaluate_params(params)
                if metrics["train_accuracy"] >= best_metrics["train_accuracy"]:
                    best_params = [param.clone() for param in params]
                    best_metrics = metrics
                history.append(
                    {
                        "iter": step,
                        "loss": metrics["train_loss"],
                        "accuracy": metrics["train_accuracy"],
                        "test_accuracy": metrics["test_accuracy"],
                        "kind": "Dense"
                        if kind == "dense"
                        else f"Low-rank (r={rank})",
                    }
                )

        return {
            "kind": "Dense" if kind == "dense" else f"Low-rank (r={rank})",
            "history": history,
            "params": best_params,
            "metrics": best_metrics,
            "param_count": int(sum(param.numel() for param in params)),
            "random_values": random_value_count(params, kind=kind, rank=rank),
            "n_steps": n_steps,
            "n_pop": n_pop,
            "sigma": sigma,
            "lr": lr,
            "batch_size": batch_size,
        }


    dense_result = train_es(
        kind="dense", rank=rank_slider.value, n_steps=steps_slider.value
    )
    low_rank_result = train_es(
        kind="low_rank", rank=rank_slider.value, n_steps=steps_slider.value
    )
    history_df = pl.DataFrame(dense_result["history"] + low_rank_result["history"])

    history_df.plot.line("iter", "loss", "kind").properties(width="container")
    return (
        SmallConvNet,
        dense_result,
        evaluate_model,
        history_df,
        low_rank_result,
    )


@app.cell
def _(history_df):
    history_df.plot.line("iter", "accuracy", "kind").properties(width="container")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Gradient Baselines

    These runs use the same CNN, dataset subset, batch size, and number of update steps as the ES runs, but train with backprop using `SGD` and `Adam`.
    """)
    return


@app.cell
def _(
    F,
    SmallConvNet,
    X_train,
    evaluate_model,
    np,
    pl,
    steps_slider,
    torch,
    y_train,
):
    import marimo as _mo


    @_mo.cache()
    def train_backprop(opt_name="SGD", n_steps=160, batch_size=256, seed=7):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SmallConvNet()
        generator = torch.Generator().manual_seed(seed)
        history = []
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_metrics = evaluate_model(model)

        if opt_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for step in range(n_steps):
            batch_idx = torch.randint(
                0, X_train.shape[0], (batch_size,), generator=generator
            )
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            if step % 5 == 0 or step == n_steps - 1:
                metrics = evaluate_model(model)
                if metrics["train_accuracy"] >= best_metrics["train_accuracy"]:
                    best_state = {
                        k: v.detach().clone()
                        for k, v in model.state_dict().items()
                    }
                    best_metrics = metrics
                history.append(
                    {
                        "iter": step,
                        "loss": metrics["train_loss"],
                        "accuracy": metrics["train_accuracy"],
                        "test_accuracy": metrics["test_accuracy"],
                        "kind": opt_name,
                    }
                )

        model.load_state_dict(best_state)
        return {
            "kind": opt_name,
            "history": history,
            "metrics": best_metrics,
            "n_steps": n_steps,
            "batch_size": batch_size,
        }


    sgd_result = train_backprop("SGD", n_steps=steps_slider.value)
    adam_result = train_backprop("Adam", n_steps=steps_slider.value)
    grad_history_df = pl.DataFrame(sgd_result["history"] + adam_result["history"])

    grad_history_df.plot.line("iter", "loss", "kind").properties(width="container")
    return adam_result, grad_history_df, sgd_result


@app.cell
def _(grad_history_df):
    grad_history_df.plot.line("iter", "accuracy", "kind").properties(
        width="container"
    )
    return


@app.cell(hide_code=True)
def _(adam_result, mo, sgd_result):
    mo.md(f"""
    ## SGD / Adam Results

    | Optimizer | Best Train Accuracy | Matching Test Accuracy | Steps |
    |-----------|---------------------|------------------------|-------|
    | {sgd_result["kind"]} | {sgd_result["metrics"]["train_accuracy"]:.3f} | {sgd_result["metrics"]["test_accuracy"]:.3f} | {sgd_result["n_steps"]} |
    | {adam_result["kind"]} | {adam_result["metrics"]["train_accuracy"]:.3f} | {adam_result["metrics"]["test_accuracy"]:.3f} | {adam_result["n_steps"]} |
    """)
    return


if __name__ == "__main__":
    app.run()
