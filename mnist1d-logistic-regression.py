# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib>=3.10.0",
#     "mnist1d>=0.0.2",
#     "scikit-learn>=1.7.0",
#     "torch>=2.7.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from mnist1d.data import get_dataset, get_dataset_args
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        Path,
        accuracy_score,
        classification_report,
        get_dataset,
        get_dataset_args,
        mo,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A linear baseline for MNIST-1D

    This starter notebook loads the frozen **MNIST-1D** dataset from
    [Greydanus & Kobak (2024)](https://arxiv.org/abs/2011.14439) and fits a
    scikit-learn logistic regression classifier.

    MNIST-1D contains one-dimensional, digit-like signals. The default frozen
    split has 4,000 training examples, 1,000 test examples, 40 features, and
    10 classes. The paper reports about **32% test accuracy** for a linear
    classifier, leaving substantial room for models with nonlinear and spatial
    inductive biases.
    """)
    return


@app.cell(hide_code=True)
def _(Path, get_dataset, get_dataset_args):
    dataset_path = Path(".context") / "mnist1d_data.pkl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_args = get_dataset_args()
    dataset = get_dataset(
        dataset_args,
        path=str(dataset_path),
        download=True,
        verbose=False,
    )

    X_train = dataset["x"]
    y_train = dataset["y"]
    X_test = dataset["x_test"]
    y_test = dataset["y_test"]
    time_axis = dataset["t"]
    return X_test, X_train, time_axis, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What do the signals look like?

    Each row below is one example from a class. Translation, scaling, smoothing,
    shear, and noise make the task difficult for a purely linear model.
    """)
    return


@app.cell
def _(X_train, np, plt, time_axis, y_train):
    figure_examples, axes = plt.subplots(
        5, 2, figsize=(6, 6), sharex=True, sharey=True
    )
    for digit, _axis in enumerate(axes.flat):
        example_index = np.flatnonzero(y_train == digit)[0]
        _axis.plot(time_axis, X_train[example_index], linewidth=2)
        _axis.set_title(f"Class {digit}")
        _axis.grid(alpha=0.2)

    figure_examples.suptitle("One MNIST-1D training example per class")
    figure_examples.supxlabel("Position")
    figure_examples.supylabel("Signal value")
    figure_examples.tight_layout()
    figure_examples
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit logistic regression

    `StandardScaler` estimates its statistics from the training split only.
    Adjust the inverse regularization strength `C` to refit the model.
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, accuracy_score, y_test, y_train):
    model = LogisticRegression(
            max_iter=2_000,
            random_state=42,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return model, test_predictions


@app.cell
def _(ConfusionMatrixDisplay, X_test, model, plt, y_test):
    figure_confusion, _axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        colorbar=False,
        ax=_axis,
    )
    _axis.set_title("Test-set confusion matrix")
    figure_confusion.tight_layout()
    figure_confusion
    return


@app.cell
def _(classification_report, test_predictions, y_test):
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Double descent

    The paper observes double descent after corrupting 15% of the training
    labels. This compact version uses full-batch Adam and fewer hidden widths
    so the experiment finishes quickly.
    """)
    return


@app.cell
def _(X_test, X_train, nn, np, torch, y_test, y_train):
    _rng = np.random.default_rng(42)
    noisy_labels = y_train.copy()
    _noise_mask = _rng.random(len(noisy_labels)) < 0.15
    noisy_labels[_noise_mask] = _rng.integers(0, 10, _noise_mask.sum())

    _x_train = torch.tensor(X_train, dtype=torch.float32)
    _y_train = torch.tensor(noisy_labels, dtype=torch.long)
    _x_test = torch.tensor(X_test, dtype=torch.float32)
    _y_test = torch.tensor(y_test, dtype=torch.long)

    hidden_widths = np.array([2, 5, 10, 20, 35, 50, 75, 110, 160, 230, 300])

    def fit_mlp(_width):
        torch.manual_seed(42)
        _network = nn.Sequential(
            nn.Linear(40, _width),
            nn.ReLU(),
            nn.Linear(_width, _width),
            nn.ReLU(),
            nn.Linear(_width, 10),
        )
        _optimizer = torch.optim.Adam(_network.parameters(), lr=0.01)

        for _ in range(500):
            _optimizer.zero_grad()
            _loss = nn.functional.cross_entropy(_network(_x_train), _y_train)
            _loss.backward()
            _optimizer.step()

        with torch.no_grad():
            _train_error = (
                (_network(_x_train).argmax(1) != _y_train).float().mean().item()
            )
            _test_error = (
                (_network(_x_test).argmax(1) != _y_test).float().mean().item()
            )
        return 100 * _train_error, 100 * _test_error

    _errors = np.array([fit_mlp(_width) for _width in hidden_widths])
    train_errors = _errors[:, 0]
    test_errors = _errors[:, 1]
    return hidden_widths, test_errors, train_errors


@app.cell
def _(hidden_widths, np, plt, test_errors, train_errors):
    _figure, _axis = plt.subplots(figsize=(7, 4))
    _axis.plot(hidden_widths, train_errors, "o--", label="Train error")
    _axis.plot(hidden_widths, test_errors, "o-", label="Test error")

    _interpolating = np.flatnonzero(train_errors < 1)
    if len(_interpolating):
        _threshold = hidden_widths[_interpolating[0]]
        _axis.axvline(
            _threshold,
            color="black",
            linestyle=":",
            label=f"Interpolation threshold ({_threshold})",
        )

    _axis.set(
        xscale="log",
        xlabel="Hidden layer width",
        ylabel="Classification error (%)",
        title="Double descent with 15% label noise",
    )
    _axis.legend()
    _axis.grid(alpha=0.2)
    _figure.tight_layout()
    _figure
    return


if __name__ == "__main__":
    app.run()
