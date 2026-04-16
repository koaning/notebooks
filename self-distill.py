# /// script
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "numpy==2.4.4",
#     "pandas==3.0.2",
#     "plotly==6.7.0",
#     "scikit-learn==1.8.0",
#     "torch==2.11.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns", sql_output="polars")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.datasets import make_classification
    import altair as alt
    import pandas as pd

    return alt, make_classification, nn, np, pd, torch


@app.cell(hide_code=True)
def _(alt, make_classification, nn, np, pd, torch):
    import functools


    @functools.cache
    def run_experiment(n_classes, signal_pct, n_epochs, n_samples):
        n_features = 20

        X, y_true = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
        )

        split = int(0.7 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train_true, y_test_true = y_true[:split], y_true[split:]

        rng = np.random.default_rng(42)
        mask = rng.random(len(y_train_true)) < signal_pct
        y_train_noisy = np.where(
            mask, y_train_true, rng.integers(0, n_classes, size=len(y_train_true))
        )

        mask_test = rng.random(len(y_test_true)) < signal_pct
        y_test_noisy = np.where(
            mask_test, y_test_true, rng.integers(0, n_classes, size=len(y_test_true))
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_noisy_t = torch.tensor(y_train_noisy, dtype=torch.long)

        model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        history = {
            "epoch": [],
            "train_acc_noisy": [],
            "train_acc_true": [],
            "test_acc_true": [],
            "test_acc_noisy": [],
        }

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = loss_fn(logits, y_noisy_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                train_preds = model(X_train_t).argmax(dim=1).numpy()
                test_preds = model(X_test_t).argmax(dim=1).numpy()

            history["epoch"].append(epoch + 1)
            history["train_acc_noisy"].append(float((train_preds == y_train_noisy).mean()))
            history["train_acc_true"].append(float((train_preds == y_train_true).mean()))
            history["test_acc_true"].append(float((test_preds == y_test_true).mean()))
            history["test_acc_noisy"].append(float((test_preds == y_test_noisy).mean()))

        return history


    def make_chart(history, title=""):
        df_hist = pd.DataFrame(history).melt(
            id_vars="epoch",
            value_vars=["test_acc_true", "train_acc_true", "train_acc_noisy", "test_acc_noisy"],
            var_name="metric",
            value_name="accuracy",
        )
        label_map = {
            "test_acc_true": "Test accuracy (true labels)",
            "train_acc_true": "Train accuracy (true labels)",
            "train_acc_noisy": "Train accuracy (noisy labels)",
            "test_acc_noisy": "Test accuracy (noisy labels)",
        }
        df_hist["metric"] = df_hist["metric"].map(label_map)

        return (
            alt.Chart(df_hist)
            .mark_line()
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("accuracy:Q", title="Accuracy"),
                color=alt.Color("metric:N", title="Metric"),
            )
            .properties(title=title, width=450, height=350)
        )

    return make_chart, run_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Self-Distillation: Learning Signal from Noise

    An LLM can generate predictions that are partly correct and partly random noise.
    If you train a model on these noisy predictions, it can actually **improve** its accuracy. This is an idea presented in [the Embarrassingly Simple Self-Distillation Improves Code Generation](https://www.alphaxiv.org/abs/2604.01193) paper.

    To help with the intuition we run a different but related setup with a simple feed forward network: the noise is scattered uniformly across classes, but the signal is consistent. When the model trains on its own noisy outputs, it can only learn from the consistent part — the signal.

    Use the sliders below to explore this effect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <svg viewBox="0 0 900 345" xmlns="http://www.w3.org/2000/svg" style="max-width:900px; width:100%; font-family: system-ui, sans-serif;">

      <!-- True Labels: x=350 y=20 w=200 h=50 -->
      <rect x="350" y="20" width="200" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="450" y="52" text-anchor="middle" font-size="14" font-weight="bold" fill="#444">True Labels</text>

      <!-- Train True: x=130 y=100 w=200 h=50 -->
      <rect x="130" y="100" width="200" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="230" y="132" text-anchor="middle" font-size="13" fill="#444">Train (true labels)</text>
      <circle cx="118" cy="125" r="6" fill="#72b7b2"/>

      <!-- Test True: x=570 y=100 w=200 h=50 -->
      <rect x="570" y="100" width="200" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="670" y="132" text-anchor="middle" font-size="13" fill="#444">Test (true labels)</text>
      <circle cx="558" cy="125" r="6" fill="#f58518"/>

      <!-- Split: left side of True Labels (350,45) curving down to top of Train True (230,100) -->
      <path d="M 350,45 C 280,45 230,55 230,100" fill="none" stroke="#aaa" stroke-width="1.5" marker-end="url(#arr3)"/>
      <text x="265" y="45" text-anchor="middle" font-size="11" fill="#999">train split</text>

      <!-- Split: right side of True Labels (550,45) curving down to top of Test True (670,100) -->
      <path d="M 550,45 C 620,45 670,55 670,100" fill="none" stroke="#aaa" stroke-width="1.5" marker-end="url(#arr3)"/>
      <text x="635" y="45" text-anchor="middle" font-size="11" fill="#999">test split</text>

      <!-- Train Noisy: x=130 y=200 w=200 h=50 -->
      <rect x="130" y="200" width="200" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="230" y="232" text-anchor="middle" font-size="13" fill="#444">Train (noisy labels)</text>
      <circle cx="118" cy="225" r="6" fill="#e45756"/>

      <!-- Test Noisy: x=570 y=200 w=200 h=50 -->
      <rect x="570" y="200" width="200" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="670" y="232" text-anchor="middle" font-size="13" fill="#444">Test (noisy labels)</text>
      <circle cx="558" cy="225" r="6" fill="#4c78a8"/>

      <!-- Noise arrows -->
      <path d="M 230,150 C 230,170 230,180 230,200" fill="none" stroke="#aaa" stroke-width="1.5" marker-end="url(#arr3)"/>
      <path d="M 670,150 C 670,170 670,180 670,200" fill="none" stroke="#aaa" stroke-width="1.5" marker-end="url(#arr3)"/>
      <text x="275" y="180" font-size="11" fill="#999">add noise</text>
      <text x="715" y="180" font-size="11" fill="#999">add noise</text>

      <!-- Neural Network: x=370 y=280 w=160 h=50 -->
      <rect x="370" y="280" width="160" height="50" rx="10" fill="none" stroke="#999" stroke-width="1.5"/>
      <text x="450" y="312" text-anchor="middle" font-size="14" font-weight="bold" fill="#444">Neural Network</text>

      <!-- Train arrow: bottom of Train Noisy (230,250) curving to left of NN (370,305) -->
      <path d="M 230,250 C 230,290 320,305 370,305" fill="none" stroke="#aaa" stroke-width="1.5" marker-end="url(#arr3)"/>
      <text x="270" y="285" font-size="11" fill="#999">train</text>

      <defs>
        <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#aaa"/>
        </marker>
      </defs>
    </svg>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _left = mo.md("""
    **Experiment A**

    | | |
    |---|---|
    | Number of classes | {n_classes_a} |
    | Signal % | {signal_pct_a} |
    | Training epochs | {n_epochs_a} |
    | Number of samples | {n_samples_a} |
    """)

    _right = mo.md("""
    **Experiment B**

    | | |
    |---|---|
    | Number of classes | {n_classes_b} |
    | Signal % | {signal_pct_b} |
    | Training epochs | {n_epochs_b} |
    | Number of samples | {n_samples_b} |
    """)

    form = (
        mo.hstack([_left, _right], gap=2, widths="equal")
        .batch(
            n_classes_a=mo.ui.slider(2, 250, value=5),
            signal_pct_a=mo.ui.slider(5, 95, step=5, value=30),
            n_epochs_a=mo.ui.slider(1, 200, value=20),
            n_samples_a=mo.ui.slider(1000, 50000, step=1000, value=2000),
            n_classes_b=mo.ui.slider(2, 250, value=5),
            signal_pct_b=mo.ui.slider(5, 95, step=5, value=30),
            n_epochs_b=mo.ui.slider(1, 200, value=20),
            n_samples_b=mo.ui.slider(1000, 50000, step=1000, value=2000),
        )
        .form(submit_button_label="Run experiments")
    )

    form
    return (form,)


@app.cell(hide_code=True)
def _(form, make_chart, mo, run_experiment):
    mo.stop(form.value is None, mo.md("*Configure the sliders and click **Run experiments**.*"))

    v = form.value

    history_a = run_experiment(
        v["n_classes_a"],
        v["signal_pct_a"] / 100.0,
        v["n_epochs_a"],
        v["n_samples_a"],
    )
    history_b = run_experiment(
        v["n_classes_b"],
        v["signal_pct_b"] / 100.0,
        v["n_epochs_b"],
        v["n_samples_b"],
    )

    (make_chart(history_a, "Experiment A") | make_chart(history_b, "Experiment B")).resolve_scale(
        color="shared"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What's happening?

    The noise is uniformly distributed across all classes, so it cancels out in aggregate.
    The signal always points to the correct class, so it accumulates. The neural network learns the only
    consistent pattern — the signal — and ignores the noise.

    This is the same mechanism behind **self-distillation** in LLMs: when a model generates noisy outputs
    and is trained on them, the consistent signal survives while the random gibberish washes out.
    """)
    return


if __name__ == "__main__":
    app.run()
