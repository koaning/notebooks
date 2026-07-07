# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "mnist1d==0.0.2.post1",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scikit-learn==1.7.2",
#     "torch==2.12.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import accuracy_score

    from mnist1d.data import make_dataset, get_dataset_args

    return (
        LogisticRegression,
        accuracy_score,
        alt,
        fetch_openml,
        get_dataset_args,
        make_dataset,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNIST-1D: where logistic regression falls apart

    On ordinary MNIST a plain linear classifier already scores **~92%** — so it
    can't really tell architectures apart. [MNIST-1D](https://www.alphaxiv.org/abs/2011.14439v4)
    (Greydanus & Kobak) is *designed* to spread models out: the informative bump
    is translated, scaled, sheared and noised, so absolute pixel positions carry
    little signal. Models that exploit spatial structure win; a linear model does not.

    The paper's headline numbers:

    | model | test accuracy |
    |---|---|
    | logistic regression | **32%** |
    | MLP | 68% |
    | GRU | 91% |
    | CNN | 94% |
    | human | 96% |

    We generate the data with the `mnist1d` package and confirm the first row.
    """)
    return


@app.cell
def _(get_dataset_args, make_dataset):
    args = get_dataset_args()
    args.num_samples = 20000  # bump it up; 80/20 split -> 16000 train / 4000 test
    data = make_dataset(args)

    X_train, y_train = data["x"], data["y"]
    X_test, y_test = data["x_test"], data["y_test"]
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What the data looks like

    A few examples per digit. Notice how much the same class varies — the
    characteristic shape slides left/right and tilts, so there is no fixed
    position a linear weight can latch onto.
    """)
    return


@app.cell
def _(X_train, alt, np, pd, y_train):
    rng = np.random.default_rng(0)
    example_rows = []
    for digit in range(10):
        idxs = np.where(y_train == digit)[0]
        for rep, idx in enumerate(rng.choice(idxs, size=1, replace=False)):
            for pos, val in enumerate(X_train[idx]):
                example_rows.append(
                    {"position": pos, "value": float(val), "digit": str(digit), "rep": str(rep)}
                )
    example_df = pd.DataFrame(example_rows)

    alt.Chart(example_df).mark_line().encode(
        x=alt.X("position:Q"),
        y=alt.Y("value:Q"),
        color=alt.Color("rep:N", legend=None),
        facet=alt.Facet("digit:N", columns=5, title=None),
    ).properties(width=120, height=90)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Logistic regression on MNIST-1D
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, accuracy_score, y_test, y_train):
    logreg_1d = LogisticRegression(max_iter=2000)
    logreg_1d.fit(X_train, y_train)
    acc_1d = accuracy_score(y_test, logreg_1d.predict(X_test))
    return (acc_1d,)


@app.cell(hide_code=True)
def _(acc_1d, mo):
    mo.md(f"""
    **Logistic regression on MNIST-1D: `{acc_1d:.1%}` test accuracy.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sanity check: the same model on ordinary MNIST

    To show the model is fine and it's the *dataset* that's hard, we run the same
    logistic regression on real MNIST (a subset, for speed).
    """)
    return


@app.cell
def _(LogisticRegression, accuracy_score, fetch_openml, np):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    Xm = mnist.data.astype("float32") / 255.0
    ym = mnist.target.astype(int)

    rng_m = np.random.default_rng(0)
    train_idx = rng_m.choice(len(Xm), size=10000, replace=False)
    test_idx = rng_m.choice(np.setdiff1d(np.arange(len(Xm)), train_idx), size=10000, replace=False)

    logreg_mnist = LogisticRegression(max_iter=200)
    logreg_mnist.fit(Xm[train_idx], ym[train_idx])
    acc_mnist = accuracy_score(ym[test_idx], logreg_mnist.predict(Xm[test_idx]))
    return (acc_mnist,)


@app.cell(hide_code=True)
def _(acc_1d, acc_mnist, mo):
    mo.md(f"""
    ## The contrast

    | dataset | logistic regression test accuracy |
    |---|---|
    | ordinary MNIST | **{acc_mnist:.1%}** |
    | MNIST-1D | **{acc_1d:.1%}** |

    Same model, same code — but MNIST-1D drops it from "looks solved" down toward
    chance-ish territory. That gap is the whole point of the benchmark: it leaves
    room for inductive biases (convolutions, recurrence) to matter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Epoch-wise double descent

    Train a deliberately over-parameterized MLP on a small, **label-noised** slice of
    MNIST-1D and watch the *test* error as training proceeds. Classical bias–variance
    says test error should fall and then flatten. Instead you can see **epoch-wise
    double descent**: test error drops, then *rises* as the network starts memorizing
    the corrupted labels, then falls again with more training.

    The effect is subtle on a dataset this small — turn the label-noise slider up and
    train longer to bring it out. (The model-*size* version from the paper is the more
    dramatic one; that's the natural next step.)
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device, nn, torch


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(0.0, 0.5, value=0.3, step=0.05, label="label noise")
    epochs_slider = mo.ui.slider(200, 2000, value=1200, step=100, label="epochs")
    width_slider = mo.ui.slider(8, 1024, value=512, step=8, label="hidden width")
    depth_slider = mo.ui.slider(1, 5, value=2, step=1, label="hidden depth")
    activation = mo.ui.dropdown(["relu", "sigmoid"], value="relu", label="activation")
    train_button = mo.ui.run_button(label="Train")

    mo.vstack(
        [
            mo.hstack([noise_slider, epochs_slider], justify="start"),
            mo.hstack([width_slider, depth_slider, activation], justify="start"),
            train_button,
        ]
    )
    return (
        activation,
        depth_slider,
        epochs_slider,
        noise_slider,
        train_button,
        width_slider,
    )


@app.cell(hide_code=True)
def _(noise_slider, np, y_train):
    dd_n = 8000  # small slice so the model can interpolate (and memorize the noise)
    rng_dd = np.random.default_rng(0)
    y_noisy = y_train[:dd_n].copy()
    flip_mask = rng_dd.random(dd_n) < noise_slider.value
    y_noisy[flip_mask] = rng_dd.integers(0, 10, flip_mask.sum())
    return dd_n, y_noisy


@app.cell(hide_code=True)
def _(
    X_test,
    X_train,
    activation,
    dd_n,
    depth_slider,
    device,
    epochs_slider,
    mo,
    nn,
    pd,
    torch,
    train_button,
    width_slider,
    y_noisy,
    y_test,
):
    torch.manual_seed(0)

    if not train_button.value:
        dd_df = None
    else:
        Xtr = torch.tensor(X_train[:dd_n], dtype=torch.float32, device=device)
        ytr = torch.tensor(y_noisy, dtype=torch.long, device=device)
        Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
        yte = torch.tensor(y_test, dtype=torch.long, device=device)

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        act_layer = activations[activation.value]
        layers = []
        in_dim = 40
        for _ in range(depth_slider.value):
            layers.append(nn.Linear(in_dim, width_slider.value))
            layers.append(act_layer())
            in_dim = width_slider.value
        layers.append(nn.Linear(in_dim, 10))
        model = nn.Sequential(*layers).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        batch_size = 128
        n_epochs = epochs_slider.value

        records = []
        with mo.status.progress_bar(total=n_epochs) as bar:
            for ep in range(n_epochs):
                perm = torch.randperm(dd_n, device=device)
                model.train()
                for i in range(0, dd_n, batch_size):
                    batch_idx = perm[i : i + batch_size]
                    optimizer.zero_grad()
                    loss_fn(model(Xtr[batch_idx]), ytr[batch_idx]).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    tr_err = 1 - (model(Xtr).argmax(1) == ytr).float().mean().item()
                    te_err = 1 - (model(Xte).argmax(1) == yte).float().mean().item()
                records.append({"epoch": ep + 1, "train error": tr_err, "test error": te_err})
                bar.update()

        dd_df = pd.DataFrame(records)

    mo.md("*Set the knobs, then click **Train**.*") if dd_df is None else None
    return (dd_df,)


@app.cell(hide_code=True)
def _(alt, dd_df, mo):
    mo.stop(dd_df is None, mo.md("*Train a model above to see the double-descent curve.*"))

    dd_long = dd_df.melt(
        "epoch", ["train error", "test error"], var_name="series", value_name="error"
    )
    alt.Chart(dd_long).mark_line().encode(
        x=alt.X("epoch:Q", scale=alt.Scale(type="log"), title="epoch (log scale)"),
        y=alt.Y("error:Q", scale=alt.Scale(zero=False), title="error rate"),
        color=alt.Color("series:N", title=None),
    ).properties(width=560, height=320)
    return


if __name__ == "__main__":
    app.run()
