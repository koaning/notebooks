# /// script
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numba==0.64.0",
#     "numpy==2.4.3",
#     "polars==1.39.3",
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

    s = 15
    SHAPE = (s, s)

    data = np.random.randn(1000, SHAPE[0]) 
    w_true = np.random.randn(*SHAPE)
    y_true = data @ w_true
    return SHAPE, data, np, pl, y_true


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1, label="rank")
    slider
    return (slider,)


@app.cell
def _(SHAPE, data, mo, np, pl, slider, y_true):
    @mo.cache()
    def train_optimized(n_iter=1250, n_pop=150, lr=0.01, kind="default", rank=1):
        m, n = SHAPE
        history = []
        np.random.seed(41)
        w_curr = np.random.randn(m, n)

        for j in range(n_iter):
            if kind == "default":
                eps = np.random.randn(n_pop, m, n) * 0.02
            else:
                A = np.random.randn(n_pop, m, rank) * 0.02
                B = np.random.randn(n_pop, rank, n)
                eps = np.matmul(A, B)

            w_perturbed = w_curr[None, :, :] + eps

            y_pred = np.matmul(w_perturbed, data.T).transpose(0, 2, 1)
            scores = np.abs(y_pred - y_true[None, :, :]).sum(axis=(1, 2))

            s_std = (scores - np.mean(scores)) / (np.std(scores))

            update = np.tensordot(s_std, w_perturbed, axes=([0], [0]))
            w_curr -= lr * update

            if j % 5 == 0:
                history.append(dict(score=np.abs(data @ w_curr.T - y_true).sum(), iter=j, kind=kind))

        return history

    pl.DataFrame(
        train_optimized() + 
        train_optimized(kind="split", rank=slider.value)
    ).plot.line("iter", "score", "kind").properties(width="container")
    return


@app.cell
def _(np):
    import matplotlib.pylab as plt 

    plt.hist(np.random.randn(10000), 30)
    plt.show()
    return (plt,)


@app.cell
def _(np, plt):
    plt.hist(np.random.randn(10000) * np.random.randn(10000), 50)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix
    """)
    return


@app.cell
def _(SHAPE, data, np, y_true):
    def noise(kind="default", seed=None, rank=1):
        if seed:
            np.random.seed(seed)
        if kind == "default":
            return np.random.randn(*SHAPE) * 0.02
        if kind == "split":
            return np.random.randn(SHAPE[0], rank) * np.random.randn(rank, SHAPE[1]) * 0.02

    def score(w):
        return np.abs(data @ w - y_true).sum()


    def train(n_iter=1250, n_pop=150, lr=0.05, kind="default"):
        data = [] 
        np.random.seed(41)
        w_init = np.random.randn(*SHAPE)
        data.append(dict(score=score(w_init), iter=-1, kind=kind))

        for j in range(n_iter):
            scores = []
            weights = np.zeros([n_pop, *SHAPE])
            for i in range(n_pop):
                w_new = w_init + noise(kind=kind)
                weights[i] = w_new
                scores.append(score(w_new))

            for i, s in enumerate((np.array(scores) - np.mean(scores)) / np.std(scores)):
                w_init -= lr * s * weights[i]
            data.append(dict(score=score(w_init), iter=j, kind=kind))
        return data

    # pl.DataFrame(train() + train(kind="split")).plot.line("iter", "score", "kind").properties(width="container")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
