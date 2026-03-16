# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cvxpy",
#     "marimo",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from sklearn.datasets import fetch_openml
    from itertools import combinations

    return combinations, cp, fetch_openml, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNIST Linear Separability

    Can you draw a straight line (well, a hyperplane) between two handwritten digits?

    This notebook checks pairwise linear separability of MNIST digit classes using a
    [CVXPY](https://www.cvxpy.org/) feasibility problem. If a separating hyperplane exists
    such that $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ for all samples, the pair is
    linearly separable.

    Based on the approach from [this paper](https://www.alphaxiv.org/abs/2603.12850).
    """)
    return


@app.cell
def _(fetch_openml, np):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_all = mnist.data.astype("float32") / 255.0
    y_all = mnist.target.astype(int)

    # Standard train/test split
    X_train = X_all[:60000]
    y_train = y_all[:60000]
    X_test = X_all[60000:]
    y_test = y_all[60000:]

    return X_test, X_train, y_test, y_train


@app.cell
def _(cp, np):
    def is_linearly_separable(digit_a, digit_b, X, y):
        """Check if two digit classes are linearly separable via a feasibility LP."""
        mask = (y == digit_a) | (y == digit_b)
        X_sub = X[mask]
        y_sub = np.where(y[mask] == digit_a, 1.0, -1.0)

        n_features = X_sub.shape[1]
        w = cp.Variable(n_features)
        b = cp.Variable()

        constraints = [cp.multiply(y_sub, X_sub @ w + b) >= 1]
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.CLARABEL)
        return prob.status == cp.OPTIMAL

    return (is_linearly_separable,)


@app.cell
def _(np):
    def triangle_html(mat, title):
        """Render upper-triangle separability matrix as a styled HTML table."""
        _green = "#d4edda"
        _red = "#f8d7da"
        _grey = "#f0f0f0"
        _parts = []
        _parts.append(f"<h3 style='margin:0 0 8px 0'>{title}</h3>")
        _parts.append("<table style='border-collapse:collapse;font-family:monospace;font-size:14px'>")
        _parts.append("<tr><th style='padding:6px 10px'></th>")
        for _d in range(10):
            _parts.append(f"<th style='padding:6px 10px'>{_d}</th>")
        _parts.append("</tr>")
        for _i in range(10):
            _parts.append("<tr>")
            _parts.append(f"<th style='padding:6px 10px'>{_i}</th>")
            for _j in range(10):
                if _j <= _i:
                    _parts.append(f"<td style='padding:6px 10px;background:{_grey};text-align:center'>·</td>")
                else:
                    _bg = _green if mat[_i, _j] else _red
                    _label = "✓" if mat[_i, _j] else "✗"
                    _parts.append(f"<td style='padding:6px 10px;background:{_bg};text-align:center;font-weight:bold'>{_label}</td>")
            _parts.append("</tr>")
        _parts.append("</table>")
        _n = int(np.sum(np.triu(mat, k=1)))
        _parts.append(f"<p><b>{_n} / 45</b> pairs separable</p>")
        return "".join(_parts)

    return (triangle_html,)


@app.cell
def _(X_test, X_train, combinations, is_linearly_separable, mo, np, y_test, y_train):
    mo.output.append(mo.md("Computing pairwise separability (CVXPY) for all 45 digit pairs ..."))

    cvx_train = np.full((10, 10), True)
    cvx_test = np.full((10, 10), True)

    for _a, _b in combinations(range(10), 2):
        cvx_train[_a, _b] = is_linearly_separable(_a, _b, X_train, y_train)
        cvx_test[_a, _b] = is_linearly_separable(_a, _b, X_test, y_test)

    return cvx_test, cvx_train


@app.cell(hide_code=True)
def _(cvx_test, cvx_train, mo, triangle_html):
    mo.Html(
        f"<div style='display:flex;gap:40px;flex-wrap:wrap'>"
        f"<div>{triangle_html(cvx_train, 'Train set (60k)')}</div>"
        f"<div>{triangle_html(cvx_test, 'Test set (10k)')}</div>"
        f"</div>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic Regression as a faster alternative

    Solving a full feasibility LP is rigorous but slow. A simpler proxy: fit a logistic regression
    on each digit pair and check if it achieves **100% training accuracy**. If it does, the data is
    linearly separable. This runs much faster because sklearn's solver is heavily optimized for
    this exact problem.
    """)
    return


@app.cell
def _(X_test, X_train, combinations, np, y_test, y_train):
    from sklearn.linear_model import LogisticRegression

    def _check_pair(X, y, a, b):
        _mask = (y == a) | (y == b)
        _clf = LogisticRegression(max_iter=5000, C=1e10, solver="lbfgs")
        _clf.fit(X[_mask], y[_mask])
        return _clf.score(X[_mask], y[_mask]) == 1.0

    lr_train = np.full((10, 10), True)
    lr_test = np.full((10, 10), True)
    for _a, _b in combinations(range(10), 2):
        lr_train[_a, _b] = _check_pair(X_train, y_train, _a, _b)
        lr_test[_a, _b] = _check_pair(X_test, y_test, _a, _b)

    return lr_test, lr_train


@app.cell(hide_code=True)
def _(lr_test, lr_train, mo, triangle_html):
    mo.Html(
        f"<div style='display:flex;gap:40px;flex-wrap:wrap'>"
        f"<div>{triangle_html(lr_train, 'Train set (60k)')}</div>"
        f"<div>{triangle_html(lr_test, 'Test set (10k)')}</div>"
        f"</div>"
    )
    return


if __name__ == "__main__":
    app.run()
