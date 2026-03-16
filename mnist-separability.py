# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cvxpy",
#     "marimo",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "scikit-learn",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LogisticRegression, Perceptron
    from itertools import combinations

    return LogisticRegression, combinations, cp, fetch_openml, mo, np, warnings


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
def _(fetch_openml):
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
        prob.solve()
        return prob.status == cp.OPTIMAL

    return


@app.cell
def _(np):
    def triangle_html(mat, title):
        """Render upper-triangle separability matrix as a styled HTML table."""
        green = "#d4edda"
        red = "#f8d7da"
        grey = "#f0f0f0"
        parts = []
        parts.append(f"<h3 style='margin:0 0 8px 0'>{title}</h3>")
        parts.append(
            "<table style='border-collapse:collapse;font-family:monospace;font-size:14px'>"
        )
        parts.append("<tr><th style='padding:6px 10px'></th>")
        for d in range(10):
            parts.append(f"<th style='padding:6px 10px'>{d}</th>")
        parts.append("</tr>")
        for i in range(10):
            parts.append("<tr>")
            parts.append(f"<th style='padding:6px 10px'>{i}</th>")
            for j in range(10):
                if j <= i:
                    parts.append(
                        f"<td style='padding:6px 10px;background:{grey};text-align:center'>·</td>"
                    )
                else:
                    bg = green if mat[i, j] else red
                    label = "✓" if mat[i, j] else "✗"
                    parts.append(
                        f"<td style='padding:6px 10px;background:{bg};text-align:center;font-weight:bold'>{label}</td>"
                    )
            parts.append("</tr>")
        parts.append("</table>")
        n = int(np.sum(np.triu(mat, k=1)))
        parts.append(f"<p><b>{n} / 45</b> pairs separable</p>")
        return "".join(parts)

    return (triangle_html,)


@app.cell
def _(np):
    cvx_train = np.full((10, 10), True)
    cvx_test = np.full((10, 10), True)

    # for _a, _b in combinations(range(10), 2):
    #     cvx_train[_a, _b] = is_linearly_separable(_a, _b, X_train, y_train)
    #     cvx_test[_a, _b] = is_linearly_separable(_a, _b, X_test, y_test)
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
def _(
    LogisticRegression,
    X_test,
    X_train,
    combinations,
    np,
    warnings,
    y_test,
    y_train,
):
    # sklearn bug: C=np.inf internally sets penalty=None, which then warns about C not being default
    warnings.filterwarnings(
        "ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters"
    )


    def check_pair(X, y, a, b):
        mask = (y == a) | (y == b)
        clf = LogisticRegression(max_iter=10000, C=np.inf, solver="lbfgs")
        clf.fit(X[mask], y[mask])
        return (clf.predict(X[mask]) == y[mask]).all()


    lr_train = np.full((10, 10), True)
    lr_test = np.full((10, 10), True)
    for _a, _b in combinations(range(10), 2):
        lr_train[_a, _b] = check_pair(X_train, y_train, _a, _b)
        lr_test[_a, _b] = check_pair(X_test, y_test, _a, _b)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing coefficients: CVXPY vs sklearn (4 vs 8)

    Do the two methods find the same separating hyperplane? Below we fit both on the
    4 vs 8 pair and visualize the learned coefficients as 28×28 heatmaps side by side.
    """)
    return


@app.cell
def _(LogisticRegression, X_train, cp, np, warnings, y_train):
    mask_4v8 = (y_train == 5) | (y_train == 6)
    X_sub = X_train[mask_4v8]
    y_sub_signed = np.where(y_train[mask_4v8] == 4, 1.0, -1.0)

    # CVXPY feasibility
    w = cp.Variable(X_sub.shape[1])
    b = cp.Variable()
    prob = cp.Problem(cp.Minimize(0), [cp.multiply(y_sub_signed, X_sub @ w + b) >= 1])
    prob.solve()
    cvx_coef_4v8 = w.value
    cvx_bias_4v8 = b.value

    # Logistic Regression
    warnings.filterwarnings(
        "ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters"
    )
    clf_4v8 = LogisticRegression(max_iter=10000, C=np.inf, solver="lbfgs")
    clf_4v8.fit(X_sub, y_train[mask_4v8])
    lr_coef_4v8 = clf_4v8.coef_[0]
    return clf_4v8, cvx_bias_4v8, cvx_coef_4v8, lr_coef_4v8


@app.cell(hide_code=True)
def _(cvx_coef_4v8, lr_coef_4v8, mo, np):
    def _coef_heatmap(coef, title):
        grid = coef.reshape(28, 28)
        abs_max = np.abs(grid).max()
        parts = []
        parts.append(f"<div><h3 style='margin:0 0 8px 0'>{title}</h3>")
        parts.append("<p style='margin:0 0 4px 0;font-size:12px'>Blue = class 8, Red = class 4</p>")
        parts.append("<table style='border-collapse:collapse'>")
        for i in range(28):
            parts.append("<tr>")
            for j in range(28):
                v = grid[i, j]
                alpha = f"{abs(v) / abs_max:.2f}"
                color = f"rgba(0,100,200,{alpha})" if v > 0 else f"rgba(200,50,50,{alpha})"
                parts.append(f"<td style='width:12px;height:12px;background:{color}'></td>")
            parts.append("</tr>")
        parts.append("</table></div>")
        return "".join(parts)


    mo.Html(
        f"<div style='display:flex;gap:40px;flex-wrap:wrap;font-family:monospace'>"
        f"{_coef_heatmap(cvx_coef_4v8, 'CVXPY')}"
        f"{_coef_heatmap(lr_coef_4v8, 'Logistic Regression')}"
        f"</div>"
    )
    return


@app.cell(hide_code=True)
def _(X_train, clf_4v8, cvx_bias_4v8, cvx_coef_4v8, mo, np, y_train):
    mask_4v8_check = (y_train == 4) | (y_train == 8)
    X_check = X_train[mask_4v8_check]
    y_check = y_train[mask_4v8_check]

    # CVXPY predictions: sign(w^T x + b), mapping +1 -> 4, -1 -> 8
    cvx_scores = X_check @ cvx_coef_4v8 + cvx_bias_4v8
    cvx_preds = np.where(cvx_scores >= 0, 4, 8)
    cvx_errors = int(np.sum(cvx_preds != y_check))

    # Logistic Regression predictions
    lr_preds = clf_4v8.predict(X_check)
    lr_errors = int(np.sum(lr_preds != y_check))
    lr_wrong_mask = lr_preds != y_check

    n_total = len(y_check)

    # Show misclassified digits from LR as small HTML pixel grids
    wrong_indices = np.where(lr_wrong_mask)[0]
    digit_parts = []
    for idx in wrong_indices[:12]:
        img = X_check[idx].reshape(28, 28)
        true_label = int(y_check[idx])
        pred_label = int(lr_preds[idx])
        rows = []
        for row in range(28):
            cells = []
            for col in range(28):
                v = img[row, col]
                cells.append(
                    f"<td style='width:4px;height:4px;padding:0;background:rgba(0,0,0,{v:.2f})'></td>"
                )
            rows.append(f"<tr>{''.join(cells)}</tr>")
        table = f"<table style='border-collapse:collapse'>{''.join(rows)}</table>"
        digit_parts.append(
            f"<div style='text-align:center'>{table}"
            f"<div style='font-size:11px;margin-top:2px'>true={true_label} pred={pred_label}</div></div>"
        )

    misclassified_html = ""
    if digit_parts:
        misclassified_html = (
            f"<h4 style='margin:16px 0 8px 0'>Misclassified by Logistic Regression</h4>"
            f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{''.join(digit_parts)}</div>"
        )

    mo.Html(
        f"<div style='font-family:monospace'>"
        f"<h3 style='margin:0 0 8px 0'>Classification accuracy on 4 vs 8 (train set)</h3>"
        f"<table style='border-collapse:collapse;font-size:14px'>"
        f"<tr><th style='padding:6px 12px;text-align:left'>Method</th>"
        f"<th style='padding:6px 12px'>Errors</th>"
        f"<th style='padding:6px 12px'>Accuracy</th></tr>"
        f"<tr><td style='padding:6px 12px'>CVXPY feasibility</td>"
        f"<td style='padding:6px 12px;text-align:center'>{cvx_errors}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{100 * (1 - cvx_errors / n_total):.2f}%</td></tr>"
        f"<tr><td style='padding:6px 12px'>Logistic Regression</td>"
        f"<td style='padding:6px 12px;text-align:center'>{lr_errors}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{100 * (1 - lr_errors / n_total):.2f}%</td></tr>"
        f"</table>"
        f"{misclassified_html}"
        f"</div>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostic: why do CVXPY and LR disagree?

    CVXPY solves a feasibility problem (`Minimize(0)`) — it just checks if *any* `w, b` satisfies
    $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$. But CLARABEL is an interior-point solver with
    numerical tolerances. It might declare `OPTIMAL` even when some constraints are only
    *approximately* satisfied.

    Below we take the CVXPY solution for a "separable" pair and check how many margin
    constraints are actually violated.
    """)
    return


@app.cell
def _(X_train, cp, mo, np, y_train):
    # Pick a pair where the methods might disagree, e.g. 3 vs 5
    digit_a, digit_b = 5, 6
    mask = (y_train == digit_a) | (y_train == digit_b)
    X_diag = X_train[mask]
    y_diag = np.where(y_train[mask] == digit_a, 1.0, -1.0)

    w_diag = cp.Variable(X_diag.shape[1])
    b_diag = cp.Variable()
    prob_diag = cp.Problem(cp.Minimize(0), [cp.multiply(y_diag, X_diag @ w_diag + b_diag) >= 1])
    prob_diag.solve()

    if w_diag.value is not None:
        margins = y_diag * (X_diag @ w_diag.value + b_diag.value)
        n_violated = int(np.sum(margins < 1.0))
        n_wrong_side = int(np.sum(margins < 0.0))
        min_margin = float(np.min(margins))

        result = mo.md(f"""
        ### Pair {digit_a} vs {digit_b} — CVXPY status: `{prob_diag.status}`

        - Solver tolerance: `{prob_diag.solver_stats.solve_time:.4f}s`
        - Constraints with margin < 1.0: **{n_violated}** / {len(margins)}
        - Constraints with margin < 0.0 (wrong side): **{n_wrong_side}** / {len(margins)}
        - Minimum margin: **{min_margin:.6f}**

        If `n_violated > 0` but the solver says `OPTIMAL`, CLARABEL is declaring feasibility
        within its numerical tolerance — the data may not be *truly* linearly separable.
        """)
    else:
        result = mo.md(f"""
        ### Pair {digit_a} vs {digit_b} — CVXPY status: `{prob_diag.status}`

        The solver did not find a separating hyperplane — this pair is not linearly separable.
        """)
    result
    return b_diag, mask, w_diag


@app.cell
def _(LogisticRegression, X_train, b_diag, mask, mo, np, w_diag, y_train):
    # You will only find the hyperplane if you set tol to be super low and penalty to None!
    lr = LogisticRegression(max_iter=1_000, tol=1e-30, penalty=None).fit(
        X_train[mask], y_train[mask]
    )
    lr_acc = np.mean(lr.predict(X_train[mask]) == y_train[mask])
    lr_coef = lr.coef_.copy()
    lr_norm = np.linalg.norm(lr_coef)

    cvx_w = -w_diag.value.reshape(1, -1)
    cvx_b = np.array([-b_diag.value])
    cvx_norm = np.linalg.norm(cvx_w)
    diff_coef = cvx_w.flatten() - lr_coef.flatten()

    lr.coef_ = cvx_w
    lr.intercept_ = cvx_b
    cvx_acc = np.mean(lr.predict(X_train[mask]) == y_train[mask])

    mo.Html(
        f"<div style='font-family:monospace'>"
        f"<h3>LR vs CVXPY coefficients (5 vs 6)</h3>"
        f"<table style='font-size:14px;border-collapse:collapse'>"
        f"<tr><th style='padding:4px 12px;text-align:left'>Method</th>"
        f"<th style='padding:4px 12px'>||w||</th>"
        f"<th style='padding:4px 12px'>Accuracy</th>"
        f"<th style='padding:4px 12px'>n_iter</th></tr>"
        f"<tr><td style='padding:4px 12px'>Logistic Regression</td>"
        f"<td style='padding:4px 12px;text-align:center'>{lr_norm:.2f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>{lr_acc:.6f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>{int(lr.n_iter_[0])}</td></tr>"
        f"<tr><td style='padding:4px 12px'>CVXPY feasibility</td>"
        f"<td style='padding:4px 12px;text-align:center'>{cvx_norm:.2f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>{cvx_acc:.6f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>—</td></tr>"
        f"</table>"
        f"<p style='margin-top:8px;font-size:13px'>CVXPY's ||w|| is <b>{cvx_norm / lr_norm:.0f}×</b> larger than LR's — "
        f"the feasibility solution lives at a completely different scale.</p>"
        f"</div>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PyTorch: unregularised linear classifier

    sklearn's `LogisticRegression` with `C=np.inf` *should* remove regularisation, but the L-BFGS
    solver still has convergence tolerances that can prevent it from finding a perfect separator.

    Below we train a simple `nn.Linear(784, 1)` with BCE loss, Adam (no weight decay), for up to
    200 epochs. If it reaches 100 % training accuracy, it stops early.
    """)
    return


@app.cell
def _(mo):
    digits = [str(d) for d in range(10)]
    digit_a_input = mo.ui.dropdown(digits, value="5", label="Digit A")
    digit_b_input = mo.ui.dropdown(digits, value="6", label="Digit B")
    mo.hstack([digit_a_input, digit_b_input])
    return digit_a_input, digit_b_input


@app.cell
def _(
    LogisticRegression,
    X_train,
    digit_a_input,
    digit_b_input,
    mo,
    np,
    warnings,
    y_train,
):
    import torch
    import torch.nn as nn

    da = int(digit_a_input.value)
    db = int(digit_b_input.value)
    mask_pt = (y_train == da) | (y_train == db)
    X_pair = X_train[mask_pt]
    y_pair = y_train[mask_pt]

    # --- PyTorch (no regularisation) ---
    X_pt = torch.tensor(X_pair, dtype=torch.float32)
    y_pt = torch.tensor((y_pair == da).astype(np.float32)).unsqueeze(1)

    model = nn.Linear(784, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    loss_fn = nn.BCEWithLogitsLoss()

    pt_epochs = 0
    for epoch in range(1, 20001):
        logits = model(X_pt)
        loss = loss_fn(logits, y_pt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ((model(X_pt) > 0).float() == y_pt).all():
            pt_epochs = epoch
            break
    else:
        pt_epochs = 200

    pt_preds = (model(X_pt) > 0).float()
    pt_acc = float((pt_preds == y_pt).float().mean())

    # --- sklearn LR (C=inf) ---
    warnings.filterwarnings(
        "ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters"
    )
    clf = LogisticRegression(max_iter=10_000, C=np.inf, solver="lbfgs")
    clf.fit(X_pair, y_pair)
    sklearn_acc = float(np.mean(clf.predict(X_pair) == y_pair))

    mo.Html(
        f"<div style='font-family:monospace'>"
        f"<h3>Digit {da} vs {db} — train set</h3>"
        f"<table style='border-collapse:collapse;font-size:14px'>"
        f"<tr><th style='padding:4px 12px;text-align:left'>Method</th>"
        f"<th style='padding:4px 12px'>Accuracy</th>"
        f"<th style='padding:4px 12px'>Detail</th></tr>"
        f"<tr><td style='padding:4px 12px'>PyTorch (no reg.)</td>"
        f"<td style='padding:4px 12px;text-align:center'>{pt_acc:.6f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>{pt_epochs} epochs</td></tr>"
        f"<tr><td style='padding:4px 12px'>sklearn LR (C=inf)</td>"
        f"<td style='padding:4px 12px;text-align:center'>{sklearn_acc:.6f}</td>"
        f"<td style='padding:4px 12px;text-align:center'>{int(clf.n_iter_[0])} iters</td></tr>"
        f"</table></div>"
    )
    return


if __name__ == "__main__":
    app.run()
