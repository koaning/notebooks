# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.18.0",
#     "scikit-learn>=1.3.0",
#     "numpy>=1.24.0",
#     "plotly>=5.14.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Warm Starts in Regularized Linear Models

    When training regularized models like Ridge or Logistic Regression, does it help to
    initialize parameters from a previous fit? This is called a **warm start**.

    The answer depends on whether the model uses:
    - **Closed-form solution** (Ridge): Warm starts are irrelevant
    - **Iterative optimization** (Logistic Regression): Warm starts can dramatically speed things up

    This notebook benchmarks both cases.
    """)
    return


@app.cell
def _():
    import numpy as np
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.datasets import make_regression, make_classification
    import time
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return (
        LogisticRegression,
        Ridge,
        go,
        make_classification,
        make_regression,
        make_subplots,
        np,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Ridge Regression - Closed Form Solution

    Ridge regression has the closed-form solution:

    $$ \mathbf{w} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y} $$

    Because of this, **warm starts are irrelevant** - there's no iterative optimization to speed up.

    Even better: using SVD, we can compute solutions for *all* alpha values almost as fast as one!
    """)
    return


@app.cell
def _(mo):
    ridge_n_samples = mo.ui.slider(
        500, 500000, value=2000, step=500, label="n_samples"
    )
    ridge_n_features = mo.ui.slider(
        50, 500, value=200, step=50, label="n_features"
    )
    ridge_n_alphas = mo.ui.slider(
        10, 200, value=50, step=10, label="Number of alpha values"
    )
    mo.hstack([ridge_n_samples, ridge_n_features, ridge_n_alphas])
    return ridge_n_alphas, ridge_n_features, ridge_n_samples


@app.cell
def _(make_regression, np, ridge_n_alphas, ridge_n_features, ridge_n_samples):
    # Generate regression data
    X_ridge, y_ridge = make_regression(
        n_samples=ridge_n_samples.value,
        n_features=ridge_n_features.value,
        noise=10,
        random_state=42,
    )
    alphas = np.logspace(-2, 4, ridge_n_alphas.value)
    return X_ridge, alphas, y_ridge


@app.cell
def _(Ridge, X_ridge, alphas, np, time, y_ridge):
    # Benchmark: Naive loop vs SVD-based computation
    # Using fit_intercept=False for clean SVD comparison

    # Method 1: Naive loop - fit Ridge separately for each alpha
    start = time.perf_counter()
    coefs_naive = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, solver="cholesky", fit_intercept=False)
        ridge.fit(X_ridge, y_ridge)
        coefs_naive.append(ridge.coef_.copy())
    time_naive = time.perf_counter() - start
    coefs_naive = np.array(coefs_naive)

    # Method 2: SVD-based computation (what RidgeCV does internally)
    # Ridge solution via SVD: w = V @ diag(s / (s^2 + alpha)) @ U.T @ y
    start = time.perf_counter()
    U, s, Vt = np.linalg.svd(X_ridge, full_matrices=False)
    # Precompute U.T @ y (shared across all alphas)
    UTy = U.T @ y_ridge
    # For each alpha, compute: V.T @ diag(s / (s^2 + alpha)) @ U.T @ y
    # d[i, j] = s[j] / (s[j]^2 + alphas[i]) for all (alpha, singular value) pairs
    d = s / (s**2 + alphas[:, np.newaxis])  # shape: (n_alphas, n_features)
    # Multiply d by UTy element-wise, then by Vt
    coefs_svd = (d * UTy) @ Vt  # shape: (n_alphas, n_features)
    time_svd = time.perf_counter() - start

    # Verify both methods produce the same coefficients
    _max_diff = np.abs(coefs_naive - coefs_svd).max()

    ridge_results = {
        "naive_time": time_naive,
        "svd_time": time_svd,
        "n_alphas": len(alphas),
        "speedup": time_naive / time_svd,
        "max_coef_diff": _max_diff,
        "coefs_match": np.allclose(coefs_naive, coefs_svd, rtol=1e-5),
    }
    return (ridge_results,)


@app.cell
def _(ridge_results):
    ridge_results
    return


@app.cell
def _(go, mo, ridge_results):
    ridge_bar = go.Figure(
        data=[
            go.Bar(
                x=["Naive Loop", "SVD (all at once)"],
                y=[ridge_results["naive_time"], ridge_results["svd_time"]],
                text=[
                    f"{ridge_results['naive_time']:.3f}s",
                    f"{ridge_results['svd_time']:.3f}s",
                ],
                textposition="auto",
                marker_color=["#636EFA", "#00CC96"],
            )
        ]
    )
    ridge_bar.update_layout(
        title=f"Ridge: {ridge_results['n_alphas']} alpha values ({ridge_results['speedup']:.1f}x speedup)",
        yaxis_title="Time (seconds)",
        showlegend=False,
    )
    mo.ui.plotly(ridge_bar)
    return


@app.cell
def _(mo, ridge_results):
    _match_status = "✓ Coefficients match!" if ridge_results["coefs_match"] else "✗ Mismatch detected"
    mo.md(f"""
    ### Key Insight

    With SVD, computing Ridge coefficients for **{ridge_results['n_alphas']} different alpha values**
    takes almost the same time as computing just one! The SVD is computed once, then each alpha
    just requires cheap vector operations.

    This is why `RidgeCV` is so efficient and why Ridge doesn't have a `warm_start` parameter.

    **Verification**: {_match_status} (max difference: {ridge_results['max_coef_diff']:.2e})
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Part 2: Logistic Regression - Where Warm Starts Shine

    Logistic Regression has **no closed-form solution**. It must be solved iteratively using
    methods like L-BFGS, Newton-CG, or SAG/SAGA.

    When sweeping regularization parameters, each optimization starts from scratch by default.
    But if we use `warm_start=True`, each fit initializes from the previous solution - which
    is typically very close to the new optimum.
    """)
    return


@app.cell
def _(mo):
    log_n_samples = mo.ui.slider(
        1000, 10_000_000, value=5000, step=1000, label="n_samples"
    )
    log_n_features = mo.ui.slider(
        20, 200, value=100, step=20, label="n_features"
    )
    log_n_C_values = mo.ui.slider(
        10, 50, value=20, step=5, label="Number of C values"
    )
    solver_select = mo.ui.dropdown(
        options=["lbfgs", "newton-cg", "sag", "saga"],
        value="lbfgs",
        label="Solver",
    )
    mo.hstack([log_n_samples, log_n_features, log_n_C_values, solver_select])
    return log_n_C_values, log_n_features, log_n_samples, solver_select


@app.cell
def _(log_n_features, log_n_samples, make_classification):
    # Generate classification data
    X_log, y_log = make_classification(
        n_samples=log_n_samples.value,
        n_features=log_n_features.value,
        n_informative=log_n_features.value // 2,
        n_redundant=log_n_features.value // 4,
        random_state=42,
    )
    return X_log, y_log


@app.cell
def _(
    LogisticRegression,
    X_log,
    log_n_C_values,
    np,
    solver_select,
    time,
    y_log,
):
    # Regularization path from high regularization to low
    C_values = np.logspace(-3, 2, log_n_C_values.value)

    # Cold start: each fit starts from zeros
    cold_times = []
    cold_coefs = []
    for C in C_values:
        model_cold = LogisticRegression(
            C=C,
            solver=solver_select.value,
            max_iter=1000,
            warm_start=False,
            random_state=42,
        )
        start_cold = time.perf_counter()
        model_cold.fit(X_log, y_log)
        cold_times.append(time.perf_counter() - start_cold)
        cold_coefs.append(model_cold.coef_.copy())

    # Warm start: each fit starts from previous solution
    warm_times = []
    warm_coefs = []
    model_warm = LogisticRegression(
        C=C_values[0],
        solver=solver_select.value,
        max_iter=1000,
        warm_start=True,
        random_state=42,
    )
    for C in C_values:
        model_warm.C = C
        start_warm = time.perf_counter()
        model_warm.fit(X_log, y_log)
        warm_times.append(time.perf_counter() - start_warm)
        warm_coefs.append(model_warm.coef_.copy())

    _cold_coefs_arr = np.array(cold_coefs).squeeze()
    _warm_coefs_arr = np.array(warm_coefs).squeeze()

    # Verify both methods produce the same coefficients
    _max_diff = np.abs(_cold_coefs_arr - _warm_coefs_arr).max()

    log_results = {
        "cold_total": sum(cold_times),
        "warm_total": sum(warm_times),
        "cold_times": cold_times,
        "warm_times": warm_times,
        "C_values": C_values,
        "cold_coefs": _cold_coefs_arr,
        "warm_coefs": _warm_coefs_arr,
        "speedup": sum(cold_times) / sum(warm_times),
        "max_coef_diff": _max_diff,
        "coefs_match": np.allclose(_cold_coefs_arr, _warm_coefs_arr, rtol=1e-4),
    }
    return (log_results,)


@app.cell
def _(go, log_results, make_subplots, mo, np):
    # Create comparison plots
    fig_log = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Total Time Comparison",
            "Time per C Value",
        ],
    )

    # Total time bar chart
    fig_log.add_trace(
        go.Bar(
            x=["Cold Start", "Warm Start"],
            y=[log_results["cold_total"], log_results["warm_total"]],
            text=[
                f"{log_results['cold_total']:.3f}s",
                f"{log_results['warm_total']:.3f}s",
            ],
            textposition="auto",
            marker_color=["#EF553B", "#00CC96"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Time per C value
    fig_log.add_trace(
        go.Scatter(
            x=np.log10(log_results["C_values"]),
            y=log_results["cold_times"],
            mode="lines+markers",
            name="Cold Start",
            line=dict(color="#EF553B"),
        ),
        row=1,
        col=2,
    )
    fig_log.add_trace(
        go.Scatter(
            x=np.log10(log_results["C_values"]),
            y=log_results["warm_times"],
            mode="lines+markers",
            name="Warm Start",
            line=dict(color="#00CC96"),
        ),
        row=1,
        col=2,
    )

    fig_log.update_layout(
        title=f"Logistic Regression: {log_results['speedup']:.1f}x speedup with warm starts",
        height=400,
    )
    fig_log.update_xaxes(title_text="log10(C)", row=1, col=2)
    fig_log.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig_log.update_yaxes(title_text="Time (seconds)", row=1, col=2)

    mo.ui.plotly(fig_log)
    return


@app.cell
def _(log_results, mo):
    _match_status = "✓ Coefficients match!" if log_results["coefs_match"] else "✗ Mismatch detected"
    mo.md(f"""
    ### Why Does Warm Start Help So Much?

    When we sweep from high regularization (small C) to low regularization (large C):

    1. **High regularization** pushes coefficients toward zero
    2. **Each step** slightly reduces regularization, coefficients grow slightly
    3. **Adjacent solutions** are very similar - a perfect starting point!

    The optimizer only needs a few iterations to go from "almost there" to "converged".

    **Verification**: {_match_status} (max difference: {log_results['max_coef_diff']:.2e})
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Part 3: Visualizing the Coefficient Path

    The regularization path shows how coefficients evolve as we change regularization strength.
    Notice how coefficients change smoothly - this is why warm starts work!
    """)
    return


@app.cell
def _(go, log_results, mo, np):
    # Coefficient path visualization
    n_coefs_to_show = min(10, log_results["warm_coefs"].shape[1])

    fig_path = go.Figure()
    for i in range(n_coefs_to_show):
        fig_path.add_trace(
            go.Scatter(
                x=np.log10(log_results["C_values"]),
                y=log_results["warm_coefs"][:, i],
                mode="lines",
                name=f"Coef {i}",
                line=dict(width=2),
            )
        )

    fig_path.update_layout(
        title="Coefficient Path (first 10 coefficients)",
        xaxis_title="log10(C) - less regularization →",
        yaxis_title="Coefficient value",
        height=400,
        hovermode="x unified",
    )
    mo.ui.plotly(fig_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Summary: When Do Warm Starts Help?

    | Model | Has warm_start? | Closed-form? | Benefit |
    |-------|-----------------|--------------|---------|
    | **Ridge** | No | Yes (SVD) | N/A - use RidgeCV |
    | **Lasso/ElasticNet** | Yes | No | High - critical for paths |
    | **LogisticRegression** | Yes | No | High - for regularization sweeps |
    | **SGDClassifier** | Yes | No | High - online learning |

    ### Practical Recommendations

    1. **For Ridge**: Use `RidgeCV` - it's already optimized via SVD
    2. **For Logistic Regression**: Use `warm_start=True` when sweeping C values
    3. **For Lasso/ElasticNet**: Use `warm_start=True` or better yet, use `lasso_path()` / `enet_path()`
    4. **Order matters**: Sweep from high regularization to low for best warm start benefit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Code Pattern for Warm Start

    ```python
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    # Create model with warm_start=True
    model = LogisticRegression(warm_start=True, solver='lbfgs')

    # Sweep from high to low regularization
    C_values = np.logspace(-3, 2, 50)

    for C in C_values:
        model.C = C
        model.fit(X, y)
        # model.coef_ now available for this C
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
