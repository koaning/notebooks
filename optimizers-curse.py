# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25.1",
#     "numpy==2.5.3",
#     "scipy==1.16.3",
#     "scikit-learn==1.9.1",
#     "polars==1.39.3",
#     "wigglystuff==0.5.32",
#     "joblib==1.6.0",
# ]
# [tool.marimo.opengraph]
# title = "The optimizer's curse"
# description = "A big randomized search over logistic regression picks a winner whose CV score is partly luck. Simulated data lets us measure how much."
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import marimo as mo
    import polars as pl
    from joblib import parallel_config
    from scipy.stats import loguniform, randint, uniform
    from sklearn.datasets import make_classification
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        RandomizedSearchCV,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.utils.parallel import Parallel, delayed
    from wigglystuff import ParallelCoordinates, ProgressBar

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    return (
        LogisticRegression,
        Parallel,
        ParallelCoordinates,
        ProgressBar,
        RandomizedSearchCV,
        StratifiedKFold,
        cross_val_score,
        delayed,
        loguniform,
        make_classification,
        mo,
        parallel_config,
        pl,
        randint,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The optimizer's curse

    Run enough hyperparameter configs and one of them will get a lucky CV score. The search picks that one, so the winner's score is biased upward.

    To see how much luck is in the winner's score, we take the best candidates and score them again on fresh CV splits of the same data.
    """)
    return


@app.cell
def _(mo):
    n_train = mo.ui.slider(500, 10_000, step=500, value=2000, label="training rows")
    pos_rate = mo.ui.slider(
        0.01, 0.5, step=0.01, value=0.05, label="positive class rate"
    )
    n_iter = mo.ui.slider(50, 1000, step=50, value=500, label="search iterations")
    mo.vstack([n_train, pos_rate, n_iter])
    return n_iter, n_train, pos_rate


@app.cell
def _(make_classification, n_train, pos_rate):
    X_train, y_train = make_classification(
        n_samples=n_train.value,
        n_features=50,
        n_informative=5,
        n_redundant=5,
        weights=[1 - pos_rate.value],
        flip_y=0.02,
        class_sep=0.8,
        random_state=0,
    )
    return X_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The search

    Logistic regression with the `saga` solver, so `random_state` actually changes the fit. The search space covers regularization strength `C`, the L1/L2 mix (`l1_ratio`), class weighting, `max_iter`, and the seed. Scored with average precision since the positive class is rare.
    """)
    return


@app.class_definition
class SearchProgress:
    """sklearn callback that ticks a wigglystuff ProgressBar once per CV fit."""

    max_propagation_depth = 0

    def __init__(self, bar):
        self.bar = bar

    def setup(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context, **kwargs):
        pass

    def on_fit_task_end(self, estimator, context, **kwargs):
        if context.task_name == "candidate-split-evaluation":
            self.bar.value += 1

    def teardown(self, estimator, context):
        pass


@app.cell
def _(
    LogisticRegression,
    ProgressBar,
    RandomizedSearchCV,
    StratifiedKFold,
    X_train,
    loguniform,
    mo,
    n_iter,
    parallel_config,
    randint,
    uniform,
    y_train,
):
    space = {
        "C": loguniform(1e-3, 1e2),
        "l1_ratio": uniform(0, 1),
        "class_weight": [None, "balanced"],
        "max_iter": randint(5, 500),
        "random_state": randint(0, 10_000),
    }

    n_splits = 5
    search_bar = ProgressBar(max_value=n_iter.value * n_splits)

    mo.output.replace(
        mo.vstack([mo.md("**Searching** (one tick per CV fit)"), search_bar])
    )

    search = RandomizedSearchCV(
        LogisticRegression(solver="saga"),
        space,
        n_iter=n_iter.value,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
        n_jobs=-1,
        random_state=0,
    )




    search.set_callbacks(SearchProgress(search_bar))
    # Threads, not processes, so the callback can update the widget directly.
    with parallel_config(backend="threading"):
        search.fit(X_train, y_train)
    return (search,)


@app.cell
def _(pl, search):
    results = pl.DataFrame(
        [
            {**p, "cv_score": cv}
            for p, cv in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
            )
        ]
    ).with_row_index("iteration")
    results.sort("cv_score", descending=True)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every line is one candidate from the search, colored by its CV score.
    """)
    return


@app.cell
def _(slider):
    a = slider.value
    return (a,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1,10,1)
    slider
    return (slider,)


@app.cell
def _(a):
    a
    return


@app.cell
def _(ParallelCoordinates, mo, pl, results):
    param_plot = mo.ui.anywidget(
        ParallelCoordinates(
            results.with_columns(pl.col("class_weight").fill_null("none")),
            color_by="cv_score",
            ignore=["iteration"],
        )
    )
    param_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rescoring the top candidates

    The search scored every candidate on one fixed set of folds. Here the top candidates get scored again on fresh fold splits. The drop from their search score is luck the search selected for.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    Parallel,
    ProgressBar,
    StratifiedKFold,
    X_train,
    cross_val_score,
    delayed,
    mo,
    parallel_config,
    pl,
    results,
    y_train,
):
    n_top = 20
    n_reshuffles = 10
    top = results.sort("cv_score", descending=True).head(n_top)
    param_names = ["C", "l1_ratio", "class_weight", "max_iter", "random_state"]

    def rescore(row, fold_seed):
        cv = StratifiedKFold(5, shuffle=True, random_state=fold_seed)
        model = LogisticRegression(solver="saga", **{k: row[k] for k in param_names})
        scores = cross_val_score(
            model, X_train, y_train, scoring="average_precision", cv=cv
        )
        return row["iteration"], scores.mean()

    rescore_bar = ProgressBar(max_value=n_top * n_reshuffles)
    mo.output.replace(mo.vstack([mo.md("**Rescoring on fresh folds**"), rescore_bar]))

    rescores = []
    with parallel_config(backend="threading"):
        for item in Parallel(n_jobs=-1, return_as="generator_unordered")(
            delayed(rescore)(row, seed)
            for row in top.iter_rows(named=True)
            for seed in range(1, n_reshuffles + 1)
        ):
            rescores.append(item)
            rescore_bar.value += 1

    rescored = (
        pl.DataFrame(rescores, schema=["iteration", "fresh_cv_score"], orient="row")
        .group_by("iteration")
        .agg(pl.col("fresh_cv_score").mean())
    )
    top_rescored = (
        top.join(rescored, on="iteration")
        .with_columns(drop=pl.col("cv_score") - pl.col("fresh_cv_score"))
        .sort("cv_score", descending=True)
    )
    top_rescored
    return


if __name__ == "__main__":
    app.run()
