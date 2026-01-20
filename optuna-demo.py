# /// script
# dependencies = [
#     "marimo",
#     "optuna==4.7.0",
#     "sqlalchemy==2.0.45",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns", sql_output="polars")


@app.cell
def _():
    import optuna

    study = optuna.create_study(storage="sqlite:///optuna_study.db")
    return (study,)


@app.cell
def _(study):
    def objective(trial):
        x = trial.suggest_float("x", -100, 100)
        return x**2


    study.optimize(objective, n_trials=1000, timeout=3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
