# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "pandas==3.0.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns", auto_download=["ipynb"], sql_output="polars")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## i can haz altair demo?
    """)
    return


@app.cell
def _():
    import altair as alt
    import pandas as pd

    return alt, pd


@app.cell
def _(pd):
    df = pd.read_csv("https://calmcode.io/static/data/chickweight.csv")
    df
    return (df,)


@app.cell
def _(alt, df):
    alt.Chart(df).mark_point().encode(x="Time", y="weight")
    return


if __name__ == "__main__":
    app.run()
