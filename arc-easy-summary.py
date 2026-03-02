# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datasets==4.6.1",
#     "diskcache",
#     "polars",
#     "altair==6.0.0",
#     "scipy==1.17.1",
#     "numpy==2.4.2",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ARC-Easy: Results Summary

    Reads all cached results from the experiment runs, joins with the original
    dataset for full questions, and shows a summary.
    """)
    return


@app.cell
def _():
    import diskcache

    cache = diskcache.Cache(".cache/arc-easy-llm")
    return (cache,)


@app.cell
def _():
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", streaming=True)
    # Load enough examples to cover whatever was cached
    questions_lookup = {}
    for ex in ds.take(2500):
        questions_lookup[ex["id"]] = ex["question"]
    len(questions_lookup)
    return (questions_lookup,)


@app.cell
def _(cache, questions_lookup):
    import polars as pl

    rows = []
    for key in cache:
        value = cache[key]
        if isinstance(value, dict) and "model" in value:
            row = {**value}
            row["full_question"] = questions_lookup.get(row["id"], row.get("question", ""))
            rows.append(row)

    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "id": pl.Utf8, "model": pl.Utf8, "repeat_count": pl.Int64,
        "prompt_order": pl.Utf8, "structured_output": pl.Boolean,
        "question": pl.Utf8, "correct": pl.Utf8, "predicted": pl.Utf8,
        "raw_response": pl.Utf8, "match": pl.Boolean, "full_question": pl.Utf8,
    })
    # Backfill prompt_order for old cache entries that lack it
    if "prompt_order" not in df.columns:
        df = df.with_columns(pl.lit("question_first").alias("prompt_order"))
    df.head()
    return df, pl


@app.cell
def _(df, pl):
    chart_df = (
        df.with_columns(
            pl.when(pl.col("structured_output"))
            .then(pl.lit("structured"))
            .otherwise(pl.lit("plain"))
            .alias("output_format"),
        )
        .group_by("model", "repeat_count", "prompt_order", "output_format")
        .agg(
            pl.col("match").mean().alias("accuracy"),
            pl.col("match").sum().alias("correct"),
            pl.col("match").count().alias("total"),
        )
        .sort("model", "repeat_count")
    )
    return (chart_df,)


@app.cell
def _(chart_df):
    chart_df
    return


@app.cell
def _(chart_df, mo, pl):
    import altair as alt

    _chart = (
        alt.Chart(chart_df.filter(pl.col("output_format") == "plain"))
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X("repeat_count:O", title="Repetitions"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.85, 1.0])),
            column=alt.Column("model:N", title="Model"),
            row=alt.Row("prompt_order:N", title="Prompt Order"),
        )
        .properties(width=200, height=200)
    )
    mo.ui.altair_chart(_chart)
    return (alt,)


@app.cell
def _(alt, chart_df, mo, pl):
    _chart = (
        alt.Chart(chart_df.filter(pl.col("output_format") == "structured"))
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X("repeat_count:O", title="Repetitions"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.85, 1.0])),
            column=alt.Column("model:N", title="Model"),
            row=alt.Row("prompt_order:N", title="Prompt Order"),
        )
        .properties(width=200, height=200)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(chart_df, mo):
    mo.ui.table(chart_df.to_dicts())
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    mismatches = df.filter(pl.col("match").not_())
    mo.md(f"## Mismatches ({len(mismatches)} total)")
    return (mismatches,)


@app.cell
def _(mismatches, mo):
    mo.ui.table(
        mismatches.select(
            "model", "repeat_count", "full_question", "correct", "predicted", "raw_response"
        ).to_dicts()
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Statistical Significance

    For each model and output format, we compare the accuracy at different
    repetition counts against the baseline (1 repetition) using Beta distributions.

    Each accuracy is modeled as Beta(correct + 1, incorrect + 1). We estimate
    P(repeated > baseline) by Monte Carlo sampling.
    """)
    return


@app.cell
def _(chart_df, mo, pl):
    from scipy.stats import beta
    import numpy as np

    n_samples = 100_000
    rng = np.random.default_rng(42)
    sig_rows = []

    for model in chart_df["model"].unique().sort().to_list():
        for order in chart_df["prompt_order"].unique().sort().to_list():
            for fmt in chart_df["output_format"].unique().sort().to_list():
                subset = chart_df.filter(
                    (pl.col("model") == model)
                    & (pl.col("prompt_order") == order)
                    & (pl.col("output_format") == fmt)
                )
                baseline = subset.filter(pl.col("repeat_count") == 1)
                if baseline.is_empty():
                    continue
                b_correct = baseline["correct"][0]
                b_total = baseline["total"][0]
                b_samples = rng.beta(b_correct + 1, b_total - b_correct + 1, n_samples)

                for _row in subset.filter(pl.col("repeat_count") > 1).iter_rows(named=True):
                    r_samples = rng.beta(
                        _row["correct"] + 1, _row["total"] - _row["correct"] + 1, n_samples
                    )
                    p_better = float(np.mean(r_samples > b_samples))
                    sig_rows.append({
                        "model": model,
                        "prompt_order": order,
                        "output_format": fmt,
                        "repeat_count": _row["repeat_count"],
                        "baseline_accuracy": round(b_correct / b_total, 4),
                        "repeated_accuracy": round(_row["correct"] / _row["total"], 4),
                        "delta": round(_row["correct"] / _row["total"] - b_correct / b_total, 4),
                        "P(repeated > baseline)": round(p_better, 4),
                    })

    sig_df = pl.DataFrame(sig_rows)
    mo.ui.table(sig_df.to_dicts())
    return


if __name__ == "__main__":
    app.run()
