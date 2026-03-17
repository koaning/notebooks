import marimo

__generated_with = "0.13.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # Angles Between Random Vectors in High Dimensions

    As dimensionality increases, random vectors tend to become nearly orthogonal.
    This notebook lets you explore that phenomenon interactively.
    """)
    return (mo,)


@app.cell
def _():
    import numpy as np

    n_pairs = 10_000
    max_dim = 200
    return max_dim, n_pairs, np


@app.cell
def _(np, max_dim, n_pairs):
    dims = np.arange(2, max_dim + 1)

    # Generate all vectors at max dimension, then compute angles for each prefix length.
    # By rotational invariance, angle to e1 depends only on v[:,0] / norm(v[:,:d]).
    rng = np.random.default_rng(42)
    v = rng.normal(size=(n_pairs, max_dim))
    cumsum_sq = np.cumsum(v ** 2, axis=1)  # (n_pairs, max_dim)
    # norms for dims 2..max_dim -> columns 1..max_dim-1
    norms = np.sqrt(cumsum_sq[:, 1:])  # (n_pairs, max_dim-1)
    cos_sim = np.clip(v[:, 0:1] / norms, -1, 1)  # (n_pairs, max_dim-1)
    angles = np.degrees(np.arccos(cos_sim))  # (n_pairs, max_dim-1)
    min_angles = angles.min(axis=0)  # (max_dim-1,)

    stats = [{"dimensions": int(d), "min": float(m)} for d, m in zip(dims, min_angles)]
    return stats,


@app.cell(hide_code=True)
def _(mo, stats):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(stats)

    line = alt.Chart(df).mark_line().encode(
        alt.X("dimensions:Q", scale=alt.Scale(type="log"), title="Number of dimensions"),
        alt.Y("min:Q", title="Angle (degrees)", scale=alt.Scale(domain=[0, 180])),
    )

    rule = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(alt.Y("y:Q"))

    chart = (line + rule).properties(width=600, height=300)

    mo.as_html(chart)
    return


if __name__ == "__main__":
    app.run()
