# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.25.1",
#     "matplotlib==3.11.2",
#     "numpy==2.5.3",
#     "wigglystuff==0.5.33",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Brains, brawn and a scholarship

    Everyone gets two scores: **intelligence** and **athleticism**. Use the
    correlation slider to decide how related they really are. At 0 they're
    independent, so knowing one tells you nothing about the other.

    Now a scholarship gets handed out. Two questions:

    1. You're very smart **and** you got the scholarship. What does that say about how athletic you are?
    2. You're very smart but you **didn't** get it. Are you more likely to be athletic?

    "Very smart" means intelligence above the scholarship bar. The answer
    depends on how the scholarship is decided. Pick a rule below.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import HoverSlider

    return HoverSlider, mo, np, plt


@app.cell
def _(HoverSlider, mo):
    def hslider(start, stop, value, label, step=0.1):
        return mo.ui.anywidget(
            HoverSlider(start=start, stop=stop, step=step, value=value, label=label, width=320)
        )

    rule = mo.ui.dropdown(
        options={
            "smart only: intelligence > bar": "smart",
            "combined: average of both > bar": "combined",
            "either: intelligence > bar or athleticism > bar": "either",
        },
        value="combined: average of both > bar",
        label="scholarship rule",
    )
    bar = hslider(0.0, 2.5, 1.0, "scholarship bar")
    corr = hslider(-0.9, 0.9, 0.0, "true correlation")
    mo.vstack([rule, mo.hstack([bar, corr], justify="start")])
    return bar, corr, rule


@app.cell
def _(bar, corr, np, rule):
    bar_val = float(bar.value["hover_value"])
    corr_val = float(corr.value["hover_value"])

    def draw(n_people):
        rng = np.random.default_rng(42)
        intelligence = rng.normal(size=n_people)
        athleticism = corr_val * intelligence + np.sqrt(1 - corr_val**2) * rng.normal(size=n_people)
        if rule.value == "smart":
            scholarship = intelligence > bar_val
        elif rule.value == "combined":
            scholarship = (intelligence + athleticism) / 2 > bar_val
        else:
            scholarship = (intelligence > bar_val) | (athleticism > bar_val)
        return intelligence, athleticism, scholarship

    intelligence, athleticism, scholarship = draw(5000)
    return athleticism, bar_val, corr_val, draw, intelligence, scholarship


@app.cell
def _(
    athleticism,
    bar_val,
    corr_val,
    intelligence,
    mo,
    np,
    plt,
    rule,
    scholarship,
):
    groups = [
        ("scholarship", scholarship, "#4C78A8"),
        ("no scholarship", ~scholarship, "#E45756"),
    ]
    bins = np.linspace(-4, 4, 41)

    fig = plt.figure(figsize=(11, 5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    ax_scatter = fig.add_subplot(grid[:, 0])
    ax_int = fig.add_subplot(grid[0, 1])
    ax_ath = fig.add_subplot(grid[1, 1], sharex=ax_int)

    for name, mask, color in groups:
        ax_scatter.scatter(intelligence[mask], athleticism[mask], s=4, c=color, label=name)
        ax_int.hist(intelligence[mask], bins=bins, density=True, color=color, alpha=0.5)
        ax_ath.hist(athleticism[mask], bins=bins, density=True, color=color, alpha=0.5)
    ax_scatter.set_xlabel("intelligence")
    ax_scatter.set_ylabel("athleticism")
    ax_scatter.set_xlim(-4, 4)
    ax_scatter.set_ylim(-4, 4)
    ax_scatter.legend(loc="lower left", markerscale=3, frameon=False)
    ax_int.set_title("intelligence", fontsize=10)
    ax_ath.set_title("athleticism", fontsize=10)
    for ax in (ax_int, ax_ath):
        ax.set_yticks([])
    fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"Rule: **{rule.selected_key}**, bar = {bar_val:.1f}, "
                f"true correlation = {corr_val:+.1f}"
            ),
            fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inside the scholarship

    Now only look at the people who got the scholarship. If you know someone has
    it and they're extremely smart, what should you expect about their
    athleticism? Hover the slider to say how smart you know they are. The blue
    dots are the holders who match, and the histogram on the right shows their
    athleticism. The line is the average athleticism at each level of
    intelligence.
    """)
    return


@app.cell
def _(HoverSlider, mo):
    known_smart = mo.ui.anywidget(
        HoverSlider(start=-1.0, stop=3.5, step=0.1, value=2.0, label="I know your intelligence is above", width=320)
    )
    inside_n = mo.ui.anywidget(
        HoverSlider(start=5000, stop=100000, step=5000, value=50000, label="people simulated", width=320)
    )
    mo.hstack([known_smart, inside_n], justify="start")
    return inside_n, known_smart


@app.cell
def _(corr_val, draw, inside_n, known_smart, mo, np, plt):
    known_val = float(known_smart.value["hover_value"])
    big_int, big_ath, big_scholarship = draw(int(inside_n.value["hover_value"]))
    holder_int = big_int[big_scholarship]
    holder_ath = big_ath[big_scholarship]
    picked = holder_int > known_val

    edges = np.arange(-2, 4.25, 0.25)
    centers, avg_ath = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (holder_int >= lo) & (holder_int < hi)
        if in_bin.sum() >= 20:
            centers.append((lo + hi) / 2)
            avg_ath.append(holder_ath[in_bin].mean())
    inside_corr = np.corrcoef(holder_int, holder_ath)[0, 1] if len(holder_int) > 2 else np.nan

    fig2 = plt.figure(figsize=(10, 4.5))
    grid2 = fig2.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_dots = fig2.add_subplot(grid2[0, 0])
    ax_side = fig2.add_subplot(grid2[0, 1])

    ax_dots.scatter(holder_int[~picked], holder_ath[~picked], s=2, c="#cccccc")
    ax_dots.scatter(holder_int[picked], holder_ath[picked], s=2, c="#4C78A8")
    ax_dots.plot(centers, avg_ath, color="#1F3B5C", lw=2.5)
    ax_dots.set_xlabel("intelligence")
    ax_dots.set_ylabel("athleticism")
    ax_dots.set_xlim(-4, 4)
    ax_dots.set_ylim(-4, 4)

    bins2 = np.linspace(-4, 4, 41)
    ax_side.hist(holder_ath, bins=bins2, density=True, color="#cccccc")
    if picked.any():
        ax_side.hist(holder_ath[picked], bins=bins2, density=True, color="#4C78A8", alpha=0.7)
    ax_side.set_yticks([])
    ax_side.set_title("athleticism", fontsize=10)
    fig2.tight_layout()

    picked_avg = f"{holder_ath[picked].mean():+.2f}" if picked.any() else "nobody"
    all_avg = f"{holder_ath.mean():+.2f}" if len(holder_ath) else "nobody"
    mo.vstack(
        [
            mo.md(
                f"Scholarship holders with intelligence above {known_val:.1f}: "
                f"**{picked.sum()}** people, average athleticism **{picked_avg}** "
                f"(all holders: {all_avg}). "
                f"Correlation among holders: {inside_corr:+.2f} (true: {corr_val:+.1f})"
            ),
            fig2,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
