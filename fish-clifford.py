# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "wigglystuff==0.5.34",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from wigglystuff import ChartPuck, TangleLatex

    return ChartPuck, TangleLatex, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fishing in the Clifford attractor

    Each point of a Clifford attractor is computed from the previous one using
    four parameters $a, b, c, d$. Drag a number in the formula below, or move the
    pucks, to change them.

    Move the left puck to choose $(a, b)$. The middle heatmap recomputes
    **pixel_score** over the $(c, d)$ plane for that $(a, b)$: bright cells are
    parameters whose attractor covers lots of the plane. Drag the middle puck onto
    a bright spot and the attractor on the right is rendered as a density image.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def pixel_scores(A, B, C, D, n_iter=600, burn=120, pgrid=64, lo=-2.5, hi=2.5):
        """Number of pgrid x pgrid cells visited by each (A, B, C, D) trajectory."""
        n = len(A)
        x = np.full(n, 0.1)
        y = np.zeros(n)
        occ = np.zeros((n, pgrid, pgrid), dtype=bool)
        idx = np.arange(n)
        for step in range(n_iter):
            x, y = np.sin(A * y) + C * np.cos(A * x), np.sin(B * x) + D * np.cos(B * y)
            if step >= burn:
                ix = np.clip(((x - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                iy = np.clip(((y - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                occ[idx, ix, iy] = True
        return occ.sum(axis=(1, 2))


    def score_grid(fixed_1, fixed_2, free="cd", gres=80, lo=-2.5, hi=2.5):
        """Score a gres x gres grid over the free pair, holding the other pair fixed."""
        vals = np.linspace(lo, hi, gres)
        U, V = np.meshgrid(vals, vals, indexing="xy")
        U, V = U.ravel(), V.ravel()
        F1, F2 = np.full_like(U, fixed_1), np.full_like(U, fixed_2)
        if free == "cd":
            scores = pixel_scores(F1, F2, U, V)
        else:
            scores = pixel_scores(U, V, F1, F2)
        return scores.reshape(gres, gres)


    def clifford_density(a, b, c, d, n_points=1_000_000, n_seeds=4000, burn=50, bins=700, seed=0):
        """Iterate many random seeds in parallel and histogram the visited points."""
        rng = np.random.default_rng(seed)
        steps = n_points // n_seeds
        x = rng.uniform(-1, 1, n_seeds)
        y = rng.uniform(-1, 1, n_seeds)
        xs = np.empty((steps, n_seeds))
        ys = np.empty((steps, n_seeds))
        for step in range(burn + steps):
            x, y = np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
            if step >= burn:
                xs[step - burn] = x
                ys[step - burn] = y
        # Square frame around the attractor so it fills the panel.
        cx = (xs.max() + xs.min()) / 2
        cy = (ys.max() + ys.min()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.05 + 1e-9
        extent = (cx - half, cx + half, cy - half, cy + half)
        hist, _, _ = np.histogram2d(
            xs.ravel(), ys.ravel(), bins=bins, range=[extent[:2], extent[2:]]
        )
        return np.log1p(hist.T), extent

    return clifford_density, score_grid


@app.cell(hide_code=True)
def _():
    init = {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7}
    return (init,)


@app.cell(hide_code=True)
def _(mo):
    # The draw callbacks read from this dict rather than the sliders, so moving a
    # slider redraws the charts instead of re-running (and resetting) the widget cell.
    render_settings = {"gres": 80, "points": 1_000_000}
    gres_slider = mo.ui.slider(
        20, 160, step=4, value=render_settings["gres"],
        label="heatmap resolution",
        show_value=True,
    )
    points_slider = mo.ui.slider(
        steps=[250_000, 500_000, 1_000_000, 2_000_000, 4_000_000],
        value=render_settings["points"],
        label="attractor points",
        show_value=True,
    )
    mo.hstack([gres_slider, points_slider], justify="start", gap=2.0)
    return gres_slider, points_slider, render_settings


@app.cell(hide_code=True)
def _(
    ChartPuck,
    TangleLatex,
    clifford_density,
    init,
    render_settings,
    score_grid,
):
    def _heatmap(ax, scores, xlabel, ylabel, title):
        ax.imshow(
            scores,
            extent=(-2.5, 2.5, -2.5, 2.5),
            origin="lower",
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)


    def _ab_draw(ax, widget):
        try:
            c_val, d_val = cd_puck.x[0], cd_puck.y[0]
        except NameError:
            c_val, d_val = init["c"], init["d"]
        scores = score_grid(c_val, d_val, free="ab", gres=render_settings["gres"])
        _heatmap(ax, scores, "a", "b", f"score over (a, b) | a={widget.x[0]:.2f}, b={widget.y[0]:.2f}")


    ab_puck = ChartPuck.from_callback(
        draw_fn=_ab_draw,
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        figsize=(4.5, 4.5),
        x=init["a"],
        y=init["b"],
        puck_radius=10,
        throttle=150,
    )


    def _cd_draw(ax, widget):
        scores = score_grid(ab_puck.x[0], ab_puck.y[0], free="cd", gres=render_settings["gres"])
        _heatmap(ax, scores, "c", "d", f"score over (c, d) | c={widget.x[0]:.2f}, d={widget.y[0]:.2f}")


    cd_puck = ChartPuck.from_callback(
        draw_fn=_cd_draw,
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        figsize=(4.5, 4.5),
        x=init["c"],
        y=init["d"],
        puck_radius=10,
        throttle=150,
    )


    def _attractor_draw(ax, widget):
        a_v, b_v = ab_puck.x[0], ab_puck.y[0]
        c_v, d_v = cd_puck.x[0], cd_puck.y[0]
        img, extent = clifford_density(a_v, b_v, c_v, d_v, n_points=render_settings["points"])
        ax.imshow(img, extent=extent, origin="lower", cmap="magma", interpolation="bilinear")
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"a={a_v:.2f}, b={b_v:.2f}, c={c_v:.2f}, d={d_v:.2f}", fontsize=10)


    attractor_puck = ChartPuck.from_callback(
        draw_fn=_attractor_draw,
        x_bounds=(-3.5, 3.5),
        y_bounds=(-3.5, 3.5),
        figsize=(4.5, 4.5),
        x=0.0,
        y=0.0,
        puck_radius=0,
    )


    formula = TangleLatex(
        latex=(
            r"\begin{aligned}"
            r"x_{n+1} &= \sin(\tangle{a}\, y_n) + \tangle{c}\, \cos(\tangle{a}\, x_n) \\"
            r"y_{n+1} &= \sin(\tangle{b}\, x_n) + \tangle{d}\, \cos(\tangle{b}\, y_n)"
            r"\end{aligned}"
        ),
        parameters={
            name: {"value": init[name], "min_value": -2.5, "max_value": 2.5, "step": 0.01, "digits": 2}
            for name in "abcd"
        },
        theme="light",
    )


    def _on_ab(_change):
        formula.values = {**formula.values, "a": ab_puck.x[0], "b": ab_puck.y[0]}
        cd_puck.redraw()
        attractor_puck.redraw()


    def _on_cd(_change):
        formula.values = {**formula.values, "c": cd_puck.x[0], "d": cd_puck.y[0]}
        ab_puck.redraw()
        attractor_puck.redraw()


    def _on_formula(change):
        # Move only the puck coordinate that changed; the puck observers handle the redraws.
        vals = change["new"]
        for puck, axis, name in [(ab_puck, "x", "a"), (ab_puck, "y", "b"), (cd_puck, "x", "c"), (cd_puck, "y", "d")]:
            if getattr(puck, axis)[0] != vals[name]:
                setattr(puck, axis, [vals[name]])


    ab_puck.observe(_on_ab, names=["x", "y"])
    cd_puck.observe(_on_cd, names=["x", "y"])
    formula.observe(_on_formula, names=["values"])
    return ab_puck, attractor_puck, cd_puck, formula


@app.cell(hide_code=True, expand_output=True)
def _(ab_puck, attractor_puck, cd_puck, formula, mo):
    mo.vstack(
        [
            formula,
            mo.hstack(
                [ab_puck, cd_puck, attractor_puck],
                justify="start",
                align="center",
                gap=1.0,
                wrap=False,
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(
    ab_puck,
    attractor_puck,
    cd_puck,
    gres_slider,
    points_slider,
    render_settings,
):
    render_settings["gres"] = int(gres_slider.value)
    render_settings["points"] = int(points_slider.value)
    ab_puck.redraw()
    cd_puck.redraw()
    attractor_puck.redraw()
    return


if __name__ == "__main__":
    app.run()
