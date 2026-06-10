# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "wigglystuff==0.5.9",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck, ParallelCoordinates

    return ChartPuck, ParallelCoordinates, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Clifford Attractors

    A *Clifford attractor* is a strange attractor from a 2D map:

    $$x_{n+1} = \sin(a\, y_n) + c\, \cos(a\, x_n)$$
    $$y_{n+1} = \sin(b\, x_n) + d\, \cos(b\, y_n)$$

    The shape is fully determined by four parameters $a, b, c, d$. Tiny changes can
    produce wildly different patterns — this notebook lets you wiggle them and watch
    the trajectory respond.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    a = mo.ui.slider(-3, 3, step=0.01, value=-1.4, label="a", show_value=True, full_width=True)
    b = mo.ui.slider(-3, 3, step=0.01, value=1.6,  label="b", show_value=True, full_width=True)
    c = mo.ui.slider(-3, 3, step=0.01, value=1.0,  label="c", show_value=True, full_width=True)
    d = mo.ui.slider(-3, 3, step=0.01, value=0.7,  label="d", show_value=True, full_width=True)
    return a, b, c, d


@app.cell(hide_code=True)
def _(np):
    def clifford(a, b, c, d, x0, y0, n=80_000):
        xs = np.empty(n)
        ys = np.empty(n)
        x, y = x0, y0
        for i in range(n):
            xs[i] = x
            ys[i] = y
            x, y = np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
        return xs, ys

    return (clifford,)


@app.cell(hide_code=True)
def _(ChartPuck, a, b, c, clifford, d):
    def _draw(ax, widget):
        xs, ys = clifford(a.value, b.value, c.value, d.value, widget.x[0], widget.y[0])
        ax.scatter(xs, ys, s=0.05, c="black", alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"a={a.value:.2f}, b={b.value:.2f}, c={c.value:.2f}, d={d.value:.2f}")

    puck = ChartPuck.from_callback(
        draw_fn=_draw,
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        figsize=(5, 5),
        x=0.1, y=0.0,
        puck_radius=8,
    )
    return (puck,)


@app.cell(hide_code=True)
def _(a, b, c, d, mo, puck):
    mo.hstack(
        [mo.vstack([a, b, c, d], gap=0.5), puck],
        justify="start",
        align="center",
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Where in $(a, b, c, d)$-space are the interesting attractors?

    Sampling 800 random parameter combinations and scoring each by **coverage** —
    the number of cells of an 80×80 grid that the attractor visits after a burn-in.
    High coverage ≈ a spread-out, dense attractor (lots of black/grey pixels). Low
    coverage ≈ a degenerate fixed point or short cycle. The parallel-coordinates
    plot below lets you brush across $a, b, c, d$ to see which slices of parameter
    space produce dense vs. sparse attractors.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    def sample_clifford_coverage(n_samples=10_000, n_iter=4000, burn=500, grid=80, lo=-2.5, hi=2.5, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.uniform(lo, hi, n_samples)
        B = rng.uniform(lo, hi, n_samples)
        C = rng.uniform(lo, hi, n_samples)
        D = rng.uniform(lo, hi, n_samples)
        x = np.full(n_samples, 0.1)
        y = np.zeros(n_samples)
        occ = np.zeros((n_samples, grid, grid), dtype=bool)
        idx = np.arange(n_samples)
        x_min = np.full(n_samples,  np.inf)
        x_max = np.full(n_samples, -np.inf)
        y_min = np.full(n_samples,  np.inf)
        y_max = np.full(n_samples, -np.inf)
        for step in range(n_iter):
            x, y = np.sin(A * y) + C * np.cos(A * x), np.sin(B * x) + D * np.cos(B * y)
            if step >= burn:
                ix = np.clip(((x - lo) / (hi - lo) * grid).astype(int), 0, grid - 1)
                iy = np.clip(((y - lo) / (hi - lo) * grid).astype(int), 0, grid - 1)
                occ[idx, ix, iy] = True
                np.minimum(x_min, x, out=x_min)
                np.maximum(x_max, x, out=x_max)
                np.minimum(y_min, y, out=y_min)
                np.maximum(y_max, y, out=y_max)
        coverage = occ.sum(axis=(1, 2))
        width = x_max - x_min
        height = y_max - y_min
        ratio = np.where(width > 1e-6, height / np.maximum(width, 1e-6), np.nan)
        return A, B, C, D, coverage, ratio


    def quantile_rank(values):
        """Empirical CDF rank in [0, 1]; NaNs stay NaN."""
        out = np.full_like(values, np.nan, dtype=float)
        finite = np.isfinite(values)
        n = finite.sum()
        if n > 1:
            ranks = np.argsort(np.argsort(values[finite]))
            out[finite] = ranks / (n - 1)
        elif n == 1:
            out[finite] = 0.5
        return out


    A_s, B_s, C_s, D_s, cov_s, ratio_s = sample_clifford_coverage()
    hw_q = quantile_rank(ratio_s)

    samples = [
        {"a": float(A_s[i]), "b": float(B_s[i]), "c": float(C_s[i]),
         "d": float(D_s[i]), "pixel_score": int(cov_s[i]),
         "hw_quantile": float(hw_q[i]) if np.isfinite(hw_q[i]) else 0.0}
        for i in range(len(cov_s))
    ]
    mo.md(f"Sampled **{len(samples)}** combos — pixel score "
          f"{cov_s.min()}–{cov_s.max()} (median {int(np.median(cov_s))}); "
          f"h/w raw range {np.nanmin(ratio_s):.2f}–{np.nanmax(ratio_s):.2f}, "
          f"shown as quantile in [0, 1]")
    return (samples,)


@app.cell(hide_code=True)
def _(ParallelCoordinates, mo, samples):
    pc = mo.ui.anywidget(ParallelCoordinates(
        data=samples,
        color_by="pixel_score",
        height=420,
    ))
    pc
    return (pc,)


@app.cell(hide_code=True)
def _(clifford, mo, np, pc, plt, samples):
    _state = pc.value
    selected = _state.get("selected_uids") or []
    if selected:
        pool = sorted(int(u) for u in selected)
        label = f"**{len(pool)}** rows in current brush"
    else:
        pool = list(range(len(samples)))
        label = f"No brush — sampling from all **{len(pool)}** rows"

    _rng = np.random.default_rng(42)
    k = min(8, len(pool))
    picks = _rng.choice(pool, size=k, replace=False) if k > 0 else []

    if k == 0:
        out = mo.md("*Nothing in the brush.*")
    else:
        rows = (k + 3) // 4
        fig_ex, axes = plt.subplots(rows, 4, figsize=(11, 3 * rows))
        axes = np.atleast_2d(axes)
        for ax_, idx in zip(axes.ravel(), picks):
            s = samples[idx]
            xs, ys = clifford(s["a"], s["b"], s["c"], s["d"], 0.1, 0.0, n=30_000)
            ax_.scatter(xs, ys, s=0.05, c="black", alpha=0.4)
            ax_.set_aspect("equal")
            ax_.set_xticks([]); ax_.set_yticks([])
            ax_.set_title(
                f"a={s['a']:.2f} b={s['b']:.2f} c={s['c']:.2f} d={s['d']:.2f}\n"
                f"score={s['pixel_score']}", fontsize=8,
            )
        for ax_ in axes.ravel()[k:]:
            ax_.set_visible(False)
        fig_ex.tight_layout()
        out = mo.vstack([mo.md(label), fig_ex])
    out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Chained pucks: pick (a, b), then (c, d)

    Move the left puck to choose $(a, b)$. The right chart recomputes a heatmap of
    **pixel_score** over the $(c, d)$ plane for that $(a, b)$, and the puck on top
    of it picks the final $(c, d)$. The bottom panel renders the resulting Clifford
    attractor.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def score_cd_grid(a_val, b_val, gres=30, n_iter=600, burn=120, pgrid=48, lo=-2.5, hi=2.5):
        """For fixed (a, b), compute pixel_score on a gres x gres grid of (c, d)."""
        cs = np.linspace(lo, hi, gres)
        ds = np.linspace(lo, hi, gres)
        C, D = np.meshgrid(cs, ds, indexing="xy")
        n = gres * gres
        Cf = C.ravel(); Df = D.ravel()
        A = np.full(n, a_val); B = np.full(n, b_val)
        x = np.full(n, 0.1); y = np.zeros(n)
        occ = np.zeros((n, pgrid, pgrid), dtype=bool)
        idx = np.arange(n)
        for step in range(n_iter):
            x, y = np.sin(A * y) + Cf * np.cos(A * x), np.sin(B * x) + Df * np.cos(B * y)
            if step >= burn:
                ix = np.clip(((x - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                iy = np.clip(((y - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                occ[idx, ix, iy] = True
        return cs, ds, occ.sum(axis=(1, 2)).reshape(gres, gres)


    def score_ab_grid(c_val, d_val, gres=30, n_iter=600, burn=120, pgrid=48, lo=-2.5, hi=2.5):
        """For fixed (c, d), compute pixel_score on a gres x gres grid of (a, b)."""
        as_ = np.linspace(lo, hi, gres)
        bs_ = np.linspace(lo, hi, gres)
        Ag, Bg = np.meshgrid(as_, bs_, indexing="xy")
        n = gres * gres
        Af = Ag.ravel(); Bf = Bg.ravel()
        C = np.full(n, c_val); D = np.full(n, d_val)
        x = np.full(n, 0.1); y = np.zeros(n)
        occ = np.zeros((n, pgrid, pgrid), dtype=bool)
        idx = np.arange(n)
        for step in range(n_iter):
            x, y = np.sin(Af * y) + C * np.cos(Af * x), np.sin(Bf * x) + D * np.cos(Bf * y)
            if step >= burn:
                ix = np.clip(((x - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                iy = np.clip(((y - lo) / (hi - lo) * pgrid).astype(int), 0, pgrid - 1)
                occ[idx, ix, iy] = True
        return as_, bs_, occ.sum(axis=(1, 2)).reshape(gres, gres)


    return score_ab_grid, score_cd_grid


@app.cell(hide_code=True)
def _():
    INIT_A, INIT_B = -1.4, 1.6
    INIT_C, INIT_D = 1.0, 0.7

    return INIT_A, INIT_B, INIT_C, INIT_D


@app.cell(hide_code=True)
def _(
    ChartPuck,
    INIT_A,
    INIT_B,
    INIT_C,
    INIT_D,
    clifford,
    gres_slider,
    score_ab_grid,
    score_cd_grid,
):
    def _ab_draw(ax, widget):
        try:
            c_val = cd_puck.x[0]
            d_val = cd_puck.y[0]
        except NameError:
            c_val, d_val = INIT_C, INIT_D
        gres = int(gres_slider.value)
        _, _, cov = score_ab_grid(c_val, d_val, gres=gres)
        ax.imshow(
            cov,
            extent=(-2.5, 2.5, -2.5, 2.5),
            origin="lower",
            cmap="magma",
            aspect="equal",
        )
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title(
            f"score over (a, b) | a={widget.x[0]:.2f}, b={widget.y[0]:.2f}",
            fontsize=8,
        )


    ab_puck = ChartPuck.from_callback(
        draw_fn=_ab_draw,
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        figsize=(3, 3),
        x=INIT_A,
        y=INIT_B,
        puck_radius=10,
        throttle=150,
    )


    def _cd_draw(ax, widget):
        a_val = ab_puck.x[0]
        b_val = ab_puck.y[0]
        gres = int(gres_slider.value)
        _, _, cov = score_cd_grid(a_val, b_val, gres=gres)
        ax.imshow(
            cov,
            extent=(-2.5, 2.5, -2.5, 2.5),
            origin="lower",
            cmap="magma",
            aspect="equal",
        )
        ax.set_xlabel("c")
        ax.set_ylabel("d")
        ax.set_title(
            f"score over (c, d) | c={widget.x[0]:.2f}, d={widget.y[0]:.2f}",
            fontsize=8,
        )


    cd_puck = ChartPuck.from_callback(
        draw_fn=_cd_draw,
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        figsize=(3, 3),
        x=INIT_C,
        y=INIT_D,
        puck_radius=10,
        throttle=150,
    )


    def _attractor_draw(ax, widget):
        a_v, b_v = ab_puck.x[0], ab_puck.y[0]
        c_v, d_v = cd_puck.x[0], cd_puck.y[0]
        xs, ys = clifford(a_v, b_v, c_v, d_v, 0.1, 0.0, n=80_000)
        ax.scatter(xs, ys, s=0.05, c="black", alpha=0.4)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"a={a_v:.2f}, b={b_v:.2f}, c={c_v:.2f}, d={d_v:.2f}",
            fontsize=8,
        )


    attractor_puck = ChartPuck.from_callback(
        draw_fn=_attractor_draw,
        x_bounds=(-3.5, 3.5),
        y_bounds=(-3.5, 3.5),
        figsize=(3, 3),
        x=0.0,
        y=0.0,
        puck_radius=0,
    )


    def _on_ab(_change):
        cd_puck.redraw()
        attractor_puck.redraw()


    def _on_cd(_change):
        ab_puck.redraw()
        attractor_puck.redraw()


    ab_puck.observe(_on_ab, names=["x", "y"])
    cd_puck.observe(_on_cd, names=["x", "y"])

    return ab_puck, attractor_puck, cd_puck


@app.cell(hide_code=True)
def _(ab_puck, attractor_puck, cd_puck, mo):
    mo.hstack(
        [ab_puck, cd_puck, attractor_puck],
        justify="start",
        align="center",
        gap=1.0,
    )

    return


@app.cell(hide_code=True)
def _(mo):
    gres_slider = mo.ui.slider(
        10, 220, step=2, value=30,
        label="(c, d) heatmap resolution",
        show_value=True,
    )
    gres_slider

    return (gres_slider,)


@app.cell(hide_code=True)
def _(ab_puck, attractor_puck, cd_puck, gres_slider):
    # Redraw all three when the resolution slider changes
    _ = gres_slider.value
    ab_puck.redraw()
    cd_puck.redraw()
    attractor_puck.redraw()

    return


if __name__ == "__main__":
    app.run()
