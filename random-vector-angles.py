# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "matplotlib==3.10.6",
#     "wigglystuff==0.5.32",
# ]
# [tool.marimo.opengraph]
# title = "Angles between random vectors"
# description = "Two random unit vectors in D dimensions are nearly orthogonal — simulated the slow way and with a rotational-invariance shortcut."
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import io
    import math

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import FormulaAnimation, FramePlayer, TangleLatex

    return FormulaAnimation, FramePlayer, TangleLatex, io, math, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Angles between two random vectors

    Pick two random unit vectors in $D$ dimensions. What's the angle between them?
    As $D$ grows the answer piles up at $90°$: random vectors are almost always
    nearly orthogonal.

    **The slow way.** Draw $A, B \sim \mathcal{N}(0,1)^D$, normalize, and take
    $\cos\theta = \frac{A\cdot B}{\|A\|\,\|B\|}$. That's $2D$ random numbers per sample.

    **The shortcut.** A circle (or sphere) doesn't care how it's rotated, so we can
    rotate one of the two vectors onto $e_D = (0,\dots,0,1)$ for free. Now the dot
    product collapses: $A \cdot e_D$ is just $A_D$, the last coordinate, and
    $\|e_D\| = 1$. So we keep the one random array $A$ and read a single number out of
    it:

    $$\cos\theta = \frac{A_D}{\|A\|}.$$

    No second vector, no dot-product sum over $D$ terms — just one coordinate divided by
    the norm. The two methods give the *same distribution*, not the same samples: the
    slow one still folds in a random second vector, so its angles differ value-by-value.
    The overlaid histograms sit almost on top of each other, and the little bin-to-bin
    wiggle is just Monte Carlo noise that shrinks as you add samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dim = mo.ui.slider(2, 1000, value=50, label="D (dimensions)", show_value=True)
    n_samples = mo.ui.slider(
        1000, 50000, value=20000, step=1000, label="samples", show_value=True
    )
    seed = mo.ui.number(0, 9999, value=0, label="seed")
    mo.vstack([dim, n_samples, seed])
    return dim, n_samples, seed


@app.cell
def _(np, plt):
    def angles_full(D, N, s):
        rng = np.random.default_rng(s)
        A = rng.standard_normal((N, D))
        B = rng.standard_normal((N, D))
        cos = np.sum(A * B, axis=1) / (
            np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        )
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def angles_trick(D, N, s):
        rng = np.random.default_rng(s)
        A = rng.standard_normal((N, D))
        # Fix the other vector to e_D = (0,...,0,1): the dot product is then
        # just A's last coordinate, and its norm is 1. Reuse the one array.
        cos = A[:, -1] / np.linalg.norm(A, axis=1)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def hist_fig(D, N, s):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0, 180, 80)
        ax.hist(
            angles_full(D, N, s), bins=bins, density=True,
            alpha=0.45, color="#4C78A8", label="full numpy",
        )
        ax.hist(
            angles_trick(D, N, s), bins=bins, density=True,
            histtype="step", linewidth=2.0, color="#E45756", label="rotational shortcut",
        )
        ax.axvline(90, color="#555555", linewidth=1)
        ax.set_xlim(0, 180)
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("density")
        ax.set_title(f"Angle between two random vectors, D = {D}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    return angles_full, angles_trick, hist_fig


@app.cell(hide_code=True)
def _(dim, hist_fig, n_samples, seed):
    hist_fig(dim.value, n_samples.value, seed.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Watch the angle concentrate as $D$ grows

    A scatter of angle against dimension, one dimension added per frame. Hit play and
    the cloud funnels onto $90°$ as $D$ climbs. In high dimensions almost every pair of
    random vectors is nearly orthogonal. Both methods are drawn, and they land on top of
    each other.

    The histogram above is a bell curve centred on $90°$, and it gets narrower as $D$
    grows. How narrow, exactly?

    A single coordinate of a random unit vector has spread $1/\sqrt{D}$, because the $D$
    squared coordinates share a total of $1$. That coordinate is the $\cos\theta$ from
    the shortcut, and near $90°$ the angle moves with it one-for-one. So the angle's
    spread is $1/\sqrt{D}$ **radians**.

    Degrees are easier to read. One radian is $\tfrac{180}{\pi} \approx 57.3°$, so the
    same spread is $\tfrac{180}{\pi}\big/\sqrt{D}$ degrees. The $57.3$ is nothing but the
    radian-to-degree conversion. You can check the width against the histogram above.
    """)
    return


@app.cell
def _(angles_full, angles_trick, np, seed):
    # Sample once for every dimension; each animation frame reveals more of this.
    M = 20  # points per dimension, per method
    dims = np.arange(2, 501)
    xs = np.repeat(dims, M)
    ys_full = np.concatenate([angles_full(int(d), M, seed.value) for d in dims])
    ys_trick = np.concatenate([angles_trick(int(d), M, seed.value) for d in dims])
    return M, xs, ys_full


@app.cell(hide_code=True)
def _(FramePlayer, M, io, mo, plt, xs, ys_full):
    def frame_png(i):
        d_max = i + 2  # dimensions run 2, 3, ...
        n = (i + 1) * M  # points for dimensions 2..d_max
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(xs[:n], ys_full[:n], s=5, alpha=0.25, linewidths=0,
                   color="#4C78A8", label="full numpy")
        ax.axhline(90, color="#555555", linewidth=1)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 180)
        ax.set_xlabel("dimension D")
        ax.set_ylabel("angle (degrees)")
        ax.set_title(f"Angle vs dimension, D up to {d_max}")
        ax.legend(loc="upper right", markerscale=3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        return buf.getvalue()

    frames = [frame_png(i) for i in range(len(xs) // M)]
    player = mo.ui.anywidget(FramePlayer(frames, interval_ms=40, loop=False))
    player
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How many concepts fit in $D$ dimensions?

    Say we want to store $m$ "concepts" as unit vectors. Two of them *collide* when their
    angle sits within $t°$ of $90°$. How big can $m$ get before some pair collides?

    The angle is that bell curve at $90°$ with spread $1/\sqrt{D}$ radians. So the chance
    one pair collides is just a normal-curve tail, $t°$ out from the centre:

    $$p(\text{collision}) = P(|\text{angle} - 90°| > t°), \qquad
    z = \frac{t \text{ in radians}}{1/\sqrt{D}} = t_{\text{rad}}\sqrt{D}.$$

    Now count pairs and find where they start to collide. The steps below do that.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    capacity_steps = mo.ui.anywidget(
        FormulaAnimation(
            title="From pairs to capacity",
            steps=[
                {"tex": r"m \text{ concepts}",
                 "note": "We store m unit vectors and want them mutually near-orthogonal."},
                {"tex": r"\binom{m}{2} = \dfrac{1}{2}m(m-1) \text{ pairs}",
                 "note": "Every pair is one chance for a collision."},
                {"tex": r"\dfrac{1}{2}m(m-1) \approx \dfrac{1}{2}m^2",
                 "note": "For large m the -1 barely matters."},
                {"tex": r"\dfrac{m^{2}}{2}\, p(\text{collision}, D) \text{ collisions expected}",
                 "note": "Multiply the pairs by p(collision), the chance a single pair collides."},
                {"tex": r"\dfrac{1}{2}m^2\, p(\text{collision}, D) \approx 1",
                 "note": "Capacity is where collisions just start to appear."},
                {"tex": r"m^\star \approx \sqrt{\dfrac{2}{p(\text{collision, D})}}",
                 "note": "Solve for m. That is the capacity."},
            ],
        )
    )
    capacity_steps
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How fast does $p(\text{collision})$ fall as we add dimensions? A pair counts as a
    collision when the two vectors' cosine similarity climbs above an *allowance*. A
    tighter allowance is stricter, so those pairs collide more often.
    """)
    return


@app.cell
def _(math, np):
    # cos(theta) ~ N(0, 1/D), so a collision (|cos| > allowance) has probability
    # p = erfc(allowance * sqrt(D) / sqrt(2)), the two-sided normal tail. Precompute
    # the four curves once; the animations below just reveal more of the D axis.
    erfc = np.vectorize(math.erfc)
    d_grid = np.arange(1, 1001)
    allowances = [0.01, 0.02, 0.05, 0.10, 0.20]
    allow_colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756", "#B279A2"]
    p_curves = [erfc(a * np.sqrt(d_grid) / np.sqrt(2.0)) for a in allowances]
    reveal = np.linspace(2, 1000, 80).astype(int)
    return allow_colors, allowances, d_grid, erfc, p_curves, reveal


@app.cell(hide_code=True)
def _(
    FramePlayer,
    allow_colors,
    allowances,
    d_grid,
    io,
    mo,
    p_curves,
    plt,
    reveal,
):
    def frame_p(n):
        fig, ax = plt.subplots(figsize=(7, 4))
        for allow, color, p in zip(allowances, allow_colors, p_curves):
            ax.plot(d_grid[:n], p[:n], color=color, linewidth=2,
                    label=f"{allow:.0%} allowance")
        ax.set_yscale("log")
        ax.set_xlim(0, 1000)
        lo = min(float(c[c > 0].min()) for c in p_curves)
        ax.set_ylim(lo * 0.5, 1.3)
        ax.set_xlabel("dimension D")
        ax.set_ylabel("p(collision)")
        ax.set_title(f"Collision chance vs dimension (D up to {int(d_grid[n - 1])})")
        ax.legend(loc="upper right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    frames_p = [frame_p(n) for n in reveal]
    mo.ui.anywidget(FramePlayer(frames_p, interval_ms=60, loop=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now feed each collision chance through $m^\star = \sqrt{2/p(\text{collision})}$ and
    watch capacity grow the other way: as $p(\text{collision})$ falls, the number of
    concepts you can pack climbs.
    """)
    return


@app.cell(hide_code=True)
def _(FramePlayer, allow_colors, allowances, erfc, io, mo, np, plt):
    # Capacity keeps climbing past D=1000, so give this plot its own longer grid.
    d_grid_m = np.arange(1, 2001)
    reveal_m = np.linspace(2, 2000, 80).astype(int)
    m_curves = [
        np.sqrt(2.0 / erfc(a * np.sqrt(d_grid_m) / np.sqrt(2.0)))
        for a in allowances
    ]

    def frame_m(n):
        fig, ax = plt.subplots(figsize=(7, 4))
        for allow, color, m in zip(allowances, allow_colors, m_curves):
            ax.plot(d_grid_m[:n], m[:n], color=color, linewidth=2,
                    label=f"{allow:.0%} allowance")
        ax.set_yscale("log")
        ax.set_xlim(0, 2000)
        hi = max(float(m.max()) for m in m_curves)
        ax.set_ylim(1, hi * 2)
        ax.set_xlabel("dimension D")
        ax.set_ylabel("capacity m*")
        ax.set_title(f"Capacity vs dimension (D up to {int(d_grid_m[n - 1])})")
        ax.legend(loc="upper left")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()

    frames_m = [frame_m(n) for n in reveal_m]
    mo.ui.anywidget(FramePlayer(frames_m, interval_ms=60, loop=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## So how many concepts fit?

    Pick an *allowance*: how many degrees a pair may tilt off $90°$ before we call it a
    collision. A larger allowance is more forgiving, so fewer pairs collide and more
    concepts fit. Drag the numbers and read the count off the formula.

    It is a rough estimate from the bell curve, and it errs low, so the true count is
    even higher.
    """)
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo):
    capacity_formula = mo.ui.anywidget(
        TangleLatex(
            latex=r"m^\star \;\approx\; \sqrt{\dfrac{2}{P\!\left(\,\text{tilt off }90^\circ > \tangle{allowance}^\circ\,\right)}}"
            r"\quad\text{at}\quad D=\tangle{D}\ \text{dims}",
            parameters={
                "allowance": {"value": 30, "min_value": 1, "max_value": 89, "step": 1},
                "D": {"value": 300, "min_value": 2, "max_value": 5000, "step": 1},
            },
            reveal_all_on_drag=True,
        )
    )
    capacity_formula
    return (capacity_formula,)


@app.cell
def _(math):
    def log10_capacity(D, t_deg):
        # Read capacity off the bell curve. The angle is normal around 90 with spread
        # 57.3/sqrt(D) degrees, so a collision (more than t degrees off 90) is a normal
        # tail p = erfc(z / sqrt(2)), z = t / (57.3/sqrt(D)) = radians(t) * sqrt(D).
        # Capacity is m* = sqrt(2/p). We work in log10 so it never overflows: p gets as
        # small as 1e-100 for large D. When erfc underflows to 0, use its tail formula
        # erfc(a) ~ exp(-a^2)/(a sqrt(pi)) so log p stays finite.
        z = math.radians(t_deg) * math.sqrt(D)
        a = z / math.sqrt(2.0)
        tail = math.erfc(a)
        if tail > 0.0:
            log_p = math.log(tail)
        else:
            log_p = -a * a - math.log(a) - 0.5 * math.log(math.pi)
        return 0.5 * (math.log(2.0) - log_p) / math.log(10.0)

    return (log10_capacity,)


@app.cell(hide_code=True)
def _(capacity_formula, log10_capacity, math, mo):
    def humanize(log10_m):
        # Turn a log10 count into a spoken number. Above a quadrillion, just say 10^N.
        if log10_m > 17:
            return f"about $10^{{{log10_m:.0f}}}$"
        for power, name in [(15, "quadrillion"), (12, "trillion"),
                            (9, "billion"), (6, "million"), (3, "thousand")]:
            if log10_m >= power:
                return f"about {10 ** (log10_m - power):.1f} {name}"
        return f"about {10 ** log10_m:.0f}"

    vals = capacity_formula.value["values"]
    D_cap = int(vals["D"])
    allow_cap = vals["allowance"]
    log10_m = log10_capacity(D_cap, allow_cap)
    log10_p = 0.30103 - 2 * log10_m  # p = 2/m^2, so log10 p = log10(2) - 2 log10(m)
    # tilt off 90 by more than the allowance == cosine similarity above sin(allowance)
    cos_thresh = math.sin(math.radians(allow_cap))

    mo.md(
        f"### Capacity: {humanize(log10_m)} concepts\n\n"
        f"In {D_cap} dimensions, with a {allow_cap:g}° allowance. A pair collides once its "
        f"tilt off 90° tops {allow_cap:g}° — the same as a cosine similarity above "
        f"**{cos_thresh:.2f}**. One random pair crosses that line only about 1 in "
        f"$10^{{{-log10_p:.0f}}}$ of the time, which is why so many concepts fit. The "
        f"bell-curve estimate errs low, so the true count is even higher."
    )
    return


if __name__ == "__main__":
    app.run()
