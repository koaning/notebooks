# /// script
# dependencies = [
#     "altair==6.1.0",
#     "marimo",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "wigglystuff==0.5.6",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns", sql_output="polars")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Morton codes (Z-order curves)

    A **Morton code** maps a 2D point to a 1D integer by *interleaving the bits*
    of its x and y coordinates. Sorting points by their Morton code traces a
    "Z"-shaped (or "N"-shaped) space-filling curve through the plane —
    nearby points on the curve tend to be nearby in 2D.

    Move the slider below to pick a window of points along the sorted Morton
    order and see which 2D points it selects.
    """)
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import altair as alt
    from wigglystuff import CircularRangeSlider

    return CircularRangeSlider, alt, np, pd


@app.cell(hide_code=True)
def _(np):
    rng = np.random.default_rng(0)
    N = 1500
    BITS = 10  # quantization resolution per axis: 2**10 = 1024 bins

    xs = rng.random(N)
    ys = rng.random(N)
    return BITS, N, xs, ys


@app.cell(hide_code=True)
def _(BITS, N, np, pd, xs, ys):
    def morton_code(x_int, y_int, bits):
        """Interleave bits of x_int and y_int (each in [0, 2**bits))."""
        code = np.zeros_like(x_int, dtype=np.int64)
        for i in range(bits):
            code |= ((x_int >> i) & 1) << (2 * i)
            code |= ((y_int >> i) & 1) << (2 * i + 1)
        return code


    def as_bits(values, width):
        return [format(int(v), f"0{width}b") for v in values]


    x_int = np.minimum((xs * (1 << BITS)).astype(np.int64), (1 << BITS) - 1)
    y_int = np.minimum((ys * (1 << BITS)).astype(np.int64), (1 << BITS) - 1)
    codes = morton_code(x_int, y_int, BITS)

    order = np.argsort(codes)
    df = pd.DataFrame(
        {
            "rank": np.arange(N),
            "x": xs[order],
            "y": ys[order],
            "x_bits": as_bits(x_int[order], BITS),
            "y_bits": as_bits(y_int[order], BITS),
            "code": codes[order],
            "code_bits": as_bits(codes[order], 2 * BITS),
        }
    )
    return as_bits, df, morton_code, x_int, y_int


@app.cell(hide_code=True)
def _(N, mo):
    window = mo.ui.range_slider(
        start=0,
        stop=N - 1,
        step=1,
        value=[0, N // 4],
        show_value=True,
        label="Selected rank window",
        full_width=True,
    )
    return (window,)


@app.cell(hide_code=True)
def _(mo):
    curve_pick = mo.ui.dropdown(
        options={
            "Morton (x, y)": "xy",
            "Morton (y, x)": "yx",
            "Hilbert": "h",
            "Moore (closed loop)": "m",
        },
        value="Morton (x, y)",
        label="Curve",
    )
    curve_pick
    return (curve_pick,)


@app.cell(hide_code=True)
def _(chart_h, chart_m, chart_xy, chart_yx, circular, curve_pick, mo, window):
    _chart = {"xy": chart_xy, "yx": chart_yx, "h": chart_h, "m": chart_m}[curve_pick.value]
    _slider = circular if curve_pick.value == "m" else window
    mo.vstack([_slider, _chart])
    return


@app.cell(hide_code=True)
def _(alt, base_xy, df, window):
    lo, hi = window.value
    sel = df[(df["rank"] >= lo) & (df["rank"] <= hi)]

    overlay = (
        alt.Chart(sel)
        .mark_line(color="#e45756", strokeWidth=2)
        .encode(x="x:Q", y="y:Q", order="rank:Q")
    ) + (
        alt.Chart(sel)
        .mark_circle(size=70, color="#e45756", opacity=0.95)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "code:Q", "x:Q", "y:Q"])
    )

    chart_xy = (base_xy + overlay).properties(
        width=520,
        height=520,
        title=f"Morton (x, y) — ranks [{lo}, {hi}]",
    )
    return (chart_xy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What if we swap the interleave order?

    Above we interleaved as `... y1 x1 y0 x0`. What happens if we swap and
    interleave as `... x1 y1 x0 y0` — i.e. feed the bits as `(y, x)` instead
    of `(x, y)`? The same slider drives both so you can compare the two
    orderings side by side.
    """)
    return


@app.cell(hide_code=True)
def _(BITS, N, as_bits, morton_code, np, pd, x_int, xs, y_int, ys):
    codes_yx = morton_code(y_int, x_int, BITS)  # swapped order
    order_yx = np.argsort(codes_yx)
    df_yx = pd.DataFrame(
        {
            "rank": np.arange(N),
            "x": xs[order_yx],
            "y": ys[order_yx],
            "x_bits": as_bits(x_int[order_yx], BITS),
            "y_bits": as_bits(y_int[order_yx], BITS),
            "code": codes_yx[order_yx],
            "code_bits": as_bits(codes_yx[order_yx], 2 * BITS),
        }
    )
    return (df_yx,)


@app.cell(hide_code=True)
def _(alt, base_yx, df_yx, window):
    lo2, hi2 = window.value
    sel_yx = df_yx[(df_yx["rank"] >= lo2) & (df_yx["rank"] <= hi2)]

    overlay_yx = (
        alt.Chart(sel_yx)
        .mark_line(color="#4c78a8", strokeWidth=2)
        .encode(x="x:Q", y="y:Q", order="rank:Q")
    ) + (
        alt.Chart(sel_yx)
        .mark_circle(size=70, color="#4c78a8", opacity=0.95)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "code:Q", "x:Q", "y:Q"])
    )

    chart_yx = (base_yx + overlay_yx).properties(
        width=520,
        height=520,
        title=f"Morton (y, x) — ranks [{lo2}, {hi2}]",
    )
    return (chart_yx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hilbert curve — no big jumps

    The Morton curve "teleports" whenever a high bit flips. The **Hilbert
    curve** also visits every cell of the grid, but consecutive indices are
    *always* in adjacent cells — so the curve never makes long jumps. The same
    slider drives this one too.
    """)
    return


@app.cell(hide_code=True)
def _(BITS, N, as_bits, np, pd, x_int, xs, y_int, ys):
    def hilbert_d(x, y, bits):
        """Map 2D integer coords to Hilbert curve index. Grid side = 2**bits."""
        n = 1 << bits
        x = x.copy()
        y = y.copy()
        d = np.zeros_like(x, dtype=np.int64)
        s = n // 2
        while s > 0:
            rx = ((x & s) > 0).astype(np.int64)
            ry = ((y & s) > 0).astype(np.int64)
            d += s * s * ((3 * rx) ^ ry)
            # rotate quadrant when ry == 0
            flip = ry == 0
            # if rx == 1 and ry == 0: reflect within the sub-square
            refl = flip & (rx == 1)
            x = np.where(refl, s - 1 - x, x)
            y = np.where(refl, s - 1 - y, y)
            # if ry == 0: swap x and y
            x_new = np.where(flip, y, x)
            y_new = np.where(flip, x, y)
            x, y = x_new, y_new
            s //= 2
        return d


    hcodes = hilbert_d(x_int, y_int, BITS)
    order_h = np.argsort(hcodes)
    df_h = pd.DataFrame(
        {
            "rank": np.arange(N),
            "x": xs[order_h],
            "y": ys[order_h],
            "x_bits": as_bits(x_int[order_h], BITS),
            "y_bits": as_bits(y_int[order_h], BITS),
            "hilbert": hcodes[order_h],
        }
    )
    return df_h, hilbert_d


@app.cell(hide_code=True)
def _(alt, base_h, df_h, window):
    lo3, hi3 = window.value
    sel_h = df_h[(df_h["rank"] >= lo3) & (df_h["rank"] <= hi3)]

    overlay_h = (
        alt.Chart(sel_h)
        .mark_line(color="#54a24b", strokeWidth=2)
        .encode(x="x:Q", y="y:Q", order="rank:Q")
    ) + (
        alt.Chart(sel_h)
        .mark_circle(size=70, color="#54a24b", opacity=0.95)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "hilbert:Q", "x:Q", "y:Q"])
    )

    chart_h = (base_h + overlay_h).properties(
        width=520,
        height=520,
        title=f"Hilbert — ranks [{lo3}, {hi3}]",
    )
    return (chart_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### A better slider for a closed loop

    A linear range slider has fixed endpoints — selecting "the bit around the
    seam" of the Moore curve means dragging two windows. The new
    `wigglystuff.CircularRangeSlider` has *no* seam: drag the filled arc past
    12 o'clock and the range wraps around (so `low > high` is allowed and
    means "go clockwise from low through the boundary to high"). It maps
    exactly onto the topology of the closed-loop Moore curve.
    """)
    return


@app.cell(hide_code=True)
def _(CircularRangeSlider, N, mo):
    circular = mo.ui.anywidget(
        CircularRangeSlider(
            start=0,
            stop=N - 1,
            step=1,
            value=(0, N // 4),
            size=240,
            color="#b279a2",
            label="Moore rank window (circular)",
        )
    )
    return (circular,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Moore curve — a closed-loop Hilbert variant

    The **Moore curve** is built from four order-`bits−1` Hilbert sub-curves,
    one per quadrant, rotated so they connect end-to-end *and* the global start
    and end cells are adjacent. Result: a closed loop that still preserves
    locality.
    """)
    return


@app.cell(hide_code=True)
def _(BITS, N, hilbert_d, np, pd, x_int, xs, y_int, ys):
    def moore_d(x, y, bits):
        """Moore curve index. Grid side = 2**bits, bits >= 1."""
        half = 1 << (bits - 1)
        qsize = half * half

        qx = (x >= half).astype(np.int64)
        qy = (y >= half).astype(np.int64)
        lx = x - qx * half
        ly = y - qy * half

        # quadrant traversal order: BL=0, TL=1, TR=2, BR=3
        quad_order = np.where(
            qy == 0,
            np.where(qx == 0, 0, 3),
            np.where(qx == 0, 1, 2),
        )

        s = half
        hx = np.where(qx == 0, ly, s - 1 - ly)
        hy = np.where(qx == 0, s - 1 - lx, lx)

        sub_d = hilbert_d(hx, hy, bits - 1)
        return quad_order * qsize + sub_d


    mcodes = moore_d(x_int, y_int, BITS)
    order_m = np.argsort(mcodes)
    df_m = pd.DataFrame(
        {
            "rank": np.arange(N),
            "x": xs[order_m],
            "y": ys[order_m],
            "moore": mcodes[order_m],
        }
    )
    return (df_m,)


@app.cell(hide_code=True)
def _(N, alt, base_m, circular, df_m, pd):
    lo4, hi4 = (int(v) for v in circular.value["value"])

    if lo4 <= hi4:
        sel_m = df_m[(df_m["rank"] >= lo4) & (df_m["rank"] <= hi4)].assign(
            loop_order=lambda d: d["rank"]
        )
    else:
        upper = df_m[df_m["rank"] >= lo4].assign(loop_order=lambda d: d["rank"])
        lower = df_m[df_m["rank"] <= hi4].assign(loop_order=lambda d: d["rank"] + N)
        sel_m = pd.concat([upper, lower], ignore_index=True)

    overlay_m = (
        alt.Chart(sel_m)
        .mark_line(color="#b279a2", strokeWidth=2)
        .encode(x="x:Q", y="y:Q", order="loop_order:Q")
    ) + (
        alt.Chart(sel_m)
        .mark_circle(size=70, color="#b279a2", opacity=0.95)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "moore:Q", "x:Q", "y:Q"])
    )

    chart_m = (base_m + overlay_m).properties(
        width=520,
        height=520,
        title=f"Moore (closed loop) — ranks [{lo4}, {hi4}]" + (" — wrapping" if lo4 > hi4 else ""),
    )
    return (chart_m,)


@app.cell(hide_code=True)
def _(alt, df):
    base_xy = (
        alt.Chart(df)
        .mark_line(color="#bbbbbb", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
            order="rank:Q",
        )
    ) + (
        alt.Chart(df)
        .mark_circle(size=70, color="#cccccc", opacity=0.85)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "code:Q", "x:Q", "y:Q"])
    )
    return (base_xy,)


@app.cell(hide_code=True)
def _(alt, df_yx):
    base_yx = (
        alt.Chart(df_yx)
        .mark_line(color="#bbbbbb", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
            order="rank:Q",
        )
    ) + (
        alt.Chart(df_yx)
        .mark_circle(size=70, color="#cccccc", opacity=0.85)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "code:Q", "x:Q", "y:Q"])
    )
    return (base_yx,)


@app.cell(hide_code=True)
def _(alt, df_h):
    base_h = (
        alt.Chart(df_h)
        .mark_line(color="#bbbbbb", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
            order="rank:Q",
        )
    ) + (
        alt.Chart(df_h)
        .mark_circle(size=70, color="#cccccc", opacity=0.85)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "hilbert:Q", "x:Q", "y:Q"])
    )
    return (base_h,)


@app.cell(hide_code=True)
def _(N, alt, df_m, pd):
    loop_df_m = pd.concat([df_m, df_m.iloc[[0]].assign(rank=N)], ignore_index=True)

    base_m = (
        alt.Chart(loop_df_m)
        .mark_line(color="#bbbbbb", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
            order="rank:Q",
        )
    ) + (
        alt.Chart(df_m)
        .mark_circle(size=70, color="#cccccc", opacity=0.85)
        .encode(x="x:Q", y="y:Q", tooltip=["rank:Q", "moore:Q", "x:Q", "y:Q"])
    )
    return (base_m,)


if __name__ == "__main__":
    app.run()
