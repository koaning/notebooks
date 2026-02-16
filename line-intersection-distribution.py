# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "marimo",
#     "numpy",
#     "polars>=1.18.0",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import polars as pl
    from wigglystuff import TangleSlider

    return TangleSlider, alt, mo, np, pl


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    c_b_slider = mo.ui.anywidget(TangleSlider(amount=200, min_value=0, max_value=500, step=10, digits=0))
    m_b_lo_slider = mo.ui.anywidget(TangleSlider(amount=2.0, min_value=0.0, max_value=15.0, step=0.5, digits=1))
    m_b_hi_slider = mo.ui.anywidget(TangleSlider(amount=5.0, min_value=0.0, max_value=15.0, step=0.5, digits=1))

    c_o_slider = mo.ui.anywidget(TangleSlider(amount=0, min_value=0, max_value=500, step=10, digits=0))
    m_o_lo_slider = mo.ui.anywidget(TangleSlider(amount=5.0, min_value=0.0, max_value=20.0, step=0.5, digits=1))
    m_o_hi_slider = mo.ui.anywidget(TangleSlider(amount=10.0, min_value=0.0, max_value=20.0, step=0.5, digits=1))

    param_text = mo.md(f"""
    ## Parameters

    **Blue line:** y = c_b + m_b x

    - Intercept c_b = {c_b_slider}
    - Slope m_b ~ U[{m_b_lo_slider}, {m_b_hi_slider}]

    **Orange line:** y = c_o + m_o x

    - Intercept c_o = {c_o_slider}
    - Slope m_o ~ U[{m_o_lo_slider}, {m_o_hi_slider}]
    """)
    return (
        c_b_slider,
        c_o_slider,
        m_b_hi_slider,
        m_b_lo_slider,
        m_o_hi_slider,
        m_o_lo_slider,
        param_text,
    )


@app.cell
def _(
    c_b_slider,
    c_o_slider,
    m_b_hi_slider,
    m_b_lo_slider,
    m_o_hi_slider,
    m_o_lo_slider,
    np,
):
    rng = np.random.default_rng(42)
    n_sim = 200_000

    c_b = c_b_slider.amount
    c_o = c_o_slider.amount
    m_b_lo = m_b_lo_slider.amount
    m_b_hi = max(m_b_hi_slider.amount, m_b_lo + 0.1)
    m_o_lo = m_o_lo_slider.amount
    m_o_hi = max(m_o_hi_slider.amount, m_o_lo + 0.1)

    K = c_o - c_b

    m_b_samples = rng.uniform(m_b_lo, m_b_hi, size=n_sim)
    m_o_samples = rng.uniform(m_o_lo, m_o_hi, size=n_sim)
    D_samples = m_b_samples - m_o_samples

    # Intersection x* = K / D, keeping only x* > 0
    valid = np.abs(D_samples) > 1e-9
    x_raw = K / D_samples[valid]
    x_star = x_raw[x_raw > 0]
    return D_samples, K, c_b, c_o, m_b_hi, m_b_lo, m_o_hi, m_o_lo, x_star


@app.cell
def _(alt, c_b, c_o, m_b_hi, m_b_lo, m_o_hi, m_o_lo, np, pl):
    x_max = 80
    x_vals = np.linspace(0, x_max, 200)

    # Envelope (min/max bands)
    envelope_df = pl.DataFrame({
        "x": x_vals.tolist(),
        "blue_lo": (c_b + m_b_lo * x_vals).tolist(),
        "blue_hi": (c_b + m_b_hi * x_vals).tolist(),
        "orange_lo": (c_o + m_o_lo * x_vals).tolist(),
        "orange_hi": (c_o + m_o_hi * x_vals).tolist(),
        "blue_mid": (c_b + (m_b_lo + m_b_hi) / 2 * x_vals).tolist(),
        "orange_mid": (c_o + (m_o_lo + m_o_hi) / 2 * x_vals).tolist(),
    })

    blue_band = alt.Chart(envelope_df).mark_area(opacity=0.15, color="#1f77b4").encode(
        x="x:Q", y="blue_lo:Q", y2="blue_hi:Q"
    )
    orange_band = alt.Chart(envelope_df).mark_area(opacity=0.15, color="#ff7f0e").encode(
        x="x:Q", y="orange_lo:Q", y2="orange_hi:Q"
    )
    blue_mid = alt.Chart(envelope_df).mark_line(strokeWidth=2, color="#1f77b4").encode(
        x=alt.X("x:Q", title="x"), y=alt.Y("blue_mid:Q", title="y")
    )
    orange_mid = alt.Chart(envelope_df).mark_line(strokeWidth=2, color="#ff7f0e").encode(
        x="x:Q", y="orange_mid:Q"
    )

    fan_chart = alt.layer(blue_band, orange_band, blue_mid, orange_mid).properties(
        width=450, height=300, title="Two lines with random slopes"
    )
    return (fan_chart,)


@app.cell
def _(D_samples, alt, np, pl):
    # Histogram of D = m_b - m_o
    d_hist, d_edges = np.histogram(D_samples, bins=100, density=True)
    d_centers = (d_edges[:-1] + d_edges[1:]) / 2

    d_df = pl.DataFrame({"D": d_centers.tolist(), "density": d_hist.tolist()})

    zero_in_support = D_samples.min() < 0 < D_samples.max()
    title_suffix = " (0 in support — heavy tails!)" if zero_in_support else " (0 not in support — bounded)"

    d_chart = alt.Chart(d_df).mark_area(opacity=0.4, color="purple").encode(
        x=alt.X("D:Q", title="D = m_b − m_o"),
        y=alt.Y("density:Q", title="Density"),
    ).properties(width=450, height=200, title="Distribution of slope difference" + title_suffix)
    return (d_chart,)


@app.cell
def _(K, alt, m_b_hi, m_b_lo, m_o_hi, m_o_lo, np, pl, x_star):
    # Closed-form density of D = m_b - m_o (difference of two independent uniforms)
    # D = m_b + (-m_o), where m_b ~ U[a,b] and -m_o ~ U[-d,-c]
    # f_D(z) = max(0, min(b, z+d) - max(a, z+c)) / ((b-a)*(d-c))
    a, b = m_b_lo, m_b_hi
    c, d = m_o_lo, m_o_hi
    w1, w2 = b - a, d - c

    def f_D(z):
        return np.maximum(0, np.minimum(b, z + d) - np.maximum(a, z + c)) / (w1 * w2)

    # f(x*) = |K| / x² · f_D(K / x)  for x > 0
    def f_xstar(x):
        return np.where(x > 1e-12, np.abs(K) / (x ** 2) * f_D(K / x), 0.0)

    # Clip x_star to a reasonable display range
    p5, p95 = np.percentile(x_star, [2, 98])
    x_display = x_star[(x_star >= p5) & (x_star <= p95)]

    # Monte Carlo histogram
    mc_hist, mc_edges = np.histogram(x_display, bins=150, density=True)
    mc_centers = (mc_edges[:-1] + mc_edges[1:]) / 2

    mc_df = pl.DataFrame({"x_star": mc_centers.tolist(), "density": mc_hist.tolist()})
    mc_area = alt.Chart(mc_df).mark_area(opacity=0.4, color="steelblue").encode(
        x=alt.X("x_star:Q", title="x* (intersection)"),
        y=alt.Y("density:Q", title="Density"),
    )

    # Analytical curve (closed-form)
    x_analytical = np.linspace(max(p5, 1e-6), p95, 500)
    analytical_df = pl.DataFrame({
        "x_star": x_analytical.tolist(),
        "density": f_xstar(x_analytical).tolist(),
    })
    analytical_line = alt.Chart(analytical_df).mark_line(color="red", strokeWidth=2).encode(
        x="x_star:Q", y="density:Q"
    )

    median_x = float(np.median(x_star))
    std_x = float(np.std(x_display))

    x_chart = alt.layer(mc_area, analytical_line).properties(
        width=450, height=250,
        title=f"Distribution of x* (median ≈ {median_x:.1f}, std ≈ {std_x:.1f})"
    )
    return (x_chart,)


@app.cell(hide_code=True)
def _(d_chart, fan_chart, mo, param_text, x_chart):
    math_text = mo.md(r"""
    ## How it works

    Two lines with **fixed** intercepts and **random uniform** slopes:

    - **Blue:** $y = c_b + m_b x, \quad m_b \sim U[a, b]$
    - **Orange:** $y = c_o + m_o x, \quad m_o \sim U[c, d]$

    Setting them equal gives the intersection:

    $$x^* = \frac{c_o - c_b}{m_b - m_o} = \frac{K}{D}$$

    where $K = c_o - c_b$ is a **constant** and $D = m_b - m_o$ is the difference of two uniforms.

    The density of $x^*$ follows from a change of variables:

    $$f_{x^*}(x) = \frac{|K|}{x^2} \cdot f_D\!\left(\frac{K}{x}\right)$$

    **Key insight:** if the slope ranges overlap so that $D$ can be near $0$, then $x^*$ has **heavy tails** ($\propto 1/x^2$) — the intersection can be arbitrarily far away. If the ranges are separated, $x^*$ stays bounded.
    """)

    mo.vstack([
        mo.md("# Where Do Two Random Lines Cross?"),
        mo.hstack([
            mo.vstack([param_text, math_text], gap=0),
            mo.vstack([fan_chart, d_chart, x_chart]),
        ], widths=[1, 1.2]),
    ])
    return


if __name__ == "__main__":
    app.run()
