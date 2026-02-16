# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "marimo",
#     "polars>=1.18.0",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import math

    import marimo as mo
    import altair as alt
    import polars as pl
    from wigglystuff import TangleSlider

    return TangleSlider, alt, math, mo, pl


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    bread_price = mo.ui.anywidget(TangleSlider(amount=4.0, min_value=2.0, max_value=8.0, step=0.01, suffix=" dollar", digits=2))
    store_loaves = mo.ui.anywidget(TangleSlider(amount=1.0, min_value=0.5, max_value=5.0, step=0.5, digits=1))
    store_loaves_unc = mo.ui.anywidget(TangleSlider(amount=0.5, min_value=0.0, max_value=2.0, step=0.25, digits=2))

    store_text = mo.md(f"""
    **Store-Bought Bread**

    Every week you buy about {store_loaves} loaves of bread at {bread_price} each.
    Some weeks you buy more, some less — the real number varies by about ±{store_loaves_unc} loaves per week.
    """)
    return bread_price, store_loaves, store_loaves_unc, store_text


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    machine_cost = mo.ui.anywidget(TangleSlider(amount=200, min_value=50, max_value=500, step=1, prefix="$", digits=0))
    flour_g = mo.ui.anywidget(TangleSlider(amount=500, min_value=300, max_value=700, step=50, digits=0))
    flour_price = mo.ui.anywidget(TangleSlider(amount=1.50, min_value=0.50, max_value=4.0, step=0.01, prefix="$", digits=2))
    yeast_g = mo.ui.anywidget(TangleSlider(amount=7, min_value=3, max_value=15, step=1, digits=0))
    yeast_price = mo.ui.anywidget(TangleSlider(amount=3.00, min_value=1.0, max_value=8.0, step=0.01, prefix="$", digits=2))
    salt_price = mo.ui.anywidget(TangleSlider(amount=1.00, min_value=0.50, max_value=3.0, step=0.01, prefix="$", digits=2))
    butter_g = mo.ui.anywidget(TangleSlider(amount=30, min_value=0, max_value=60, step=10, digits=0))
    butter_price = mo.ui.anywidget(TangleSlider(amount=3.00, min_value=1.50, max_value=6.0, step=0.01, prefix="$", digits=2))
    electricity_kwh = mo.ui.anywidget(TangleSlider(amount=0.4, min_value=0.1, max_value=1.0, step=0.05, digits=2))
    electricity_price = mo.ui.anywidget(TangleSlider(amount=0.15, min_value=0.05, max_value=0.50, step=0.01, prefix="$", digits=2))
    home_loaves = mo.ui.anywidget(TangleSlider(amount=1.0, min_value=0.5, max_value=5.0, step=0.5, digits=1))
    home_loaves_unc = mo.ui.anywidget(TangleSlider(amount=0.5, min_value=0.0, max_value=2.0, step=0.25, digits=2))
    ingredient_unc = mo.ui.anywidget(TangleSlider(amount=10, min_value=0, max_value=30, step=5, suffix="%", digits=0))
    annual_inflation = mo.ui.anywidget(TangleSlider(amount=3.0, min_value=0.0, max_value=10.0, step=0.5, suffix="%", digits=1))

    machine_text = mo.md(f"""
    **Bread Machine**

    A bread machine costs {machine_cost}. Each loaf uses roughly:

    - {flour_g}g of flour at {flour_price}/kg
    - {yeast_g}g of yeast at {yeast_price}/100g
    - a pinch of salt at {salt_price}/kg
    - {butter_g}g of butter at {butter_price}/250g
    - {electricity_kwh} kWh of electricity per bake at {electricity_price}/kWh

    With your own machine you'd bake about {home_loaves} loaves per week, give or take ±{home_loaves_unc}. Ingredient prices can vary by about ±{ingredient_unc}. Assume annual inflation of {annual_inflation}.
    """)
    return (
        annual_inflation,
        butter_g,
        butter_price,
        electricity_kwh,
        electricity_price,
        flour_g,
        flour_price,
        home_loaves,
        home_loaves_unc,
        ingredient_unc,
        machine_cost,
        machine_text,
        salt_price,
        yeast_g,
        yeast_price,
    )


@app.cell(hide_code=True)
def _(mo):
    weeks_ahead = mo.ui.slider(start=26, stop=520, step=26, value=104, label="Weeks ahead", show_value=True)
    return (weeks_ahead,)


@app.cell
def _(
    annual_inflation,
    bread_price,
    butter_g,
    butter_price,
    electricity_kwh,
    electricity_price,
    flour_g,
    flour_price,
    home_loaves,
    home_loaves_unc,
    ingredient_unc,
    machine_cost,
    math,
    pl,
    salt_price,
    store_loaves,
    store_loaves_unc,
    weeks_ahead,
    yeast_g,
    yeast_price,
):
    import random

    n_weeks = weeks_ahead.value
    weeks = list(range(n_weeks + 1))

    # Weekly inflation multiplier from annual rate
    weekly_inflation = (1 + annual_inflation.amount / 100) ** (1 / 52)

    # Ingredient cost per loaf (base, before inflation)
    cost_flour = (flour_g.amount / 1000) * flour_price.amount
    cost_yeast = (yeast_g.amount / 100) * yeast_price.amount
    cost_salt = 0.005 * salt_price.amount  # ~5g per loaf
    cost_butter = (butter_g.amount / 250) * butter_price.amount
    cost_electricity = electricity_kwh.amount * electricity_price.amount
    cost_per_loaf = cost_flour + cost_yeast + cost_salt + cost_butter + cost_electricity

    # Uniform distribution half-ranges
    unc_frac = ingredient_unc.amount / 100
    store_half = store_loaves_unc.amount
    home_half = home_loaves_unc.amount

    # Weekly costs: min / mid / max from uniform bounds
    store_weekly_mid = bread_price.amount * store_loaves.amount
    store_weekly_min = bread_price.amount * max(store_loaves.amount - store_half, 0)
    store_weekly_max = bread_price.amount * (store_loaves.amount + store_half)

    machine_weekly_mid = cost_per_loaf * home_loaves.amount
    machine_weekly_min = cost_per_loaf * (1 - unc_frac) * max(home_loaves.amount - home_half, 0)
    machine_weekly_max = cost_per_loaf * (1 + unc_frac) * (home_loaves.amount + home_half)

    mc = machine_cost.amount

    # Cumulative costs with inflation (true uniform bounds)
    store_mid = []
    store_lo = []
    store_hi = []
    machine_mid = []
    machine_lo = []
    machine_hi = []
    breakeven_week = None

    for _w in weeks:
        _gw = (weekly_inflation ** _w - 1) / (weekly_inflation - 1) if weekly_inflation > 1.0001 else float(_w)
        store_mid.append(store_weekly_mid * _gw)
        store_lo.append(store_weekly_min * _gw)
        store_hi.append(store_weekly_max * _gw)
        machine_mid.append(mc + machine_weekly_mid * _gw)
        machine_lo.append(mc + machine_weekly_min * _gw)
        machine_hi.append(mc + machine_weekly_max * _gw)

        if breakeven_week is None and _w > 0 and store_weekly_mid * _gw >= mc + machine_weekly_mid * _gw:
            breakeven_week = _w

    savings_at_end = store_mid[-1] - machine_mid[-1]

    # Analytical break-even bounds from uniform extremes
    savings_max = store_weekly_max - machine_weekly_min
    savings_min = store_weekly_min - machine_weekly_max

    def _breakeven_week(savings):
        if weekly_inflation > 1.0001:
            return math.log(mc * (weekly_inflation - 1) / savings + 1) / math.log(weekly_inflation)
        return mc / savings

    w_earliest = _breakeven_week(savings_max) if savings_max > 0 else None
    w_latest = _breakeven_week(savings_min) if savings_min > 0 else None

    # Monte Carlo break-even distribution (respects uniform input bounds)
    random.seed(42)
    n_sim = 200_000
    breakeven_samples = []
    n_never = 0
    for _ in range(n_sim):
        _sl = store_loaves.amount + random.uniform(-store_half, store_half)
        _hl = home_loaves.amount + random.uniform(-home_half, home_half)
        _im = 1.0 + random.uniform(-unc_frac, unc_frac)
        _savings = bread_price.amount * max(_sl, 0) - cost_per_loaf * _im * max(_hl, 0)
        if _savings > 0:
            breakeven_samples.append(_breakeven_week(_savings))
        else:
            n_never += 1

    frac_breakeven = len(breakeven_samples) / n_sim

    if breakeven_samples:
        breakeven_mean = sum(breakeven_samples) / len(breakeven_samples)
        breakeven_std = (sum((_bw - breakeven_mean) ** 2 for _bw in breakeven_samples) / len(breakeven_samples)) ** 0.5
        # Histogram only within the visible range [0, w_latest or n_weeks]
        _hist_max = min(w_latest, n_weeks) if w_latest is not None else n_weeks
        _hist_min = w_earliest if w_earliest is not None else min(breakeven_samples)
        _visible = [w for w in breakeven_samples if w <= _hist_max]
        _n_bins = 150
        _bin_w = (_hist_max - _hist_min) / _n_bins
        _counts = [0] * _n_bins
        for _bw in _visible:
            _idx = min(int((_bw - _hist_min) / _bin_w), _n_bins - 1)
            _counts[_idx] += 1
        _total = n_sim * _bin_w  # normalize against ALL samples, not just visible
        breakeven_pdf_df = pl.DataFrame({
            "week": [_hist_min + (i + 0.5) * _bin_w for i in range(_n_bins)],
            "density": [c / _total for c in _counts],
        })
    else:
        breakeven_mean = None
        breakeven_std = None
        breakeven_pdf_df = None

    n = len(weeks)
    cost_df = pl.DataFrame({
        "week": weeks * 2,
        "cost": store_mid + machine_mid,
        "low": store_lo + machine_lo,
        "high": store_hi + machine_hi,
        "strategy": ["Store-bought"] * n + ["Bread machine"] * n,
    })
    return (
        breakeven_mean,
        breakeven_pdf_df,
        breakeven_std,
        breakeven_week,
        cost_df,
        cost_per_loaf,
        frac_breakeven,
        n_weeks,
        savings_at_end,
        w_latest,
    )


@app.cell
def _(alt, breakeven_week, cost_df, n_weeks, pl):
    _layers = [
        alt.Chart(cost_df)
        .mark_area(opacity=0.15)
        .encode(
            x="week:Q",
            y="low:Q",
            y2="high:Q",
            color=alt.Color("strategy:N", legend=None),
        ),
        alt.Chart(cost_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, n_weeks])),
            y=alt.Y("cost:Q", title="Cumulative Cost ($)"),
            color=alt.Color("strategy:N", title="Strategy"),
        ),
    ]

    if breakeven_week is not None and breakeven_week <= n_weeks:
        _be_cost = cost_df.filter(
            (pl.col("week") == breakeven_week) & (pl.col("strategy") == "Store-bought")
        )["cost"][0]
        _layers.append(
            alt.Chart(pl.DataFrame({"week": [breakeven_week]}))
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(x="week:Q")
        )
        _layers.append(
            alt.Chart(pl.DataFrame({"week": [breakeven_week], "cost": [_be_cost]}))
            .mark_point(size=100, color="black", filled=True)
            .encode(x="week:Q", y="cost:Q")
        )

    cumulative_chart = alt.layer(*_layers).properties(
        width=400,
        height=280,
        title="Cumulative Cost: Store-Bought vs Bread Machine",
    )
    return (cumulative_chart,)


@app.cell
def _(
    alt,
    breakeven_mean,
    breakeven_pdf_df,
    breakeven_week,
    frac_breakeven,
    n_weeks,
    w_latest,
):
    if breakeven_pdf_df is not None:
        _pct = f", {frac_breakeven:.0%} break even" if frac_breakeven < 0.999 else ""
        _domain_max = min(w_latest, n_weeks) if w_latest is not None else n_weeks
        breakeven_chart = (
            alt.Chart(breakeven_pdf_df)
            .mark_area(opacity=0.3, color="steelblue", clip=True)
            .encode(
                x=alt.X("week:Q", title="Break-even week", scale=alt.Scale(domain=[0, _domain_max])),
                y=alt.Y("density:Q", title="Density"),
            )
            .properties(
                width=400,
                height=200,
                title=f"Break-even distribution (mean≈{breakeven_mean:.0f} weeks{_pct})",
            )
        )
    elif breakeven_week is not None:
        breakeven_chart = (
            alt.Chart({"values": [{"week": breakeven_week}]})
            .mark_rule(strokeWidth=2, color="steelblue")
            .encode(x="week:Q")
            .properties(width=400, height=200, title=f"Break-even at week {breakeven_week} (no uncertainty)")
        )
    else:
        breakeven_chart = None
    return (breakeven_chart,)


@app.cell(hide_code=True)
def _(
    breakeven_std,
    breakeven_week,
    cost_per_loaf,
    mo,
    n_weeks,
    savings_at_end,
):
    _be_text = f"Week {breakeven_week} (~{breakeven_week * 7 / 30:.0f} months)" if breakeven_week is not None else "—"
    if breakeven_std is not None and breakeven_std > 0:
        _be_text += f" ± {breakeven_std:.0f} weeks"

    summary_text = mo.md(f"""
    ## Summary

    | | |
    |---|---|
    | **Ingredient cost per loaf** | ${cost_per_loaf:.2f} |
    | **Break-even point** | {_be_text} |
    | **Savings after {n_weeks} weeks** | ${savings_at_end:,.2f} |

    The shaded bands show the full range of uncertainty from consumption and ingredient price variation.
    """)
    return (summary_text,)


@app.cell(hide_code=True)
def _(
    breakeven_chart,
    cumulative_chart,
    machine_text,
    mo,
    store_text,
    summary_text,
    weeks_ahead,
):
    _title = mo.md(r"""
    ### Bread Machine: When Does It Pay Off?

    Buying bread every week adds up. A bread machine costs money upfront, but the ingredients are cheap. Let's figure out when the investment breaks even.
    """)

    _right_items = [cumulative_chart]
    if breakeven_chart is not None:
        _right_items.append(breakeven_chart)

    mo.hstack([
        mo.vstack([_title, store_text, machine_text, weeks_ahead, summary_text], gap=0),
        mo.vstack(_right_items),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix: Break-Even Distribution

    ### Savings as a random variable

    Let $S$ denote the weekly savings from using the bread machine:

    $$S = p \cdot X - c \cdot Y \cdot Z$$

    where $p$ is the store bread price, $c$ is the ingredient cost per loaf, and $X$ (store loaves/week), $Y$ (home loaves/week), $Z$ (ingredient price multiplier) are independent **uniform** random variables. The sliders define their ranges.

    ### Break-even as a function of savings

    With weekly inflation multiplier $r > 1$, cumulative store cost after $w$ weeks is the geometric series $p X \cdot \frac{r^w - 1}{r - 1}$, and similarly for the machine. Setting cumulative costs equal gives:

    $$W = \frac{\log\!\bigl(C(r-1)/S + 1\bigr)}{\log r}$$

    Since the inputs are uniform with hard bounds, $S$ also has hard bounds — which means $W$ has a finite maximum. The break-even distribution is computed via Monte Carlo: we draw 200,000 samples of $(X, Y, Z)$ from their uniform ranges, compute $S$ for each, and apply the formula above to get a sample of break-even weeks. The shaded bands on the cumulative cost chart show the true min/max range from the same uniform distributions, so the break-even distribution's support matches exactly the region where the bands overlap.
    """)
    return


if __name__ == "__main__":
    app.run()
