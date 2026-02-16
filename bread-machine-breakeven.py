# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "marimo",
#     "polars>=1.18.0",
#     "uncertainties==3.2.3",
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
    from uncertainties import ufloat
    from uncertainties.umath import log as ulog
    from wigglystuff import TangleSlider

    return TangleSlider, alt, math, mo, pl, ufloat, ulog


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
    ufloat,
    ulog,
    weeks_ahead,
    yeast_g,
    yeast_price,
):
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

    # Uncertain values (uniform dist std_dev = half_range / sqrt(3))
    unc_frac = ingredient_unc.amount / 100
    store_loaves_u = ufloat(store_loaves.amount, store_loaves_unc.amount / math.sqrt(3))
    home_loaves_u = ufloat(home_loaves.amount, home_loaves_unc.amount / math.sqrt(3))
    ingr_mult_u = ufloat(1.0, unc_frac / math.sqrt(3))

    # Weekly costs with uncertainty
    store_weekly_u = bread_price.amount * store_loaves_u
    machine_weekly_u = cost_per_loaf * ingr_mult_u * home_loaves_u
    mc = machine_cost.amount

    # Cumulative costs with inflation and propagated uncertainty
    def _nom(x):
        return x.nominal_value if hasattr(x, 'nominal_value') else float(x)

    def _std(x):
        return x.std_dev if hasattr(x, 'std_dev') else 0.0

    store_mid = []
    store_lo = []
    store_hi = []
    machine_mid = []
    machine_lo = []
    machine_hi = []

    s_cum = 0.0
    m_cum = float(mc)
    breakeven_week = None

    for _w in weeks:
        _sn, _ss = _nom(s_cum), _std(s_cum)
        _mn, _ms = _nom(m_cum), _std(m_cum)
        store_mid.append(_sn)
        store_lo.append(_sn - _ss)
        store_hi.append(_sn + _ss)
        machine_mid.append(_mn)
        machine_lo.append(_mn - _ms)
        machine_hi.append(_mn + _ms)

        if breakeven_week is None and _w > 0 and _sn >= _mn:
            breakeven_week = _w

        _infl = weekly_inflation ** _w
        s_cum = s_cum + store_weekly_u * _infl
        m_cum = m_cum + machine_weekly_u * _infl

    # Break-even with uncertainty (closed-form geometric series)
    savings_weekly_u = store_weekly_u - machine_weekly_u
    if savings_weekly_u.nominal_value > 0:
        if weekly_inflation > 1.0001:
            breakeven_u = ulog(mc * (weekly_inflation - 1) / savings_weekly_u + 1) / math.log(weekly_inflation)
        else:
            breakeven_u = mc / savings_weekly_u
    else:
        breakeven_u = None

    savings_at_end = store_mid[-1] - machine_mid[-1]

    # Lognormal parameters from break-even mean/std (always positive, right-skewed)
    if breakeven_u is not None and breakeven_u.std_dev > 0:
        _be_mu = breakeven_u.nominal_value
        _be_sigma = breakeven_u.std_dev
        _cv = _be_sigma / _be_mu
        sigma_ln = math.sqrt(math.log(1 + _cv ** 2))
        mu_ln = math.log(_be_mu) - sigma_ln ** 2 / 2
        # x range: median * exp(±3σ_ln)
        x_max_dist = math.exp(mu_ln + 3 * sigma_ln)
    else:
        sigma_ln = None
        mu_ln = None
        x_max_dist = 0.0

    # Shared x-axis upper bound for both charts
    x_max_shared = max(n_weeks, x_max_dist)

    n = len(weeks)
    cost_df = pl.DataFrame({
        "week": weeks * 2,
        "cost": store_mid + machine_mid,
        "low": store_lo + machine_lo,
        "high": store_hi + machine_hi,
        "strategy": ["Store-bought"] * n + ["Bread machine"] * n,
    })
    return (
        breakeven_u,
        breakeven_week,
        cost_df,
        cost_per_loaf,
        mu_ln,
        n_weeks,
        savings_at_end,
        sigma_ln,
        x_max_shared,
    )


@app.cell
def _(alt, breakeven_week, cost_df, n_weeks, pl, x_max_shared):
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
            x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, x_max_shared])),
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
def _(alt, breakeven_u, math, mu_ln, pl, sigma_ln, x_max_shared):
    if mu_ln is not None and sigma_ln is not None:
        _mu = breakeven_u.nominal_value
        # Lognormal PDF: always positive, right-skewed
        _x_min = max(1, math.exp(mu_ln - 3 * sigma_ln))
        _x_max = math.exp(mu_ln + 3 * sigma_ln)
        _n_pts = 200
        _step = (_x_max - _x_min) / _n_pts
        _xs = [_x_min + _i * _step for _i in range(_n_pts + 1)]
        _ys = [
            math.exp(-0.5 * ((math.log(_x) - mu_ln) / sigma_ln) ** 2)
            / (_x * sigma_ln * math.sqrt(2 * math.pi))
            for _x in _xs
        ]
        _df = pl.DataFrame({"week": _xs, "density": _ys})
        breakeven_chart = (
            alt.Chart(_df)
            .mark_area(opacity=0.3, color="steelblue")
            .encode(
                x=alt.X("week:Q", title="Break-even week", scale=alt.Scale(domain=[0, x_max_shared])),
                y=alt.Y("density:Q", title="Density"),
            )
            .properties(
                width=400,
                height=200,
                title=f"Break-even distribution (mean≈{_mu:.0f} weeks)",
            )
        )
    elif breakeven_u is not None:
        _w_val = breakeven_u.nominal_value
        breakeven_chart = (
            alt.Chart(pl.DataFrame({"week": [_w_val]}))
            .mark_rule(strokeWidth=2, color="steelblue")
            .encode(x=alt.X("week:Q", scale=alt.Scale(domain=[0, x_max_shared])))
            .properties(width=400, height=200, title=f"Break-even at week {_w_val:.0f} (no uncertainty)")
        )
    else:
        breakeven_chart = None
    return (breakeven_chart,)


@app.cell(hide_code=True)
def _(breakeven_u, breakeven_week, cost_per_loaf, mo, n_weeks, savings_at_end):
    _be_text = f"Week {breakeven_week} (~{breakeven_week * 7 / 30:.0f} months)" if breakeven_week is not None else "—"
    if breakeven_u is not None and breakeven_u.std_dev > 0:
        _be_text += f" ± {breakeven_u.std_dev:.0f} weeks"

    summary_text = mo.md(f"""
    ## Summary

    | | |
    |---|---|
    | **Ingredient cost per loaf** | ${cost_per_loaf:.2f} |
    | **Break-even point** | {_be_text} |
    | **Savings after {n_weeks} weeks** | ${savings_at_end:,.2f} |

    The shaded bands show ±1σ uncertainty from consumption and ingredient price variation.
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


if __name__ == "__main__":
    app.run()
