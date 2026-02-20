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
    import marimo as mo
    import altair as alt
    import polars as pl
    from wigglystuff import TangleSlider

    return TangleSlider, alt, mo, pl


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    buy_slope = mo.ui.anywidget(
        TangleSlider(amount=8.0, min_value=1.0, max_value=16.0, step=0.25, prefix="$", digits=2)
    )
    buy_unc = mo.ui.anywidget(
        TangleSlider(amount=2.0, min_value=0.0, max_value=5.0, step=0.25, prefix="$", digits=2)
    )

    section1_text = mo.md(f"""
    ## Buying bread

    Say you spend {buy_slope} per week on bread at the store —
    give or take {buy_unc}. There's no upfront cost: the line starts at zero
    and climbs steadily. After two years that's a lot of money out the door.
    """)
    return buy_slope, buy_unc, section1_text


@app.cell
def _(buy_slope, buy_unc, pl):
    N_WEEKS = 156
    weeks = list(range(N_WEEKS + 1))

    slope_buy = buy_slope.amount
    unc_buy = buy_unc.amount

    buy_df = pl.DataFrame({
        "week": weeks,
        "cost": [slope_buy * w for w in weeks],
        "low": [max(0.0, (slope_buy - unc_buy) * w) for w in weeks],
        "high": [(slope_buy + unc_buy) * w for w in weeks],
    })
    return N_WEEKS, buy_df, slope_buy, weeks


@app.cell(hide_code=True)
def _(N_WEEKS, alt, buy_df, mo, section1_text):
    _band1 = (
        alt.Chart(buy_df)
        .mark_area(opacity=0.15, color="steelblue")
        .encode(
            x=alt.X("week:Q"),
            y=alt.Y("low:Q"),
            y2=alt.Y2("high:Q"),
        )
    )
    _line1 = (
        alt.Chart(buy_df)
        .mark_line(strokeWidth=2, color="steelblue")
        .encode(
            x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, N_WEEKS], nice=False)),
            y=alt.Y("cost:Q", title="Cumulative cost ($)", scale=alt.Scale(domain=[0, 2000])),
        )
    )
    buy_chart = (
        alt.layer(_band1, _line1)
        .properties(width=520, height=280, title="Cumulative cost of buying bread")
    )

    mo.vstack([section1_text, buy_chart])
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    machine_cost = mo.ui.anywidget(
        TangleSlider(amount=200, min_value=50, max_value=600, step=10, prefix="$", digits=0)
    )
    bake_slope = mo.ui.anywidget(
        TangleSlider(amount=2.75, min_value=0.5, max_value=8.0, step=0.25, prefix="$", digits=2)
    )
    bake_unc = mo.ui.anywidget(
        TangleSlider(amount=0.50, min_value=0.0, max_value=3.0, step=0.25, prefix="$", digits=2)
    )

    section2_text = mo.md(f"""
    ## Baking bread

    A bread machine costs {machine_cost} upfront — that's where your line starts.
    But after that, ingredients run about {bake_slope} per week,
    give or take {bake_unc}. The slope is gentler than buying,
    so the line climbs more slowly.
    """)
    return bake_slope, bake_unc, machine_cost, section2_text


@app.cell
def _(bake_slope, bake_unc, machine_cost, pl, weeks):
    intercept_bake = machine_cost.amount
    slope_bake = bake_slope.amount
    unc_bake = bake_unc.amount

    bake_df = pl.DataFrame({
        "week": weeks,
        "cost": [intercept_bake + slope_bake * w for w in weeks],
        "low": [intercept_bake + max(0.0, slope_bake - unc_bake) * w for w in weeks],
        "high": [intercept_bake + (slope_bake + unc_bake) * w for w in weeks],
    })
    return bake_df, intercept_bake, slope_bake


@app.cell(hide_code=True)
def _(N_WEEKS, alt, bake_df, mo, section2_text):
    _band2 = (
        alt.Chart(bake_df)
        .mark_area(opacity=0.15, color="darkorange")
        .encode(
            x=alt.X("week:Q"),
            y=alt.Y("low:Q"),
            y2=alt.Y2("high:Q"),
        )
    )
    _line2 = (
        alt.Chart(bake_df)
        .mark_line(strokeWidth=2, color="darkorange")
        .encode(
            x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, N_WEEKS], nice=False)),
            y=alt.Y("cost:Q", title="Cumulative cost ($)", scale=alt.Scale(domain=[0, 2000])),
        )
    )
    bake_chart = (
        alt.layer(_band2, _line2)
        .properties(width=520, height=280, title="Cumulative cost of baking bread")
    )

    mo.vstack([section2_text, bake_chart])
    return


@app.cell
def _(intercept_bake, slope_bake, slope_buy):
    delta_slope = slope_buy - slope_bake
    if delta_slope > 1e-9:
        breakeven_week = intercept_bake / delta_slope
        breakeven_cost = slope_buy * breakeven_week
    else:
        breakeven_week = None
        breakeven_cost = None
    return breakeven_cost, breakeven_week


@app.cell(hide_code=True)
def _(breakeven_week, mo):
    if breakeven_week is not None and breakeven_week <= 104:
        _months = breakeven_week * 7 / 30.44
        section3_text = mo.md(f"""
    ## Combining the two

    Put both lines on the same chart. They cross at **week {breakeven_week:.0f}**
    — roughly **{_months:.0f} months** from now. Before that point, the
    machine costs more in total. After it, you're saving money every week.
    """)
    elif breakeven_week is not None:
        section3_text = mo.md(f"""
    ## Combining the two

    With these numbers the lines cross at **week {breakeven_week:.0f}** — beyond the
    two-year window shown. The machine takes a long time to pay off.
    """)
    else:
        section3_text = mo.md("""
    ## Combining the two

    With these numbers, baking costs as much per week as buying — the lines
    run parallel and never cross. Try reducing the baking cost or increasing
    the store price to see a breakeven point.
    """)
    return (section3_text,)


@app.cell(hide_code=True)
def _(
    N_WEEKS,
    alt,
    bake_df,
    breakeven_cost,
    breakeven_week,
    buy_df,
    mo,
    pl,
    section3_text,
):
    _buy = buy_df.with_columns(pl.lit("Buying").alias("strategy"))
    _bake = bake_df.with_columns(pl.lit("Baking").alias("strategy"))
    _combined = pl.concat([_buy, _bake])

    _color_scale = alt.Scale(
        domain=["Buying", "Baking"],
        range=["steelblue", "darkorange"],
    )

    _bands = (
        alt.Chart(_combined)
        .mark_area(opacity=0.15)
        .encode(
            x="week:Q",
            y="low:Q",
            y2="high:Q",
            color=alt.Color("strategy:N", scale=_color_scale, legend=None),
        )
    )
    _lines = (
        alt.Chart(_combined)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, N_WEEKS], nice=False)),
            y=alt.Y("cost:Q", title="Cumulative cost ($)", scale=alt.Scale(domain=[0, 2000])),
            color=alt.Color("strategy:N", scale=_color_scale, title="Strategy"),
        )
    )

    _layers = [_bands, _lines]

    if breakeven_week is not None and 0 < breakeven_week <= N_WEEKS:
        _rule_df = pl.DataFrame({"week": [breakeven_week]})
        _point_df = pl.DataFrame({"week": [breakeven_week], "cost": [breakeven_cost]})

        _rule = (
            alt.Chart(_rule_df)
            .mark_rule(strokeDash=[6, 4], color="gray", strokeWidth=1.5)
            .encode(x="week:Q")
        )
        _point = (
            alt.Chart(_point_df)
            .mark_point(size=120, color="black", filled=True)
            .encode(x="week:Q", y="cost:Q")
        )
        _layers += [_rule, _point]

    combined_chart = (
        alt.layer(*_layers)
        .properties(
            width=520,
            height=320,
            title="Buying vs. Baking: cumulative cost over time",
        )
    )

    mo.vstack([section3_text, combined_chart])
    return


if __name__ == "__main__":
    app.run()
