# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.0.0",
#     "marimo",
#     "polars>=1.18.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import math
    return alt, math, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Red Ball Riddle

    You have a bag containing **99 red balls** and **1 blue ball** (100 balls total).

    Right now, **99% of the balls are red**.

    **The question:** How many red balls do you need to *remove* to make exactly **98%** of the balls red?

    Use the slider above to explore... the answer might surprise you!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    red_slider = mo.ui.slider(1, 99, 1, value=99, label="Number of red balls", show_value=True)
    red_slider
    return (red_slider,)


@app.cell(hide_code=True)
def _(alt, math, pl, red_slider):
    # Get current values
    n_red = red_slider.value
    n_blue = 1
    total = n_red + n_blue
    pct_red = (n_red / total) * 100

    # Calculate grid dimensions
    cols = math.ceil(math.sqrt(total))

    # Create ball positions
    positions = []
    for i in range(total):
        x = i % cols
        y = i // cols
        color = "red" if i < n_red else "blue"
        positions.append({"x": x, "y": y, "color": color})

    df = pl.DataFrame(positions)

    # Create the grid visualization
    chart = (
        alt.Chart(df)
        .mark_circle(size=200)
        .encode(
            x=alt.X("x:O", axis=None),
            y=alt.Y("y:O", axis=None, sort="descending"),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["red", "blue"], range=["#e74c3c", "#3498db"]),
                legend=None,
            ),
        )
        .properties(
            width=400,
            height=400,
            title=f"{n_red} red + {n_blue} blue = {total} balls ({pct_red:.1f}% red)",
        )
        .configure_view(strokeWidth=0)
    )

    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Surprising Answer

    To go from **99% red** to **98% red**, you need to remove **50 red balls**!

    Here's the math:
    - You always have 1 blue ball
    - For 98% to be red, the blue ball must be 2% of the total
    - If 1 ball = 2%, then total = 50 balls
    - So you need 49 red balls + 1 blue ball = 50 balls
    - You started with 99 red balls, so remove 99 - 49 = **50 balls**

    That's right: removing just 1 percentage point (99% → 98%) requires removing *half* of your red balls!
    """)
    return


@app.cell
def _(alt, pl):
    # Deep dive: the non-linear relationship
    # For each target percentage, how many red balls do we need to remove?

    _data = []
    for _target_pct in range(50, 100):
        # With 1 blue ball, to achieve _target_pct red:
        # red / (red + 1) = _target_pct / 100
        # red = _target_pct / (100 - _target_pct)
        _red_needed = _target_pct / (100 - _target_pct)
        _balls_to_remove = 99 - _red_needed
        _data.append(
            {
                "target_pct": _target_pct,
                "red_needed": _red_needed,
                "balls_to_remove": _balls_to_remove,
            }
        )

    cost_df = pl.DataFrame(_data)

    cost_chart = (
        alt.Chart(cost_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("target_pct:Q", title="Target % Red", scale=alt.Scale(domain=[50, 99])),
            y=alt.Y("balls_to_remove:Q", title="Red Balls to Remove (from 99)"),
        )
        .properties(
            width=500, height=300, title="The Non-Linear Cost of Reducing Red Ball Percentage"
        )
    )

    cost_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Asymmetry

    What if instead of removing red balls, we could *add* blue balls?

    To go from 99% → 98% red:
    - **Removing red:** Remove 50 balls (99 → 49 red)
    - **Adding blue:** Add just 1 ball (1 → 2 blue)

    The chart above shows this dramatic asymmetry. Adding blue balls is almost always
    more "efficient" at changing the percentage - but that's exactly why the original
    puzzle feels so counterintuitive!

    We instinctively think about *adding* things to dilute a mixture, not *removing*
    the dominant component.
    """)
    return


@app.cell
def _(alt, pl):
    # Deep dive: asymmetry - removing red vs adding blue

    _asymmetry_data = []
    for _target_pct in range(50, 100):
        # Method 1: Remove red balls (keep 1 blue)
        _red_needed = _target_pct / (100 - _target_pct)
        _red_to_remove = max(0, 99 - _red_needed)

        # Method 2: Add blue balls (keep 99 red)
        # 99 / (99 + blue) = _target_pct / 100
        # blue = 99 * (100 - _target_pct) / _target_pct
        _blue_to_add = 99 * (100 - _target_pct) / _target_pct

        _asymmetry_data.append(
            {"target_pct": _target_pct, "action": "Remove red balls", "count": _red_to_remove}
        )
        _asymmetry_data.append(
            {"target_pct": _target_pct, "action": "Add blue balls", "count": _blue_to_add}
        )

    asym_df = pl.DataFrame(_asymmetry_data)

    asym_chart = (
        alt.Chart(asym_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("target_pct:Q", title="Target % Red", scale=alt.Scale(domain=[50, 99])),
            y=alt.Y("count:Q", title="Number of Balls"),
            color=alt.Color("action:N", title="Method"),
        )
        .properties(width=500, height=300, title="Two Ways to Reduce Red Ball Percentage")
    )

    asym_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The General Formula

    For any starting configuration with $R$ red balls and $B$ blue balls,
    to achieve a target percentage $P$ (as a decimal) of red balls:

    $$\text{Red balls needed} = \frac{B \cdot P}{1 - P}$$

    $$\text{Balls to remove} = R - \frac{B \cdot P}{1 - P}$$

    **Example:** Starting with 99 red, 1 blue, targeting 98% (P = 0.98):

    $$\text{Red needed} = \frac{1 \times 0.98}{1 - 0.98} = \frac{0.98}{0.02} = 49$$

    $$\text{Balls to remove} = 99 - 49 = 50$$

    The key insight: the denominator $(1 - P)$ becomes very small as $P$ approaches 1,
    making the required number of red balls explode. This is why maintaining very high
    percentages is so expensive!
    """)
    return


if __name__ == "__main__":
    app.run()
