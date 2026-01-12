# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from widget import GrpoGdpoWidget
    return GrpoGdpoWidget, mo


@app.cell
def _(GrpoGdpoWidget):
    widget = GrpoGdpoWidget()
    return (widget,)


@app.cell
def _(mo, widget):
    widget_view = mo.ui.anywidget(widget)
    return (widget_view,)


@app.cell
def _(widget_view):
    widget_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GRPO vs GDPO Advantage Comparison

    This widget demonstrates the difference between **GRPO** (Group Relative Policy Optimization)
    and **GDPO** (Group reward-Decoupled Policy Optimization) advantage calculations.

    **Click on the reward cells** (0 or 1) to toggle values and see how the advantages change.

    ### Formulas

    **GRPO** first aggregates rewards, then normalizes:

    $$r_j = r_j^{(1)} + r_j^{(2)} + \ldots + r_j^{(n)}$$

    $$A_j^{\text{GRPO}} = \frac{r_j - \mu(r)}{\sigma(r)}$$

    **GDPO** normalizes each reward dimension separately, then sums:

    $$A_j^{(i)} = \frac{r_j^{(i)} - \mu(r^{(i)})}{\sigma(r^{(i)})}$$

    $$A_j^{\text{GDPO}} = A_j^{(1)} + A_j^{(2)} + \ldots + A_j^{(n)}$$

    ### Key Insight

    When different reward combinations produce the same total (e.g., `[1,0,1]` and `[0,1,1]` both sum to 2),
    GRPO assigns identical advantages, while GDPO can distinguish them based on which specific
    rewards were achieved.

    The **Difference** column highlights when GDPO preserves information that GRPO loses.
    """)
    return


if __name__ == "__main__":
    app.run()
