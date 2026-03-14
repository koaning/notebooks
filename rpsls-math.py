# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anywidget>=0.9.0",
#     "marimo",
#     "traitlets>=5.0.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from rpsls_widget.rpsls_widget import RpslsWidget

    return RpslsWidget, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rock, Paper, Scissors ... and beyond?

    Rock-Paper-Scissors has **3** elements. Rock-Paper-Scissors-Lizard-Spock has **5**.
    Could we make a variant with **4**? Or **6**?

    For a game to be *fair*, every element must beat the same number of others.
    With $n$ elements, each must beat exactly $k = \frac{n - 1}{2}$ others.
    That's only a whole number when $n$ is **odd**.

    Use the slider to explore: odd values produce a perfectly balanced tournament,
    while even values show why perfect balance is impossible.
    """)
    return


@app.cell
def _(RpslsWidget, mo, slider):
    widget = mo.ui.anywidget(RpslsWidget(n=slider.value))
    return (widget,)


@app.cell
def _(mo):
    slider = mo.ui.slider(3, 45, value=5, label="Number of elements (n)", show_value=True)
    return (slider,)


@app.cell
def _(widget):
    widget.animate_highlight("A")
    return


@app.cell
def _(widget):
    widget.animate_node(1)
    return


@app.cell
def _(mo, slider, widget):
    mo.vstack([slider, widget])
    return


@app.cell(hide_code=True)
def _(widget):
    widget
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
