# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from cube_widget.cube_widget import CubeWidget
    return (CubeWidget,)


@app.cell
def _(CubeWidget, mo):
    cube = CubeWidget(
        x_axis={"name": "Temp", "values": [i for i in range(20, 41)]},
        y_axis={"name": "Pressure", "values": [i * 5 for i in range(21)]},
        z_axis={"name": "Time", "values": [i * 0.1 for i in range(31)]},
    )
    cube_view = mo.ui.anywidget(cube)
    cube_view
    return (cube,)


@app.cell
def _(cube, mo):
    mo.md(f"""
    ## Widget State

    - **Plane**: {cube.plane}
    - **Line**: {cube.line}
    - **Point**: {cube.point}

    ---

    **Locked order**: {cube.locked_order}

    **Axis values**: {cube.axis_values}
    """)
    return


if __name__ == "__main__":
    app.run()
