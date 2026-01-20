# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.10.19"
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
        x_axis={"name": "Temperature", "values": [20, 25, 30, 35, 40]},
        y_axis={"name": "Pressure", "values": [0, 25, 50, 75, 100]},
        z_axis={"name": "Time", "values": [0, 0.5, 1, 1.5, 2, 2.5, 3]},
    )
    cube_view = mo.ui.anywidget(cube)
    cube_view
    return cube, cube_view


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
