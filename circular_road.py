# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget==0.9.21",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from circular_road_widget import CircularRoadWidget
    return CircularRoadWidget, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Circular Road Widget

    A car drives around a circular road using Bret Victor's steering model:

    At each step:
    1. Move forward 1 pixel
    2. If left of road (inside circle), turn right by steering angle
    3. If right of road (outside circle), turn left by steering angle

    - **Steering angle**: How sharply the car corrects when off-road
    - **Road width**: How wide the road is (affects when corrections happen)

    Drag the car along its path to see different positions.
    """)
    return


@app.cell
def _(mo):
    angle_slider = mo.ui.slider(
        start=0.1, stop=2.4, value=2, step=0.01, label="Steering angle (degrees/step)"
    )
    road_width_slider = mo.ui.slider(
        start=10, stop=80, value=40, step=5, label="Road width (pixels)"
    )
    mo.hstack([angle_slider, road_width_slider])
    return angle_slider, road_width_slider


@app.cell
def _(CircularRoadWidget, angle_slider, mo, road_width_slider):
    widget = CircularRoadWidget(angle=angle_slider.value, road_width=road_width_slider.value)
    widget_view = mo.ui.anywidget(widget)
    widget_view
    return (widget_view,)


@app.cell
def _(mo, widget_view):
    mo.md(f"""
    **Car position:** {widget_view.widget.position:.2%} around the track
    **Total path length:** {widget_view.widget.total_length:.1f} units
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
