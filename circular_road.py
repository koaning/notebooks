# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget==0.9.21",
#     "traitlets",
#     "numpy==2.4.1",
#     "polars==1.37.1",
#     "matplotlib==3.10.8",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from circular_road_widget import CircularRoadWidget
    return CircularRoadWidget, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Circular Road Widget - Ladder of Abstraction

    A car drives around a circular road using Bret Victor's steering model:

    At each step:
    1. Move forward 1 pixel
    2. If left of road (inside circle), turn right by steering angle
    3. If right of road (outside circle), turn left by steering angle

    The widget simulates **all parameter combinations** simultaneously:
    - Faded paths show all possible trajectories
    - The highlighted path shows your current selection
    - Drag the puck on the heatmap to explore different parameters
    """)
    return


@app.cell
def _(np):
    # Define the parameter space to explore
    all_angles = np.arange(0.5, 3.0, 0.25).tolist()
    all_road_widths = [20.0, 30.0, 40.0, 50.0, 60.0]
    return all_angles, all_road_widths


@app.cell
def _(CircularRoadWidget, all_angles, all_road_widths, mo):
    # Data widget - computes path_data for all parameter combinations
    # This widget's path_data will be used for the heatmap
    data_widget = CircularRoadWidget(
        angles=all_angles,
        road_widths=all_road_widths,
        selected_angle=all_angles[0],
        selected_road_width=all_road_widths[0],
    )
    data_widget_view = mo.ui.anywidget(data_widget)
    data_widget_view
    return (data_widget_view,)


@app.cell(hide_code=True)
def _(data_widget_view):
    import polars as pl

    # Get path data from the widget (computed in JS)
    path_data_df = pl.DataFrame(data_widget_view.widget.path_data)
    path_data_df
    return (path_data_df,)


@app.cell
def _(all_angles, all_road_widths, mo, np, path_data_df):
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck

    # Create a 2D array for the heatmap
    grid = np.zeros((len(all_road_widths), len(all_angles)))
    for row in path_data_df.iter_rows(named=True):
        angle_idx = all_angles.index(row['angle'])
        width_idx = all_road_widths.index(row['road_width'])
        grid[width_idx, angle_idx] = row['total_length']

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')

    # Set tick labels
    ax.set_xticks(range(len(all_angles)))
    ax.set_xticklabels([f"{a:.2f}" for a in all_angles], rotation=45, ha='right')
    ax.set_yticks(range(len(all_road_widths)))
    ax.set_yticklabels([f"{w:.0f}" for w in all_road_widths])

    ax.set_xlabel("Steering Angle (degrees/step)")
    ax.set_ylabel("Road Width (pixels)")
    ax.set_title("Path Length by Parameters - Drag the puck to explore!")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Total Path Length")

    plt.tight_layout()

    # Create puck at center of parameter space
    puck = ChartPuck(fig, x=len(all_angles) // 2, y=len(all_road_widths) // 2)
    puck_widget = mo.ui.anywidget(puck)
    puck_widget
    return (puck_widget,)


@app.cell
def _(all_angles, all_road_widths, puck_widget):
    # Convert puck position to nearest valid parameter indices
    # puck.x and puck.y are lists even for single puck
    puck_x = puck_widget.widget.x[0] if isinstance(puck_widget.widget.x, list) else puck_widget.widget.x
    puck_y = puck_widget.widget.y[0] if isinstance(puck_widget.widget.y, list) else puck_widget.widget.y

    angle_idx = int(round(max(0, min(len(all_angles) - 1, puck_x))))
    width_idx = int(round(max(0, min(len(all_road_widths) - 1, puck_y))))

    selected_angle = all_angles[angle_idx]
    selected_road_width = all_road_widths[width_idx]
    return selected_angle, selected_road_width


@app.cell
def _(
    CircularRoadWidget,
    all_angles,
    all_road_widths,
    mo,
    selected_angle,
    selected_road_width,
):
    # Interactive widget - selection controlled by the puck
    widget = CircularRoadWidget(
        angles=all_angles,
        road_widths=all_road_widths,
        selected_angle=selected_angle,
        selected_road_width=selected_road_width,
    )
    widget_view = mo.ui.anywidget(widget)
    widget_view
    return (widget_view,)


@app.cell
def _(mo, path_data_df, selected_angle, selected_road_width, widget_view):
    # Find the selected path's length from path_data
    selected_row = path_data_df.filter(
        (path_data_df['angle'] == selected_angle) &
        (path_data_df['road_width'] == selected_road_width)
    )
    path_length = selected_row['total_length'][0] if len(selected_row) > 0 else 0

    mo.md(f"""
    **Selected:** angle={selected_angle:.2f}deg, road_width={selected_road_width:.0f}px

    **Car position:** {widget_view.widget.position:.2%} around the track

    **Path length:** {path_length:.1f} units
    """)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
