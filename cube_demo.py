# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "traitlets",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
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
    cube_view = mo.ui.anywidget(CubeWidget(
        x_axis={"name": "Angle", "values": [i for i in range(0, 90, 5)]},
        y_axis={"name": "Force", "values": [i * 5 for i in range(21)]},
        z_axis={"name": "Time", "values": [i * 0.5 for i in range(31)]},
    ))
    return (cube_view,)


@app.cell
def _(cube_view, mo):
    cube = cube_view.widget
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


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell
def _(np):
    def calculate_trajectory(angle_deg, initial_velocity, time_points):
        """
        Calculate cannonball trajectory using projectile motion equations.

        Parameters:
        - angle_deg: Launch angle in degrees (0-90)
        - initial_velocity: Initial velocity in m/s (we treat force as velocity)
        - time_points: Array of time values to calculate positions for

        Returns:
        - x: Horizontal positions
        - y: Vertical positions (clamped to >= 0)

        Physics:
        - x(t) = v₀ * cos(θ) * t
        - y(t) = v₀ * sin(θ) * t - 0.5 * g * t²
        """
        g = 9.8  # gravitational acceleration (m/s²)
        angle_rad = np.radians(angle_deg)

        v0x = initial_velocity * np.cos(angle_rad)
        v0y = initial_velocity * np.sin(angle_rad)

        x = v0x * time_points
        y = v0y * time_points - 0.5 * g * time_points**2

        # Cannonball can't go below ground
        y = np.maximum(y, 0)

        return x, y
    return (calculate_trajectory,)


@app.cell
def _(calculate_trajectory, cube_view, np, plt):
    # Read all values from cube_view.value (triggers marimo reactivity)
    axis_values = cube_view.value["axis_values"]
    locked_order = cube_view.value["locked_order"]

    angle = axis_values["x"]  # x = Angle
    force = axis_values["y"]  # y = Force
    current_time = axis_values["z"]  # z = Time

    # Get axis values from widget
    all_angles = cube_view.value["x_axis"]["values"]
    all_forces = [f for f in cube_view.value["y_axis"]["values"] if f > 0]  # Skip zero force

    fig, ax = plt.subplots(figsize=(8, 6))
    g = 9.8

    # Colors
    plane_color = 'steelblue'
    line_color = 'red'
    point_color = 'green'

    num_locked = len(locked_order)

    # VOLUME (nothing locked) - show all trajectories, no highlight
    if num_locked == 0:
        for a in all_angles:
            a_safe = max(a, 1)
            t_max = 2 * force * np.sin(np.radians(a_safe)) / g + 0.5
            t_full = np.linspace(0, max(t_max, 0.1), 100)
            x_traj, y_traj = calculate_trajectory(a, force, t_full)
            ax.plot(x_traj, y_traj, color=plane_color, alpha=0.4, linewidth=1)
        ax.set_title('Volume: all free')

    else:
        # At least one axis locked - ALWAYS draw plane background first
        first_locked = locked_order[0]

        # Draw PLANE background based on which axis was locked first
        if first_locked == "x":  # Angle locked - show all force trajectories
            for f in all_forces:
                a_safe = max(angle, 1)
                t_max = 2 * f * np.sin(np.radians(a_safe)) / g + 0.5
                t_full = np.linspace(0, max(t_max, 0.1), 100)
                x_traj, y_traj = calculate_trajectory(angle, f, t_full)
                ax.plot(x_traj, y_traj, color=plane_color, alpha=0.4, linewidth=1)
            plane_desc = f'Angle={angle:.0f}°'

        elif first_locked == "y":  # Force locked - show all angle trajectories
            for a in all_angles:
                a_safe = max(a, 1)
                t_max = 2 * force * np.sin(np.radians(a_safe)) / g + 0.5
                t_full = np.linspace(0, max(t_max, 0.1), 100)
                x_traj, y_traj = calculate_trajectory(a, force, t_full)
                ax.plot(x_traj, y_traj, color=plane_color, alpha=0.4, linewidth=1)
            plane_desc = f'Force={force:.0f}'

        else:  # Time locked (z) - show all angle/force combinations
            for a in all_angles:
                for f in all_forces:
                    x_pt, y_pt = calculate_trajectory(a, f, np.array([current_time]))
                    ax.plot(x_pt[0], y_pt[0], 'o', color=plane_color, alpha=0.4, markersize=4)
            plane_desc = f'Time={current_time:.1f}s'

        # Now add LINE highlight (if 2+ locked) or preview (if just 1 locked)
        locked_set = set(locked_order)

        if num_locked == 1:
            # Just plane - don't highlight anything specific
            # (the plane already shows all possibilities for the free dimensions)
            ax.set_title(f'Plane: {plane_desc}')

        elif num_locked == 2:
            # Line - determine which axis is still free
            free_axis = [a for a in ["x", "y", "z"] if a not in locked_set][0]

            if free_axis == "z":  # Time free - line is trajectory
                t_max = 2 * force * np.sin(np.radians(max(angle, 1))) / g + 0.5
                t_full = np.linspace(0, max(t_max, 0.1), 100)
                x_traj, y_traj = calculate_trajectory(angle, force, t_full)
                ax.plot(x_traj, y_traj, color=line_color, linewidth=2)
            elif free_axis == "x":  # Angle free - line is points at different angles
                for a in all_angles:
                    x_pt, y_pt = calculate_trajectory(a, force, np.array([current_time]))
                    ax.plot(x_pt[0], y_pt[0], 'o', color=line_color, alpha=0.8, markersize=6)
            else:  # Force free - line is points at different forces
                for f in all_forces:
                    x_pt, y_pt = calculate_trajectory(angle, f, np.array([current_time]))
                    ax.plot(x_pt[0], y_pt[0], 'o', color=line_color, alpha=0.8, markersize=6)
            ax.set_title(f'Line: {plane_desc} + ...')

        else:  # num_locked == 3 - Point
            # Determine what kind of "line" was shown before third lock
            first_two_locked = set(locked_order[:2])

            if "z" not in first_two_locked:
                # Time was free until last lock - line was a trajectory
                t_max = 2 * force * np.sin(np.radians(max(angle, 1))) / g + 0.5
                t_full = np.linspace(0, max(t_max, 0.1), 100)
                x_traj, y_traj = calculate_trajectory(angle, force, t_full)
                ax.plot(x_traj, y_traj, color=line_color, linewidth=2)
            elif "x" not in first_two_locked:
                # Angle was free until last lock - line was points at different angles
                for a in all_angles:
                    x_pt, y_pt = calculate_trajectory(a, force, np.array([current_time]))
                    ax.plot(x_pt[0], y_pt[0], 'o', color=line_color, alpha=0.6, markersize=6)
            else:
                # Force was free until last lock - line was points at different forces
                for f in all_forces:
                    x_pt, y_pt = calculate_trajectory(angle, f, np.array([current_time]))
                    ax.plot(x_pt[0], y_pt[0], 'o', color=line_color, alpha=0.6, markersize=6)

            # Mark the point
            x_now, y_now = calculate_trajectory(angle, force, np.array([current_time]))
            ax.plot(x_now[0], y_now[0], 'o', color=point_color, markersize=15,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            ax.set_title(f'Point: Angle={angle:.0f}°, Force={force:.0f}, t={current_time:.1f}s')

    ax.axhline(y=0, color='green', linewidth=2, alpha=0.5)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Height (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 300)

    plt.tight_layout()
    trajectory_chart = fig
    return (trajectory_chart,)


@app.cell
def _(cube_view):
    cube_view.value["plane"]
    return


@app.cell
def _(cube_view, mo, trajectory_chart):
    mo.hstack([cube_view, trajectory_chart], justify="start", gap=1)
    return


@app.cell
def _(cube_view):
    cube_view.value["axis_values"].get("x", 45)
    return


if __name__ == "__main__":
    app.run()
