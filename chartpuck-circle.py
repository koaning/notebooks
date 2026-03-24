# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "wigglystuff",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.rcParams["figure.dpi"] = 72
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck

    return ChartPuck, mo, np, plt


@app.cell
def _(ChartPuck, mo, np, plt):
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)

    def draw_circle(ax, widget):
        px, py = widget.x[0], widget.y[0]
        r = np.sqrt(px**2 + py**2)

        circle = plt.Circle((0, 0), r, fill=False, color="#e63946", linewidth=2)
        ax.add_patch(circle)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title("Complex Plane")

    puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_circle,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            figsize=(6, 6),
            x=2.0,
            y=0.0,
            puck_radius=6,
            throttle=100,
        )
    )
    return (puck,)


@app.cell
def _(mo, np, plt, puck):
    from mohtml import div 

    px, py = puck.x[0], puck.y[0]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    log_r = np.log(max(r, 1e-9))

    # The circle (all points at radius r) maps to a vertical line at x = ln(r)
    theta_range = np.linspace(-np.pi, np.pi, 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.full_like(theta_range, log_r), theta_range, color="#e63946", linewidth=2)
    ax.plot(log_r, theta, "o", color="#e63946", markersize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("ln(r)")
    ax.set_ylabel("θ")
    ax.set_title("Log Space")

    mo.hstack([puck, mo.vstack([div(style="margin-top: 21px;"), fig])], justify="start")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
