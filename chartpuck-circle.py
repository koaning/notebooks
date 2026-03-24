# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "wigglystuff",
#     "mohtml==0.1.11",
#     "pillow",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


app._unparsable_cell(
    r"""
    jimport marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.rcParams["figure.dpi"] = 72
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck
    """,
    name="_"
)


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
        ax.set_title("")

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
    ax.set_title("")

    mo.hstack([puck, mo.vstack([div(style="margin-top: 37px;"), fig])], justify="start")
    return (div,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But what if we apply a mapping of an entire image using this technique?
    """)
    return


@app.cell
def _(div, log_img, map_log_to_complex, mo, np, plt, puck):
    arr = np.array(log_img)
    mapped = map_log_to_complex(arr)

    puck_x, puck_y = puck.x[0], puck.y[0]
    puck_r = np.sqrt(puck_x**2 + puck_y**2)

    fig_mapped, ax_mapped = plt.subplots(figsize=(6, 6))
    ax_mapped.imshow(mapped, extent=(-5, 5, -5, 5), origin="lower")
    circle = plt.Circle((0, 0), puck_r, fill=False, color="#e63946", linewidth=2)
    ax_mapped.add_patch(circle)
    ax_mapped.plot(puck_x, puck_y, "o", color="#e63946", markersize=10, zorder=5)
    ax_mapped.set_xlim(-5, 5)
    ax_mapped.set_ylim(-5, 5)
    ax_mapped.set_aspect("equal")

    fig_log, ax_log = plt.subplots(figsize=(6, 6))
    ax_log.imshow(arr, extent=(-2, 2, -np.pi, np.pi), origin="lower", aspect="auto")
    log_puck_r = np.log(max(puck_r, 1e-9))
    angles = np.linspace(-np.pi, np.pi, 200)
    ax_log.plot(np.full_like(angles, log_puck_r), angles, color="#e63946", linewidth=2)
    puck_theta = np.arctan2(puck_y, puck_x)
    ax_log.plot(log_puck_r, puck_theta, "o", color="#e63946", markersize=10, zorder=5)
    ax_log.set_xlim(-2, 2)
    ax_log.set_ylim(-np.pi, np.pi)
    ax_log.set_xlabel("ln(r)")
    ax_log.set_ylabel("θ")

    mo.hstack([div(style="padding-left: 19px;"), fig_mapped, div(style="padding-left: 28px;"), mo.vstack([fig_log])], justify="start")
    return


@app.cell
def _():
    from PIL import Image, ImageDraw
    from scipy.ndimage import map_coordinates

    return Image, ImageDraw, map_coordinates


@app.cell
def _(Image, ImageDraw):
    def make_checkerboard(width=400, height=400, tile_size=50):
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                if (row // tile_size + col // tile_size) % 2 == 0:
                    draw.rectangle([col, row, col + tile_size, row + tile_size], fill="#1f77b4")
        return img

    log_img = make_checkerboard()
    return (log_img,)


@app.cell
def _(map_coordinates, np):
    def map_log_to_complex(img_arr, output_size=400, plane_bounds=(-5, 5), log_bounds=(-2, 2)):
        h, w, _ = img_arr.shape
        lin = np.linspace(plane_bounds[0], plane_bounds[1], output_size)
        gx, gy = np.meshgrid(lin, lin)

        radius = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        log_radius = np.log(np.clip(radius, 1e-9, None))

        px_x = (log_radius - log_bounds[0]) / (log_bounds[1] - log_bounds[0]) * w
        px_y = (angle - (-np.pi)) / (2 * np.pi) * h

        channels = []
        for i in range(3):
            ch = map_coordinates(img_arr[..., i], [px_y % h, px_x % w], order=1)
            channels.append(ch)
        return np.stack(channels, axis=-1)

    return (map_log_to_complex,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
