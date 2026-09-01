# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "traitlets",
#     "numpy",
#     "pillow",
#     "wigglystuff==0.3.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import math

    import marimo as mo
    import numpy as np
    from PIL import Image
    from wigglystuff import Paint

    from recursive_image_widget.recursive_image_widget import RecursiveImageWidget

    return Image, Paint, RecursiveImageWidget, base64, io, math, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Recursive images

    1. Draw a **parent** below (transparent PNG).
    2. Place one or more **children** — copies of the parent you can drag
       around. The set of children is the recursive rule.
    """)
    return


@app.cell
def _(Paint, mo):
    paint = mo.ui.anywidget(Paint(width=300, height=300, store_background=False))
    return (paint,)


@app.cell
def _(RecursiveImageWidget, mo):
    raw_rule = RecursiveImageWidget(size=300)
    rule = mo.ui.anywidget(raw_rule)
    return raw_rule, rule


@app.cell
def _(paint, raw_rule):
    # Feed the current Paint drawing into the rule widget as the parent image.
    raw_rule.parent_image = paint.get_base64()
    return


@app.cell
def _(mo, paint, rule):
    mo.hstack([paint, rule], justify="start", gap=1)
    return


@app.cell
def _(mo):
    depth = mo.ui.slider(0, 20, value=6, label="depth")
    background = mo.ui.dropdown(
        ["transparent", "white", "black"], value="transparent", label="background"
    )
    mo.hstack([depth, background], justify="start", gap=1)
    return background, depth


@app.cell(hide_code=True)
def _(Image, base64, io, math, np):
    # Must match BASE in recursive_image_widget.js (base-box world-width).
    BASE = 0.34

    def decode_png(data_url):
        if not data_url or "," not in data_url:
            return None
        raw = base64.b64decode(data_url.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    def inv_coeffs(scale, theta, c_out, c_in):
        # forward: out = c_out + scale * R(theta) * (in - c_in)
        # PIL wants the inverse map out -> in.
        inv = 1.0 / scale
        ct, st = math.cos(theta), math.sin(theta)
        a, b = inv * ct, inv * st
        d, e = -inv * st, inv * ct
        c = c_in[0] - (a * c_out[0] + b * c_out[1])
        f = c_in[1] - (d * c_out[0] + e * c_out[1])
        return (a, b, c, d, e, f)

    def warp(img, coeffs, W):
        return img.transform(
            (W, W), Image.AFFINE, coeffs, resample=Image.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )

    def seed_canvas(parent_img, parent, W):
        px, py, ps, pa = parent
        pw, ph = parent_img.size
        k = (BASE * ps * W) / pw
        if k <= 1e-9:
            return Image.new("RGBA", (W, W), (0, 0, 0, 0))
        coeffs = inv_coeffs(k, math.radians(pa), (px * W, py * W), (pw / 2, ph / 2))
        return warp(parent_img, coeffs, W)

    def render_ifs(parent_img, parent, children, depth, W=640):
        if parent_img is None:
            return None
        seed = seed_canvas(parent_img, parent, W)
        px, py, ps, pa = parent
        maps = []
        for cx, cy, cs, ca in children:
            if ps <= 1e-9 or cs <= 1e-9:
                continue
            maps.append(
                inv_coeffs(cs / ps, math.radians(ca - pa), (cx * W, cy * W), (px * W, py * W))
            )
        canvas = seed
        for _ in range(depth):
            if not maps:
                break
            acc = seed.copy()
            for coeffs in maps:
                acc = Image.alpha_composite(acc, warp(canvas, coeffs, W))
            canvas = acc
        return canvas

    def present(img, background="transparent", tile=20):
        W = img.size[0]
        if background == "white":
            bg = Image.new("RGBA", (W, W), (255, 255, 255, 255))
        elif background == "black":
            bg = Image.new("RGBA", (W, W), (0, 0, 0, 255))
        else:  # transparent -> checkerboard preview so alpha reads
            yy, xx = np.indices((W, W))
            cells = ((xx // tile) + (yy // tile)) % 2
            val = np.where(cells == 0, 245, 224).astype("uint8")
            bg = Image.fromarray(np.dstack([val, val, val])).convert("RGBA")
        out = Image.alpha_composite(bg, img)
        return out if background == "transparent" else out.convert("RGB")

    return decode_png, present, render_ifs


@app.cell
def _(background, decode_png, depth, mo, paint, present, render_ifs, rule):
    parent_img = decode_png(paint.get_base64())
    result = render_ifs(parent_img, rule.parent, rule.children, depth.value)
    mo.stop(result is None, mo.md("*Draw a parent to see the recursion.*"))
    present(result, background.value)
    return


if __name__ == "__main__":
    app.run()
